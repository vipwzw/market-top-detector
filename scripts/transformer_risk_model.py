#!/usr/bin/env python3
"""
Transformer 逃顶/抄底模型

核心思想: 每一天都是样本, 标签来自未来价格走势
  - 逃顶标签: 未来 180 天最大回撤 (越大越危险)
  - 抄底标签: 未来 180 天最大涨幅 (越大越值得买)

训练: 滚动窗口, 用过去 N 年训练, 预测下一段
评估: 预测值 vs 实际未来走势的相关性
"""
import os, functools, time, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings('ignore')
print = functools.partial(print, flush=True)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
IMG_DIR = '/Users/king/quant/market-top-detector/img'
os.makedirs(IMG_DIR, exist_ok=True)

# ==================== 配置 ====================
SEQ_LEN = 120        # 120 交易日 ≈ 半年历史
FWD_DAYS = 180       # 前瞻 180 天
D_MODEL = 64
NHEAD = 4
N_LAYERS = 3
DROPOUT = 0.1
BATCH = 128
EPOCHS = 30
LR = 1e-3

# ==================== 数据 ====================
print("1. 加载数据...")
df = pd.read_csv('/Users/king/quant/market-top-detector/data/BTC_daily_full.csv',
                  parse_dates=['Date'], index_col='Date')
df = df.sort_index().dropna(subset=['Close'])

fng = pd.read_csv('/Users/king/quant/market-top-detector/data/fear_greed_index.csv',
                   parse_dates=['Date'], index_col='Date')
fng['fng_value'] = pd.to_numeric(fng['fng_value'], errors='coerce')
fng = fng[~fng.index.duplicated(keep='first')].sort_index()
df = df.join(fng[['fng_value']], how='left')
df['fng_value'] = df['fng_value'].ffill()

gt = pd.read_csv('/Users/king/quant/market-top-detector/data/google_trends_bitcoin.csv',
                  parse_dates=['date'], index_col='date')
gt = gt.rename(columns={'Bitcoin': 'gtrend'})
gt = gt[~gt.index.duplicated(keep='first')].sort_index()
df = df.join(gt[['gtrend']], how='left')
df['gtrend'] = df['gtrend'].ffill()

print(f"   {len(df)} 天, {df.index[0].date()} ~ {df.index[-1].date()}")

# ==================== 特征工程 ====================
print("\n2. 计算特征...")
c = df['Close']

# 均线偏离
for w in [20, 50, 100, 200, 350]:
    df[f'ma{w}'] = c.rolling(w).mean()
    df[f'ma{w}_dev'] = (c - df[f'ma{w}']) / df[f'ma{w}']

df['mayer'] = c / df['ma200']

# RSI
def rsi(s, p):
    d = s.diff(); g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100 / (1 + g/(l+1e-10))
for w in [14, 30, 90]:
    df[f'rsi_{w}'] = rsi(c, w) / 100.0  # 归一化到 0~1

# 波动率
df['ret'] = c.pct_change()
for w in [30, 90]:
    df[f'vol_{w}d'] = df['ret'].rolling(w).std() * np.sqrt(365)

# 收益率
for w in [7, 30, 90, 180, 365]:
    df[f'ret_{w}d'] = c.pct_change(w)

# ATH 相关
df['ath'] = c.expanding().max()
df['ath_dd'] = (c - df['ath']) / df['ath']
df['days_since_ath'] = 0
last = 0
for i in range(len(df)):
    if c.iloc[i] >= df['ath'].iloc[i] * 0.999: last = i
    df.iloc[i, df.columns.get_loc('days_since_ath')] = (i - last) / 365.0  # 年化

# 对数去趋势
log_p = np.log(c.values)
x = np.arange(len(log_p))
valid = ~np.isnan(log_p)
coeffs = np.polyfit(x[valid], log_p[valid], 1)
df['log_detrend'] = log_p - np.polyval(coeffs, x)

# FNG 归一化
df['fng_norm'] = df['fng_value'] / 100.0
df['fng_ma30'] = df['fng_norm'].rolling(30, min_periods=5).mean()

# Google Trends 归一化
df['gt_norm'] = df['gtrend'] / 100.0

# Pi Cycle
df['ma111'] = c.rolling(111).mean()
df['ma350x2'] = c.rolling(350).mean() * 2
df['pi_ratio'] = df['ma111'] / (df['ma350x2'] + 1e-8)

FEATURE_COLS = [
    'ma20_dev', 'ma50_dev', 'ma100_dev', 'ma200_dev', 'ma350_dev',
    'mayer', 'pi_ratio',
    'rsi_14', 'rsi_30', 'rsi_90',
    'vol_30d', 'vol_90d',
    'ret_7d', 'ret_30d', 'ret_90d', 'ret_180d', 'ret_365d',
    'ath_dd', 'days_since_ath',
    'log_detrend',
    'fng_norm', 'fng_ma30', 'gt_norm',
]
N_FEAT = len(FEATURE_COLS)

# ==================== 构建标签 (每天的未来走势) ====================
print("\n3. 构建前瞻标签 (每天 → 未来 180 天走势)...")

close_arr = c.values
n = len(close_arr)

# 前瞻最大回撤 (从今天起, 未来FWD_DAYS内的最大跌幅)
fwd_max_dd = np.full(n, np.nan)
# 前瞻最大涨幅
fwd_max_rally = np.full(n, np.nan)
# 前瞻终点收益
fwd_endpoint = np.full(n, np.nan)

for i in range(n - FWD_DAYS):
    future = close_arr[i+1 : i+1+FWD_DAYS]
    cur = close_arr[i]
    fwd_rets = future / cur - 1
    fwd_max_dd[i] = np.min(fwd_rets)        # 最大回撤 (负值)
    fwd_max_rally[i] = np.max(fwd_rets)     # 最大涨幅 (正值)
    fwd_endpoint[i] = fwd_rets[-1]          # 终点收益

df['fwd_max_dd'] = fwd_max_dd
df['fwd_max_rally'] = fwd_max_rally
df['fwd_endpoint'] = fwd_endpoint

# 逃顶标签: 未来最大回撤越大 → 越危险 (0~1, 1=极危险)
# 映射: 回撤 0% → 0, 回撤 -50% → 0.8, 回撤 -80% → 1.0
df['label_top'] = (-df['fwd_max_dd']).clip(0, 1)  # 直接用绝对回撤幅度

# 抄底标签: 未来最大涨幅越大 → 越值得买 (0~1, 1=极好机会)
df['label_bot'] = df['fwd_max_rally'].clip(0, 2) / 2  # 涨幅 200% → 1.0

# 丢弃无效行
df_valid = df.dropna(subset=FEATURE_COLS + ['label_top', 'label_bot']).copy()
for col in FEATURE_COLS:
    df_valid[col] = df_valid[col].clip(-5, 5)

print(f"   有效样本: {len(df_valid)} 天")
print(f"   标签统计:")
print(f"     逃顶标签 (未来最大回撤): mean={df_valid['label_top'].mean():.3f}, "
      f"median={df_valid['label_top'].median():.3f}")
print(f"     抄底标签 (未来最大涨幅): mean={df_valid['label_bot'].mean():.3f}, "
      f"median={df_valid['label_bot'].median():.3f}")

# ==================== 模型 ====================
class PosEnc(nn.Module):
    def __init__(self, d, mx=500):
        super().__init__()
        pe = torch.zeros(mx, d); p = torch.arange(mx).unsqueeze(1).float()
        dv = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.) / d))
        pe[:, 0::2] = torch.sin(p * dv); pe[:, 1::2] = torch.cos(p * dv)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class RiskTransformer(nn.Module):
    def __init__(self, n_feat, d_model, nhead, n_layers, dropout):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        self.pe = PosEnc(d_model)
        el = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout,
                                        batch_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(el, n_layers)
        self.head_top = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1), nn.Sigmoid())
        self.head_bot = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1), nn.Sigmoid())

    def forward(self, x):
        h = self.enc(self.pe(self.proj(x)))[:, -1, :]
        return self.head_top(h).squeeze(-1), self.head_bot(h).squeeze(-1)

# ==================== 滚动训练 ====================
print(f"\n4. 滚动训练 (设备: {DEVICE})...")
print(f"   seq_len={SEQ_LEN}, d_model={D_MODEL}, layers={N_LAYERS}, fwd={FWD_DAYS}d")

feats_all = df_valid[FEATURE_COLS].values.astype(np.float32)
label_top_all = df_valid['label_top'].values.astype(np.float32)
label_bot_all = df_valid['label_bot'].values.astype(np.float32)
dates_all = df_valid.index

def build_windows(feats, sl):
    n = len(feats) - sl
    if n <= 0: return np.empty((0, sl, feats.shape[1]))
    shape = (n, sl, feats.shape[1])
    strides = (feats.strides[0], feats.strides[0], feats.strides[1])
    return np.lib.stride_tricks.as_strided(feats, shape=shape, strides=strides).copy()

# 滚动: 训练 2 年, 预测 1 年, 滑动 1 年
TRAIN_DAYS = 365 * 2
PRED_DAYS = 365
SLIDE_DAYS = 365

results = []
pred_top = np.full(len(df_valid), np.nan)
pred_bot = np.full(len(df_valid), np.nan)

idx = TRAIN_DAYS
window_n = 0

while idx + PRED_DAYS <= len(df_valid):
    t0 = time.time()
    window_n += 1

    # 训练集
    train_feats = feats_all[max(0, idx-TRAIN_DAYS):idx]
    train_top = label_top_all[max(0, idx-TRAIN_DAYS)+SEQ_LEN:idx]
    train_bot = label_bot_all[max(0, idx-TRAIN_DAYS)+SEQ_LEN:idx]

    X_train = build_windows(train_feats, SEQ_LEN)
    n_train = min(len(X_train), len(train_top))
    X_train = X_train[:n_train]
    train_top = train_top[:n_train]
    train_bot = train_bot[:n_train]

    # 预测集 (需要前 SEQ_LEN 天作为上下文)
    ctx_start = idx - SEQ_LEN
    pred_feats = feats_all[ctx_start : min(idx + PRED_DAYS, len(df_valid))]
    X_pred = build_windows(pred_feats, SEQ_LEN)

    pred_start = idx
    pred_end = min(idx + PRED_DAYS, len(df_valid))
    actual_pred_len = pred_end - pred_start

    train_date = dates_all[idx].strftime('%Y-%m-%d')
    pred_end_date = dates_all[min(pred_end-1, len(dates_all)-1)].strftime('%Y-%m-%d')

    # 训练
    model = RiskTransformer(N_FEAT, D_MODEL, NHEAD, N_LAYERS, DROPOUT).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    crit = nn.MSELoss()

    X_t = torch.FloatTensor(X_train)
    y_top_t = torch.FloatTensor(train_top)
    y_bot_t = torch.FloatTensor(train_bot)

    model.train()
    for epoch in range(EPOCHS):
        perm = torch.randperm(n_train)
        epoch_loss = 0; n_batch = 0
        for s in range(0, n_train - BATCH + 1, BATCH):
            bi = perm[s:s+BATCH]
            xb = X_t[bi].to(DEVICE)
            yt = y_top_t[bi].to(DEVICE)
            yb = y_bot_t[bi].to(DEVICE)
            opt.zero_grad()
            pt, pb = model(xb)
            loss = crit(pt, yt) + crit(pb, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item(); n_batch += 1
        scheduler.step()

    avg_loss = epoch_loss / max(n_batch, 1)

    # 预测
    model.eval()
    X_pt = torch.FloatTensor(X_pred)
    with torch.no_grad():
        all_pt = []; all_pb = []
        for s in range(0, len(X_pt), 512):
            e = min(s+512, len(X_pt))
            pt, pb = model(X_pt[s:e].to(DEVICE))
            all_pt.append(pt.cpu().numpy())
            all_pb.append(pb.cpu().numpy())
    p_top = np.concatenate(all_pt)
    p_bot = np.concatenate(all_pb)

    # 只取预测区间 (跳过上下文部分)
    offset = SEQ_LEN  # build_windows 产出的第 i 个 = feats[i:i+SEQ_LEN], 预测 feats[i+SEQ_LEN-1] 那天
    for j in range(actual_pred_len):
        global_idx = pred_start + j
        local_idx = j + offset  # 上下文偏移后
        if local_idx < len(p_top):
            pred_top[global_idx] = p_top[local_idx]
            pred_bot[global_idx] = p_bot[local_idx]

    # 评估: 预测 vs 实际的相关性
    actual_top_seg = label_top_all[pred_start:pred_end]
    actual_bot_seg = label_bot_all[pred_start:pred_end]
    pred_top_seg = pred_top[pred_start:pred_end]
    pred_bot_seg = pred_bot[pred_start:pred_end]

    valid_mask = ~np.isnan(pred_top_seg) & ~np.isnan(actual_top_seg)
    if valid_mask.sum() > 10:
        corr_top = np.corrcoef(pred_top_seg[valid_mask], actual_top_seg[valid_mask])[0, 1]
        corr_bot = np.corrcoef(pred_bot_seg[valid_mask], actual_bot_seg[valid_mask])[0, 1]
    else:
        corr_top = corr_bot = 0

    elapsed = time.time() - t0
    print(f"   窗口{window_n}: 训练→{train_date} 预测→{pred_end_date} | "
          f"loss={avg_loss:.4f} | 逃顶r={corr_top:.3f} 抄底r={corr_bot:.3f} | {elapsed:.0f}s")

    results.append({
        'window': window_n, 'train_end': train_date, 'pred_end': pred_end_date,
        'loss': avg_loss, 'corr_top': corr_top, 'corr_bot': corr_bot
    })

    del model, X_t, y_top_t, y_bot_t, X_pt
    if DEVICE == 'mps': torch.mps.empty_cache()
    gc.collect()

    idx += SLIDE_DAYS

# ==================== 最后一个窗口: 预测当前 ====================
print("\n5. 预测当前...")

# 用最后 TRAIN_DAYS 天训练, 预测最后时刻
train_start = max(0, len(df_valid) - TRAIN_DAYS - FWD_DAYS)
train_end = len(df_valid) - FWD_DAYS  # 确保标签有效

train_feats = feats_all[train_start:train_end]
train_top = label_top_all[train_start+SEQ_LEN:train_end]
train_bot = label_bot_all[train_start+SEQ_LEN:train_end]

X_train = build_windows(train_feats, SEQ_LEN)
n_train = min(len(X_train), len(train_top))
X_train = X_train[:n_train]; train_top = train_top[:n_train]; train_bot = train_bot[:n_train]

model = RiskTransformer(N_FEAT, D_MODEL, NHEAD, N_LAYERS, DROPOUT).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
crit = nn.MSELoss()

X_t = torch.FloatTensor(X_train)
y_top_t = torch.FloatTensor(train_top)
y_bot_t = torch.FloatTensor(train_bot)

model.train()
for epoch in range(EPOCHS):
    perm = torch.randperm(n_train)
    for s in range(0, n_train-BATCH+1, BATCH):
        bi = perm[s:s+BATCH]
        xb = X_t[bi].to(DEVICE)
        opt.zero_grad()
        pt, pb = model(xb)
        loss = crit(pt, y_top_t[bi].to(DEVICE)) + crit(pb, y_bot_t[bi].to(DEVICE))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    scheduler.step()

# 预测最后 SEQ_LEN 天 → 当前
model.eval()
last_seq = feats_all[-SEQ_LEN:].reshape(1, SEQ_LEN, N_FEAT)
with torch.no_grad():
    cur_top, cur_bot = model(torch.FloatTensor(last_seq).to(DEVICE))
    cur_top = cur_top.item()
    cur_bot = cur_bot.item()

cur_date = df_valid.index[-1]
cur_price = df_valid['Close'].iloc[-1]
print(f"\n   当前 ({cur_date.strftime('%Y-%m-%d')} ${cur_price:,.0f}):")
print(f"     逃顶预测: {cur_top:.3f} (未来180天最大回撤预期: {cur_top*100:.1f}%)")
print(f"     抄底预测: {cur_bot:.3f} (未来180天最大涨幅预期: {cur_bot*200:.0f}%)")

# 填充当前预测到 pred 数组
for i in range(max(0, len(df_valid)-FWD_DAYS), len(df_valid)):
    if np.isnan(pred_top[i]):
        # 用最后训练的模型预测这些天
        si = i - SEQ_LEN + 1
        if si >= 0:
            seq = feats_all[si:i+1].reshape(1, SEQ_LEN, N_FEAT)
            with torch.no_grad():
                pt, pb = model(torch.FloatTensor(seq).to(DEVICE))
            pred_top[i] = pt.item()
            pred_bot[i] = pb.item()

del model
if DEVICE == 'mps': torch.mps.empty_cache()

# ==================== 可视化 ====================
print("\n6. 生成图表...")

df_valid['pred_top'] = pred_top
df_valid['pred_bot'] = pred_bot
df_valid['pred_top_smooth'] = pd.Series(pred_top, index=df_valid.index).rolling(14, min_periods=1).mean()
df_valid['pred_bot_smooth'] = pd.Series(pred_bot, index=df_valid.index).rolling(14, min_periods=1).mean()

fig, axes = plt.subplots(4, 1, figsize=(30, 28),
                          gridspec_kw={'height_ratios': [4, 2, 2, 2]})
fig.subplots_adjust(hspace=0.08)

MAJOR_TOPS = [pd.Timestamp('2017-12-16'), pd.Timestamp('2019-06-26'), pd.Timestamp('2021-11-08')]
MAJOR_BOTTOMS = [pd.Timestamp('2018-12-15'), pd.Timestamp('2020-03-12'), pd.Timestamp('2022-11-21')]

# 面板1: 价格
ax1 = axes[0]
pt_s = df_valid['pred_top_smooth'].values
pb_s = df_valid['pred_bot_smooth'].values
for i in range(1, len(df_valid)):
    if pt_s[i] > 0.5:
        ax1.axvspan(df_valid.index[i-1], df_valid.index[i], alpha=min(0.5, pt_s[i]*0.6), color='red', lw=0)
    if pb_s[i] > 0.5:
        ax1.axvspan(df_valid.index[i-1], df_valid.index[i], alpha=min(0.5, pb_s[i]*0.5), color='green', lw=0)

ax1.plot(df_valid.index, df_valid['Close'], 'black', lw=0.8, alpha=0.7)
for top in MAJOR_TOPS:
    if top in df_valid.index or top <= df_valid.index[-1]:
        nearest = df_valid.index[df_valid.index.searchsorted(top)]
        p = df_valid.loc[nearest, 'Close']
        ax1.plot(nearest, p, 'o', color='red', ms=12, zorder=10, markeredgecolor='darkred', markeredgewidth=2)
for bot in MAJOR_BOTTOMS:
    if bot <= df_valid.index[-1]:
        nearest = df_valid.index[df_valid.index.searchsorted(bot)]
        p = df_valid.loc[nearest, 'Close']
        ax1.plot(nearest, p, 'o', color='green', ms=12, zorder=10, markeredgecolor='darkgreen', markeredgewidth=2)

ax1.set_yscale('log')
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax1.set_ylabel('BTC Price (log)', fontsize=12)
ax1.set_title(f'Transformer 逃顶/抄底模型 (每日样本, 标签=未来{FWD_DAYS}天走势)\n'
              f'红底=逃顶高危  绿底=抄底机会', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.15, which='both')
ax1.tick_params(labelbottom=False)

# 面板2: 逃顶预测 vs 实际
ax2 = axes[1]
ax2.fill_between(df_valid.index, pt_s * 100, 0, where=pt_s > 0.5, color='red', alpha=0.4, label='高危 (>50%)')
ax2.fill_between(df_valid.index, pt_s * 100, 0, where=pt_s <= 0.5, color='orange', alpha=0.2, label='正常 (<50%)')
ax2.plot(df_valid.index, pt_s * 100, 'black', lw=0.8)
ax2.plot(df_valid.index, df_valid['label_top'] * 100, 'r-', lw=0.3, alpha=0.3, label='实际回撤')
ax2.set_ylabel('逃顶指数 (%)', fontsize=12)
ax2.set_ylim(0, 100)
cur_top_s = df_valid['pred_top_smooth'].iloc[-1]
ax2.annotate(f'当前: {cur_top_s*100:.0f}%', xy=(cur_date, cur_top_s*100),
             xytext=(-80, 10), textcoords='offset points', fontsize=11, fontweight='bold',
             color='darkred', arrowprops=dict(arrowstyle='->', color='darkred'),
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9))
ax2.legend(fontsize=8, ncol=3, loc='upper left')
ax2.grid(True, alpha=0.2); ax2.tick_params(labelbottom=False)

# 面板3: 抄底预测 vs 实际
ax3 = axes[2]
ax3.fill_between(df_valid.index, pb_s * 100, 0, where=pb_s > 0.5, color='green', alpha=0.4, label='机会 (>50%)')
ax3.fill_between(df_valid.index, pb_s * 100, 0, where=pb_s <= 0.5, color='lightgreen', alpha=0.2, label='正常 (<50%)')
ax3.plot(df_valid.index, pb_s * 100, 'black', lw=0.8)
ax3.plot(df_valid.index, df_valid['label_bot'] * 100, 'g-', lw=0.3, alpha=0.3, label='实际涨幅')
ax3.set_ylabel('抄底指数 (%)', fontsize=12)
ax3.set_ylim(0, 100)
cur_bot_s = df_valid['pred_bot_smooth'].iloc[-1]
ax3.annotate(f'当前: {cur_bot_s*100:.0f}%', xy=(cur_date, cur_bot_s*100),
             xytext=(-80, 10), textcoords='offset points', fontsize=11, fontweight='bold',
             color='darkgreen', arrowprops=dict(arrowstyle='->', color='darkgreen'),
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
ax3.legend(fontsize=8, ncol=3, loc='upper left')
ax3.grid(True, alpha=0.2); ax3.tick_params(labelbottom=False)

# 面板4: 综合信号 (抄底 - 逃顶)
ax4 = axes[3]
combined = pb_s - pt_s  # 正=抄底机会, 负=逃顶风险
ax4.fill_between(df_valid.index, combined * 100, 0,
                 where=combined > 0, color='green', alpha=0.4, label='抄底 > 逃顶')
ax4.fill_between(df_valid.index, combined * 100, 0,
                 where=combined <= 0, color='red', alpha=0.4, label='逃顶 > 抄底')
ax4.plot(df_valid.index, combined * 100, 'black', lw=0.8)
ax4.axhline(0, color='black', lw=0.5)
ax4.set_ylabel('综合信号 (%)', fontsize=12)
ax4.legend(fontsize=8, loc='upper left')
ax4.grid(True, alpha=0.2)

plt.savefig(f'{IMG_DIR}/transformer_risk_model.png', dpi=150, bbox_inches='tight')
print(f"   保存: {IMG_DIR}/transformer_risk_model.png")

# ==================== 汇总 ====================
print("\n" + "=" * 80)
print("7. 模型评估汇总")
print("=" * 80)

rdf = pd.DataFrame(results)
print(f"\n  滚动窗口数: {len(rdf)}")
print(f"  逃顶相关性: mean={rdf['corr_top'].mean():.3f}, min={rdf['corr_top'].min():.3f}, max={rdf['corr_top'].max():.3f}")
print(f"  抄底相关性: mean={rdf['corr_bot'].mean():.3f}, min={rdf['corr_bot'].min():.3f}, max={rdf['corr_bot'].max():.3f}")
print(f"\n  当前预测 ({cur_date.strftime('%Y-%m-%d')} ${cur_price:,.0f}):")
print(f"    逃顶: {cur_top*100:.1f}%  (解读: 未来180天最大回撤预期 ≈ {cur_top*100:.0f}%)")
print(f"    抄底: {cur_bot*100:.1f}%  (解读: 未来180天最大涨幅预期 ≈ {cur_bot*200:.0f}%)")

if cur_top > 0.5 and cur_bot < 0.3:
    verdict = "高危! 模型预测未来有大幅回撤风险"
elif cur_bot > 0.5 and cur_top < 0.3:
    verdict = "机会! 模型预测未来有大幅上涨空间"
elif cur_bot > cur_top:
    verdict = "偏乐观, 上涨空间 > 下跌风险"
else:
    verdict = "偏谨慎, 下跌风险 > 上涨空间"

print(f"    综合: {verdict}")
print("\n完成!")
