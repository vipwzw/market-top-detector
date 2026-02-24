#!/usr/bin/env python3
"""
多资产 Transformer 逃顶/抄底模型

核心改进: 用 22 个加密货币的日线数据联合训练,
         大幅增加样本量和牛熊周期覆盖, 提高泛化能力.

币种: BTC, ETH, XRP, BNB, SOL, TRX, DOGE, BCH, ADA, LEO, XMR, LINK,
      LTC, XLM, HBAR, ZEC, AVAX, SHIB, CRO, UNI, DOT, TON

排除: USDT, USDC, DAI, PAXG (稳定币), WBTC (=BTC), MNT/HYPE/SUI (太短)
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
DATA_DIR = '/Users/king/quant/market-top-detector/data'
IMG_DIR = '/Users/king/quant/market-top-detector/img'
os.makedirs(IMG_DIR, exist_ok=True)

# ==================== 配置 ====================
SEQ_LEN = 120        # 120 交易日 ≈ 半年历史
FWD_DAYS = 180       # 前瞻 180 天
D_MODEL = 64
NHEAD = 4
N_LAYERS = 3
DROPOUT = 0.15
BATCH = 256
EPOCHS = 40
LR = 5e-4

COINS = [
    'BTC', 'ETH', 'XRP', 'BNB', 'SOL', 'TRX', 'DOGE', 'BCH', 'ADA',
    'LEO', 'XMR', 'LINK', 'LTC', 'XLM', 'HBAR', 'ZEC', 'AVAX', 'SHIB',
    'CRO', 'UNI', 'DOT', 'TON',
]

MIN_HISTORY = 365 * 2 + FWD_DAYS  # 至少 2 年 + 前瞻期

FEATURE_COLS = [
    'ma20_dev', 'ma50_dev', 'ma100_dev', 'ma200_dev',
    'mayer',
    'rsi_14', 'rsi_30',
    'vol_30d', 'vol_90d',
    'ret_7d', 'ret_30d', 'ret_90d', 'ret_180d', 'ret_365d',
    'ath_dd', 'days_since_ath',
    'log_detrend',
    'fng_norm', 'fng_ma30', 'gt_norm',
]
N_FEAT = len(FEATURE_COLS)

# ==================== 工具函数 ====================
def rsi(s, p):
    d = s.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100 / (1 + g / (l + 1e-10))


def compute_features(df):
    """为单个币种计算全部特征 (df 需含 Close 列)"""
    c = df['Close']

    for w in [20, 50, 100, 200]:
        df[f'ma{w}'] = c.rolling(w).mean()
        df[f'ma{w}_dev'] = (c - df[f'ma{w}']) / df[f'ma{w}']

    df['mayer'] = c / df['ma200']

    for w in [14, 30]:
        df[f'rsi_{w}'] = rsi(c, w) / 100.0

    df['ret'] = c.pct_change()
    for w in [30, 90]:
        df[f'vol_{w}d'] = df['ret'].rolling(w).std() * np.sqrt(365)

    for w in [7, 30, 90, 180, 365]:
        df[f'ret_{w}d'] = c.pct_change(w)

    df['ath'] = c.expanding().max()
    df['ath_dd'] = (c - df['ath']) / df['ath']

    df['days_since_ath'] = 0.0
    last_ath_idx = 0
    close_vals = c.values
    ath_vals = df['ath'].values
    dsa = np.zeros(len(df))
    for i in range(len(df)):
        if close_vals[i] >= ath_vals[i] * 0.999:
            last_ath_idx = i
        dsa[i] = (i - last_ath_idx) / 365.0
    df['days_since_ath'] = dsa

    log_p = np.log(c.values)
    x = np.arange(len(log_p))
    valid = ~np.isnan(log_p)
    if valid.sum() > 10:
        coeffs = np.polyfit(x[valid], log_p[valid], 1)
        df['log_detrend'] = log_p - np.polyval(coeffs, x)
    else:
        df['log_detrend'] = 0.0

    return df


def compute_labels(df):
    """计算前瞻标签: 未来 FWD_DAYS 天最大回撤/涨幅"""
    close_arr = df['Close'].values
    n = len(close_arr)
    fwd_max_dd = np.full(n, np.nan)
    fwd_max_rally = np.full(n, np.nan)

    for i in range(n - FWD_DAYS):
        future = close_arr[i+1 : i+1+FWD_DAYS]
        cur = close_arr[i]
        fwd_rets = future / cur - 1
        fwd_max_dd[i] = np.min(fwd_rets)
        fwd_max_rally[i] = np.max(fwd_rets)

    df['label_top'] = (-np.array(fwd_max_dd)).clip(0, 1)
    df['label_bot'] = np.array(fwd_max_rally).clip(0, 2) / 2.0
    return df


# ==================== 加载数据 ====================
print("=" * 80)
print("多资产 Transformer 逃顶/抄底模型")
print("=" * 80)

print("\n1. 加载情绪数据...")
fng = pd.read_csv(f'{DATA_DIR}/fear_greed_index.csv', parse_dates=['Date'], index_col='Date')
fng['fng_value'] = pd.to_numeric(fng['fng_value'], errors='coerce')
fng = fng[~fng.index.duplicated(keep='first')].sort_index()

gt = pd.read_csv(f'{DATA_DIR}/google_trends_bitcoin.csv', parse_dates=['date'], index_col='date')
gt = gt.rename(columns={'Bitcoin': 'gtrend'})
gt = gt[~gt.index.duplicated(keep='first')].sort_index()

print("\n2. 逐币种加载 + 特征工程 + 标签...")
coin_data = {}
total_samples = 0

for coin in COINS:
    fname = f'{DATA_DIR}/{coin}_daily.csv'
    if not os.path.exists(fname):
        if coin == 'BTC':
            fname = f'{DATA_DIR}/BTC_daily_full.csv'
        else:
            print(f"   {coin:>5}: 文件不存在, 跳过")
            continue

    df = pd.read_csv(fname, parse_dates=[0], index_col=0)
    df = df.sort_index().dropna(subset=['Close'])

    if len(df) < MIN_HISTORY:
        print(f"   {coin:>5}: 仅 {len(df)} 天 (需 {MIN_HISTORY}), 跳过")
        continue

    df = df.join(fng[['fng_value']], how='left')
    df['fng_value'] = df['fng_value'].ffill().bfill()
    df['fng_norm'] = df['fng_value'] / 100.0
    df['fng_ma30'] = df['fng_norm'].rolling(30, min_periods=5).mean()

    df = df.join(gt[['gtrend']], how='left')
    df['gtrend'] = df['gtrend'].ffill().bfill()
    df['gt_norm'] = df['gtrend'] / 100.0

    df = compute_features(df)
    df = compute_labels(df)

    df_valid = df.dropna(subset=FEATURE_COLS + ['label_top', 'label_bot']).copy()
    for col in FEATURE_COLS:
        df_valid[col] = df_valid[col].clip(-5, 5)

    if len(df_valid) < SEQ_LEN + 100:
        print(f"   {coin:>5}: 有效样本仅 {len(df_valid)}, 跳过")
        continue

    coin_data[coin] = df_valid
    total_samples += len(df_valid)
    print(f"   {coin:>5}: {len(df_valid):>5} 天  "
          f"{df_valid.index[0].date()} ~ {df_valid.index[-1].date()}  "
          f"top_label={df_valid['label_top'].mean():.3f}  bot_label={df_valid['label_bot'].mean():.3f}")

print(f"\n   总计: {len(coin_data)} 个币种, {total_samples:,} 天样本")

# ==================== 模型 ====================
class PosEnc(nn.Module):
    def __init__(self, d, mx=500):
        super().__init__()
        pe = torch.zeros(mx, d)
        p = torch.arange(mx).unsqueeze(1).float()
        dv = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.) / d))
        pe[:, 0::2] = torch.sin(p * dv)
        pe[:, 1::2] = torch.cos(p * dv)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class RiskTransformer(nn.Module):
    def __init__(self, n_feat, d_model, nhead, n_layers, dropout):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        self.pe = PosEnc(d_model)
        el = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4, dropout,
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


# ==================== 构建滑动窗口序列 ====================
def build_windows(feats, sl):
    n = len(feats) - sl
    if n <= 0:
        return np.empty((0, sl, feats.shape[1]))
    shape = (n, sl, feats.shape[1])
    strides = (feats.strides[0], feats.strides[0], feats.strides[1])
    return np.lib.stride_tricks.as_strided(feats, shape=shape, strides=strides).copy()


# ==================== 按时间切分, 多资产联合训练 ====================
print(f"\n3. 滚动训练 (设备: {DEVICE})")
print(f"   seq={SEQ_LEN}, d_model={D_MODEL}, layers={N_LAYERS}, fwd={FWD_DAYS}d, coins={len(coin_data)}")

TRAIN_YEARS = 2
PRED_YEARS = 1
SLIDE_YEARS = 1

all_dates = sorted(set().union(*[set(v.index) for v in coin_data.values()]))
date_range = pd.DatetimeIndex(all_dates)
min_date = date_range.min()
max_date = date_range.max()
print(f"   时间跨度: {min_date.date()} ~ {max_date.date()}")

btc_df = coin_data['BTC']
btc_pred_top = np.full(len(btc_df), np.nan)
btc_pred_bot = np.full(len(btc_df), np.nan)

results = []
window_n = 0

train_start_date = min_date
train_end_date = train_start_date + pd.DateOffset(years=TRAIN_YEARS)

while train_end_date + pd.DateOffset(years=PRED_YEARS) <= max_date + pd.Timedelta(days=30):
    t0 = time.time()
    window_n += 1

    pred_end_date = min(train_end_date + pd.DateOffset(years=PRED_YEARS), max_date)

    # 从所有币种收集训练样本
    X_train_all = []
    y_top_train_all = []
    y_bot_train_all = []
    n_coins_used = 0

    for coin, df in coin_data.items():
        mask_train = (df.index >= train_start_date) & (df.index < train_end_date)
        df_train = df.loc[mask_train]

        if len(df_train) < SEQ_LEN + 50:
            continue

        feats = df_train[FEATURE_COLS].values.astype(np.float32)
        labels_top = df_train['label_top'].values.astype(np.float32)
        labels_bot = df_train['label_bot'].values.astype(np.float32)

        X_w = build_windows(feats, SEQ_LEN)
        n_w = min(len(X_w), len(labels_top) - SEQ_LEN)
        if n_w <= 0:
            continue

        X_w = X_w[:n_w]
        y_t = labels_top[SEQ_LEN:SEQ_LEN + n_w]
        y_b = labels_bot[SEQ_LEN:SEQ_LEN + n_w]

        X_train_all.append(X_w)
        y_top_train_all.append(y_t)
        y_bot_train_all.append(y_b)
        n_coins_used += 1

    if not X_train_all:
        train_start_date += pd.DateOffset(years=SLIDE_YEARS)
        train_end_date = train_start_date + pd.DateOffset(years=TRAIN_YEARS)
        continue

    X_train = np.concatenate(X_train_all)
    y_top_train = np.concatenate(y_top_train_all)
    y_bot_train = np.concatenate(y_bot_train_all)
    n_train = len(X_train)

    # 随机打乱 (跨币种混合)
    perm = np.random.permutation(n_train)
    X_train = X_train[perm]
    y_top_train = y_top_train[perm]
    y_bot_train = y_bot_train[perm]

    # 训练
    model = RiskTransformer(N_FEAT, D_MODEL, NHEAD, N_LAYERS, DROPOUT).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    crit = nn.MSELoss()

    X_t = torch.FloatTensor(X_train)
    y_top_t = torch.FloatTensor(y_top_train)
    y_bot_t = torch.FloatTensor(y_bot_train)

    model.train()
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        perm_e = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batch = 0
        for s in range(0, n_train - BATCH + 1, BATCH):
            bi = perm_e[s:s+BATCH]
            xb = X_t[bi].to(DEVICE)
            yt = y_top_t[bi].to(DEVICE)
            yb = y_bot_t[bi].to(DEVICE)
            opt.zero_grad()
            pt, pb = model(xb)
            loss = crit(pt, yt) + crit(pb, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
            n_batch += 1
        scheduler.step()
        avg_loss = epoch_loss / max(n_batch, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    # BTC 预测区间评估
    btc_mask_pred = (btc_df.index >= train_end_date) & (btc_df.index < pred_end_date)

    ctx_start = btc_df.index.searchsorted(train_end_date) - SEQ_LEN
    ctx_start = max(0, ctx_start)
    pred_end_idx = btc_df.index.searchsorted(pred_end_date)
    if pred_end_idx > len(btc_df):
        pred_end_idx = len(btc_df)

    btc_feats_slice = btc_df[FEATURE_COLS].values[ctx_start:pred_end_idx].astype(np.float32)
    X_pred = build_windows(btc_feats_slice, SEQ_LEN)

    if len(X_pred) > 0:
        model.eval()
        X_pt = torch.FloatTensor(X_pred)
        with torch.no_grad():
            all_pt = []
            all_pb = []
            for s in range(0, len(X_pt), 512):
                e = min(s + 512, len(X_pt))
                pt, pb = model(X_pt[s:e].to(DEVICE))
                all_pt.append(pt.cpu().numpy())
                all_pb.append(pb.cpu().numpy())
        p_top = np.concatenate(all_pt)
        p_bot = np.concatenate(all_pb)

        # window[k] 使用 feats[ctx_start+k : ctx_start+k+SEQ_LEN],
        # 对应 btc_df 第 ctx_start+k+SEQ_LEN 天的预测
        pred_global_start = btc_df.index.searchsorted(train_end_date)
        n_fill = min(pred_end_idx - pred_global_start, len(p_top))
        for j in range(n_fill):
            gi = pred_global_start + j
            if gi < len(btc_pred_top) and j < len(p_top):
                btc_pred_top[gi] = p_top[j]
                btc_pred_bot[gi] = p_bot[j]

    # 评估相关性 (BTC)
    pred_start_idx = btc_df.index.searchsorted(train_end_date)
    pred_end_idx2 = btc_df.index.searchsorted(pred_end_date)
    if pred_end_idx2 > len(btc_df):
        pred_end_idx2 = len(btc_df)

    actual_top_seg = btc_df['label_top'].values[pred_start_idx:pred_end_idx2]
    actual_bot_seg = btc_df['label_bot'].values[pred_start_idx:pred_end_idx2]
    pred_top_seg = btc_pred_top[pred_start_idx:pred_end_idx2]
    pred_bot_seg = btc_pred_bot[pred_start_idx:pred_end_idx2]

    valid_mask = ~np.isnan(pred_top_seg) & ~np.isnan(actual_top_seg)
    if valid_mask.sum() > 10:
        corr_top = np.corrcoef(pred_top_seg[valid_mask], actual_top_seg[valid_mask])[0, 1]
        corr_bot = np.corrcoef(pred_bot_seg[valid_mask], actual_bot_seg[valid_mask])[0, 1]
    else:
        corr_top = corr_bot = float('nan')

    # 多币种预测区间评估
    multi_corr_top = []
    multi_corr_bot = []
    for coin, df in coin_data.items():
        mask_pred = (df.index >= train_end_date) & (df.index < pred_end_date)
        df_pred = df.loc[mask_pred]
        if len(df_pred) < SEQ_LEN + 30:
            continue

        ci = df.index.searchsorted(train_end_date) - SEQ_LEN
        ci = max(0, ci)
        ei = df.index.searchsorted(pred_end_date)
        if ei > len(df):
            ei = len(df)

        feats_s = df[FEATURE_COLS].values[ci:ei].astype(np.float32)
        Xp = build_windows(feats_s, SEQ_LEN)
        if len(Xp) == 0:
            continue

        model.eval()
        with torch.no_grad():
            apt = []
            apb = []
            for s in range(0, len(Xp), 512):
                e = min(s + 512, len(Xp))
                ppt, ppb = model(torch.FloatTensor(Xp[s:e]).to(DEVICE))
                apt.append(ppt.cpu().numpy())
                apb.append(ppb.cpu().numpy())
        cpt = np.concatenate(apt)
        cpb = np.concatenate(apb)

        ps_idx = df.index.searchsorted(train_end_date)
        n_pred = min(ei - ps_idx, len(cpt))
        if n_pred <= 0:
            continue

        at = df['label_top'].values[ps_idx:ps_idx + n_pred]
        ab = df['label_bot'].values[ps_idx:ps_idx + n_pred]
        ppt_s = cpt[:n_pred]
        ppb_s = cpb[:n_pred]

        vm = ~np.isnan(ppt_s) & ~np.isnan(at)
        if vm.sum() > 10:
            multi_corr_top.append(np.corrcoef(ppt_s[vm], at[vm])[0, 1])
            multi_corr_bot.append(np.corrcoef(ppb_s[vm], ab[vm])[0, 1])

    avg_multi_corr_top = np.nanmean(multi_corr_top) if multi_corr_top else float('nan')
    avg_multi_corr_bot = np.nanmean(multi_corr_bot) if multi_corr_bot else float('nan')

    elapsed = time.time() - t0
    print(f"   窗口{window_n}: 训练 {train_start_date.date()}~{train_end_date.date()} "
          f"→ 预测~{pred_end_date.date()} | {n_coins_used}币 {n_train:,}样本 | "
          f"loss={best_loss:.4f} | BTC逃顶r={corr_top:.3f} 抄底r={corr_bot:.3f} | "
          f"全币平均r: top={avg_multi_corr_top:.3f} bot={avg_multi_corr_bot:.3f} | {elapsed:.0f}s")

    results.append({
        'window': window_n,
        'train_period': f'{train_start_date.date()}~{train_end_date.date()}',
        'pred_period': f'{train_end_date.date()}~{pred_end_date.date()}',
        'n_coins': n_coins_used, 'n_samples': n_train,
        'loss': best_loss,
        'btc_corr_top': corr_top, 'btc_corr_bot': corr_bot,
        'multi_corr_top': avg_multi_corr_top, 'multi_corr_bot': avg_multi_corr_bot,
    })

    del model, X_t, y_top_t, y_bot_t
    if DEVICE == 'mps':
        torch.mps.empty_cache()
    gc.collect()

    train_start_date += pd.DateOffset(years=SLIDE_YEARS)
    train_end_date = train_start_date + pd.DateOffset(years=TRAIN_YEARS)

# ==================== 最终模型: 用全部数据训练, 预测当前 ====================
print("\n4. 训练最终模型 (全量数据) → 预测当前 BTC...")

cutoff_date = max_date - pd.Timedelta(days=FWD_DAYS)
X_final_all = []
y_top_final_all = []
y_bot_final_all = []

for coin, df in coin_data.items():
    mask = df.index <= cutoff_date
    df_t = df.loc[mask]
    if len(df_t) < SEQ_LEN + 50:
        continue

    feats = df_t[FEATURE_COLS].values.astype(np.float32)
    lt = df_t['label_top'].values.astype(np.float32)
    lb = df_t['label_bot'].values.astype(np.float32)

    Xw = build_windows(feats, SEQ_LEN)
    nw = min(len(Xw), len(lt) - SEQ_LEN)
    if nw <= 0:
        continue

    X_final_all.append(Xw[:nw])
    y_top_final_all.append(lt[SEQ_LEN:SEQ_LEN + nw])
    y_bot_final_all.append(lb[SEQ_LEN:SEQ_LEN + nw])

X_final = np.concatenate(X_final_all)
y_top_final = np.concatenate(y_top_final_all)
y_bot_final = np.concatenate(y_bot_final_all)

valid_mask = ~np.isnan(y_top_final) & ~np.isnan(y_bot_final)
X_final = X_final[valid_mask]
y_top_final = y_top_final[valid_mask]
y_bot_final = y_bot_final[valid_mask]

n_final = len(X_final)
print(f"   最终训练集: {n_final:,} 样本 (截止 {cutoff_date.date()})")

perm = np.random.permutation(n_final)
X_final = X_final[perm]
y_top_final = y_top_final[perm]
y_bot_final = y_bot_final[perm]

model = RiskTransformer(N_FEAT, D_MODEL, NHEAD, N_LAYERS, DROPOUT).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
crit = nn.MSELoss()

X_t = torch.FloatTensor(X_final)
y_top_t = torch.FloatTensor(y_top_final)
y_bot_t = torch.FloatTensor(y_bot_final)

model.train()
best_loss = float('inf')
for epoch in range(EPOCHS):
    perm_e = torch.randperm(n_final)
    epoch_loss = 0.0
    n_batch = 0
    for s in range(0, n_final - BATCH + 1, BATCH):
        bi = perm_e[s:s+BATCH]
        xb = X_t[bi].to(DEVICE)
        yt = y_top_t[bi].to(DEVICE)
        yb = y_bot_t[bi].to(DEVICE)
        opt.zero_grad()
        pt, pb = model(xb)
        loss = crit(pt, yt) + crit(pb, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        epoch_loss += loss.item()
        n_batch += 1
    scheduler.step()
    avg_loss = epoch_loss / max(n_batch, 1)
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if (epoch + 1) % 10 == 0:
        print(f"     epoch {epoch+1}/{EPOCHS}: loss={avg_loss:.5f}")

model.load_state_dict(best_state)
print(f"   最终模型 best loss: {best_loss:.5f}")

# BTC 全量预测
model.eval()
btc_feats = btc_df[FEATURE_COLS].values.astype(np.float32)
X_btc = build_windows(btc_feats, SEQ_LEN)

with torch.no_grad():
    all_pt = []
    all_pb = []
    for s in range(0, len(X_btc), 512):
        e = min(s + 512, len(X_btc))
        pt, pb = model(torch.FloatTensor(X_btc[s:e]).to(DEVICE))
        all_pt.append(pt.cpu().numpy())
        all_pb.append(pb.cpu().numpy())

final_pred_top = np.concatenate(all_pt)
final_pred_bot = np.concatenate(all_pb)

# 当前预测
cur_date = btc_df.index[-1]
cur_price = btc_df['Close'].iloc[-1]
cur_top = final_pred_top[-1]
cur_bot = final_pred_bot[-1]

print(f"\n   当前 BTC ({cur_date.strftime('%Y-%m-%d')} ${cur_price:,.0f}):")
print(f"     逃顶: {cur_top:.3f} → 未来180天最大回撤预期 ≈ {cur_top*100:.1f}%")
print(f"     抄底: {cur_bot:.3f} → 未来180天最大涨幅预期 ≈ {cur_bot*200:.0f}%")

# 填充到 btc_pred 数组用于和滚动结果对比
for i in range(len(btc_df)):
    if np.isnan(btc_pred_top[i]) and i < len(final_pred_top):
        btc_pred_top[i] = final_pred_top[i]
        btc_pred_bot[i] = final_pred_bot[i]

del model
if DEVICE == 'mps':
    torch.mps.empty_cache()

# ==================== 可视化 ====================
print("\n5. 生成图表...")

btc_df['pred_top'] = btc_pred_top
btc_df['pred_bot'] = btc_pred_bot
btc_df['final_top'] = np.nan
btc_df['final_bot'] = np.nan
btc_df.iloc[SEQ_LEN:SEQ_LEN + len(final_pred_top),
            btc_df.columns.get_loc('final_top')] = final_pred_top
btc_df.iloc[SEQ_LEN:SEQ_LEN + len(final_pred_bot),
            btc_df.columns.get_loc('final_bot')] = final_pred_bot

sm = 14
btc_df['final_top_s'] = btc_df['final_top'].rolling(sm, min_periods=1).mean()
btc_df['final_bot_s'] = btc_df['final_bot'].rolling(sm, min_periods=1).mean()

MAJOR_TOPS = [pd.Timestamp('2017-12-16'), pd.Timestamp('2019-06-26'), pd.Timestamp('2021-11-08')]
MAJOR_BOTTOMS = [pd.Timestamp('2018-12-15'), pd.Timestamp('2020-03-12'), pd.Timestamp('2022-11-21')]

fig, axes = plt.subplots(4, 1, figsize=(30, 28),
                         gridspec_kw={'height_ratios': [4, 2, 2, 2]})
fig.subplots_adjust(hspace=0.08)

# 面板 1: BTC 价格
ax1 = axes[0]
pt_s = btc_df['final_top_s'].values
pb_s = btc_df['final_bot_s'].values
for i in range(1, len(btc_df)):
    if not np.isnan(pt_s[i]) and pt_s[i] > 0.5:
        ax1.axvspan(btc_df.index[i-1], btc_df.index[i],
                    alpha=min(0.5, pt_s[i] * 0.6), color='red', lw=0)
    if not np.isnan(pb_s[i]) and pb_s[i] > 0.5:
        ax1.axvspan(btc_df.index[i-1], btc_df.index[i],
                    alpha=min(0.5, pb_s[i] * 0.5), color='green', lw=0)

ax1.plot(btc_df.index, btc_df['Close'], 'black', lw=0.8, alpha=0.7)
for top in MAJOR_TOPS:
    if top <= btc_df.index[-1]:
        nearest = btc_df.index[btc_df.index.searchsorted(top)]
        ax1.plot(nearest, btc_df.loc[nearest, 'Close'], 'o', color='red', ms=12,
                 zorder=10, markeredgecolor='darkred', markeredgewidth=2)
for bot in MAJOR_BOTTOMS:
    if bot <= btc_df.index[-1]:
        nearest = btc_df.index[btc_df.index.searchsorted(bot)]
        ax1.plot(nearest, btc_df.loc[nearest, 'Close'], 'o', color='green', ms=12,
                 zorder=10, markeredgecolor='darkgreen', markeredgewidth=2)

ax1.set_yscale('log')
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax1.set_ylabel('BTC Price (log)', fontsize=12)
ax1.set_title(f'多资产 Transformer 逃顶/抄底模型 ({len(coin_data)}币种联合训练)\n'
              f'红底=逃顶高危  绿底=抄底机会  (前瞻{FWD_DAYS}天)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.15, which='both')
ax1.tick_params(labelbottom=False)

# 面板 2: 逃顶指数
ax2 = axes[1]
valid_idx = ~np.isnan(pt_s)
pts = np.where(valid_idx, pt_s, 0) * 100
ax2.fill_between(btc_df.index, pts, 0, where=pts > 50, color='red', alpha=0.4, label='高危 (>50%)')
ax2.fill_between(btc_df.index, pts, 0, where=pts <= 50, color='orange', alpha=0.2, label='正常 (<50%)')
ax2.plot(btc_df.index, pts, 'black', lw=0.8)
actual_top = btc_df['label_top'].values * 100
ax2.plot(btc_df.index, actual_top, 'r-', lw=0.3, alpha=0.3, label='实际回撤')
ax2.set_ylabel('逃顶指数 (%)', fontsize=12)
ax2.set_ylim(0, 100)
ax2.annotate(f'当前: {cur_top*100:.0f}%', xy=(cur_date, cur_top * 100),
             xytext=(-80, 10), textcoords='offset points', fontsize=11, fontweight='bold',
             color='darkred', arrowprops=dict(arrowstyle='->', color='darkred'),
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9))
ax2.legend(fontsize=8, ncol=3, loc='upper left')
ax2.grid(True, alpha=0.2)
ax2.tick_params(labelbottom=False)

# 面板 3: 抄底指数
ax3 = axes[2]
pbs = np.where(~np.isnan(pb_s), pb_s, 0) * 100
ax3.fill_between(btc_df.index, pbs, 0, where=pbs > 50, color='green', alpha=0.4, label='机会 (>50%)')
ax3.fill_between(btc_df.index, pbs, 0, where=pbs <= 50, color='lightgreen', alpha=0.2, label='正常 (<50%)')
ax3.plot(btc_df.index, pbs, 'black', lw=0.8)
actual_bot = btc_df['label_bot'].values * 100
ax3.plot(btc_df.index, actual_bot, 'g-', lw=0.3, alpha=0.3, label='实际涨幅')
ax3.set_ylabel('抄底指数 (%)', fontsize=12)
ax3.set_ylim(0, 100)
ax3.annotate(f'当前: {cur_bot*100:.0f}%', xy=(cur_date, cur_bot * 100),
             xytext=(-80, 10), textcoords='offset points', fontsize=11, fontweight='bold',
             color='darkgreen', arrowprops=dict(arrowstyle='->', color='darkgreen'),
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
ax3.legend(fontsize=8, ncol=3, loc='upper left')
ax3.grid(True, alpha=0.2)
ax3.tick_params(labelbottom=False)

# 面板 4: 综合信号 + 滚动窗口对比
ax4 = axes[3]
combined = np.where(~np.isnan(pb_s) & ~np.isnan(pt_s), pb_s - pt_s, 0) * 100
ax4.fill_between(btc_df.index, combined, 0, where=combined > 0,
                 color='green', alpha=0.4, label='抄底 > 逃顶')
ax4.fill_between(btc_df.index, combined, 0, where=combined <= 0,
                 color='red', alpha=0.4, label='逃顶 > 抄底')
ax4.plot(btc_df.index, combined, 'black', lw=0.8)
ax4.axhline(0, color='black', lw=0.5)
ax4.set_ylabel('综合信号 (%)', fontsize=12)
ax4.legend(fontsize=8, loc='upper left')
ax4.grid(True, alpha=0.2)

plt.savefig(f'{IMG_DIR}/multi_asset_transformer.png', dpi=150, bbox_inches='tight')
print(f"   保存: {IMG_DIR}/multi_asset_transformer.png")

# ==================== 汇总 ====================
print("\n" + "=" * 80)
print("6. 模型评估汇总")
print("=" * 80)

rdf = pd.DataFrame(results)
print(f"\n  滚动窗口数: {len(rdf)}")
print(f"\n  BTC 逃顶相关性: mean={rdf['btc_corr_top'].mean():.3f}, "
      f"min={rdf['btc_corr_top'].min():.3f}, max={rdf['btc_corr_top'].max():.3f}")
print(f"  BTC 抄底相关性: mean={rdf['btc_corr_bot'].mean():.3f}, "
      f"min={rdf['btc_corr_bot'].min():.3f}, max={rdf['btc_corr_bot'].max():.3f}")
print(f"\n  全币种平均 逃顶相关性: mean={rdf['multi_corr_top'].mean():.3f}")
print(f"  全币种平均 抄底相关性: mean={rdf['multi_corr_bot'].mean():.3f}")

print(f"\n  逐窗口详情:")
for _, r in rdf.iterrows():
    print(f"    W{r['window']}: {r['train_period']} → {r['pred_period']} | "
          f"{r['n_coins']}币 {r['n_samples']:,}样本 | "
          f"BTC r_top={r['btc_corr_top']:.3f} r_bot={r['btc_corr_bot']:.3f} | "
          f"全币 r_top={r['multi_corr_top']:.3f} r_bot={r['multi_corr_bot']:.3f}")

# 与单资产模型对比
print(f"\n  📊 对比 (单资产 → 多资产):")
print(f"     训练样本: ~2,400/窗口 → {rdf['n_samples'].mean():,.0f}/窗口")
print(f"     BTC逃顶r: -0.01 → {rdf['btc_corr_top'].mean():.3f}")
print(f"     BTC抄底r: -0.02 → {rdf['btc_corr_bot'].mean():.3f}")

print(f"\n  当前 BTC 预测 ({cur_date.strftime('%Y-%m-%d')} ${cur_price:,.0f}):")
print(f"    逃顶: {cur_top*100:.1f}%  (未来180天最大回撤预期 ≈ {cur_top*100:.0f}%)")
print(f"    抄底: {cur_bot*100:.1f}%  (未来180天最大涨幅预期 ≈ {cur_bot*200:.0f}%)")

if cur_top > 0.5 and cur_bot < 0.3:
    verdict = "⚠️ 高危! 未来有大幅回撤风险"
elif cur_bot > 0.5 and cur_top < 0.3:
    verdict = "✅ 机会! 未来有大幅上涨空间"
elif cur_bot > cur_top:
    verdict = "偏乐观, 上涨空间 > 下跌风险"
else:
    verdict = "偏谨慎, 下跌风险 > 上涨空间"

print(f"    综合: {verdict}")
print("\n完成!")
