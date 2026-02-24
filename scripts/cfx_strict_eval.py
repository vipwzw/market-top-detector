#!/usr/bin/env python3
"""
CFX 严格无泄露评估

核心区别: 对于 CFX 在时间 T 的预测, 训练数据的标签
         不能超过 T (即训练样本日期 ≤ T - FWD_DAYS).

对比:
  A. 严格滚动窗口 (无泄露)
  B. 之前的"全量训练"方法 (有潜在泄露)
  C. EMA 基线 (无泄露)
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

SEQ_LEN = 120
FWD_DAYS = 180
D_MODEL = 64
NHEAD = 4
N_LAYERS = 3
DROPOUT = 0.15
BATCH = 256
EPOCHS = 30
LR = 5e-4
FEE = 0.001

TRAIN_COINS = [
    'BTC', 'ETH', 'XRP', 'BNB', 'SOL', 'TRX', 'DOGE', 'BCH', 'ADA',
    'LEO', 'XMR', 'LINK', 'LTC', 'XLM', 'HBAR', 'ZEC', 'AVAX',
    'CRO', 'UNI', 'DOT', 'TON',
]

FEATURE_COLS = [
    'ma20_dev', 'ma50_dev', 'ma100_dev', 'ma200_dev', 'mayer',
    'rsi_14', 'rsi_30', 'vol_30d', 'vol_90d',
    'ret_7d', 'ret_30d', 'ret_90d', 'ret_180d', 'ret_365d',
    'ath_dd', 'days_since_ath', 'log_detrend',
    'fng_norm', 'fng_ma30', 'gt_norm',
    'vratio_20', 'vratio_50', 'vma20_trend', 'vma50_trend',
    'vprice_corr_30', 'obv_detrend',
]
N_FEAT = len(FEATURE_COLS)
MIN_HISTORY = 365 * 2 + FWD_DAYS


def rsi(s, p):
    d = s.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100 / (1 + g / (l + 1e-10))


def compute_features(df):
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
    v = df['Volume'].replace(0, np.nan).ffill()
    for w in [20, 50]:
        vma = v.rolling(w, min_periods=5).mean()
        df[f'vratio_{w}'] = (v / (vma + 1e-10)).clip(0, 10)
        df[f'vma{w}_trend'] = vma.pct_change(w)
    price_ret = c.pct_change()
    df['vprice_corr_30'] = price_ret.rolling(30, min_periods=10).corr(v)
    obv = (np.sign(price_ret) * v).cumsum()
    obv_x = np.arange(len(obv))
    obv_valid = ~np.isnan(obv.values)
    if obv_valid.sum() > 10:
        obv_coeffs = np.polyfit(obv_x[obv_valid], obv.values[obv_valid], 1)
        df['obv_detrend'] = obv.values - np.polyval(obv_coeffs, obv_x)
        obv_std = df['obv_detrend'].std()
        if obv_std > 0:
            df['obv_detrend'] = df['obv_detrend'] / obv_std
    else:
        df['obv_detrend'] = 0.0
    return df


def compute_labels(df):
    close_arr = df['Close'].values
    n = len(close_arr)
    fwd_max_dd = np.full(n, np.nan)
    fwd_max_rally = np.full(n, np.nan)
    for i in range(n - FWD_DAYS):
        future = close_arr[i+1:i+1+FWD_DAYS]
        cur = close_arr[i]
        fwd_rets = future / cur - 1
        fwd_max_dd[i] = np.min(fwd_rets)
        fwd_max_rally[i] = np.max(fwd_rets)
    df['label_top'] = (-np.array(fwd_max_dd)).clip(0, 1)
    df['label_bot'] = np.array(fwd_max_rally).clip(0, 2) / 2.0
    return df


def build_windows(feats, sl):
    n = len(feats) - sl
    if n <= 0:
        return np.empty((0, sl, feats.shape[1]))
    shape = (n, sl, feats.shape[1])
    strides = (feats.strides[0], feats.strides[0], feats.strides[1])
    return np.lib.stride_tricks.as_strided(feats, shape=shape, strides=strides).copy()


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


def backtest(prices, dates, signals, name, fee=FEE):
    n = len(prices)
    cash = 1.0
    pos = 0.0
    equity = np.ones(n)
    trades = []
    entry_price = 0
    entry_idx = 0
    for i in range(n):
        if i > 0 and signals[i] != signals[i-1]:
            if signals[i] == 1 and pos == 0:
                pos = cash * (1 - fee) / prices[i]
                cash = 0
                entry_price = prices[i]
                entry_idx = i
            elif signals[i] == 0 and pos > 0:
                cash = pos * prices[i] * (1 - fee)
                trades.append({
                    'entry': dates[entry_idx], 'exit': dates[i],
                    'ret': prices[i] / entry_price - 1, 'days': i - entry_idx
                })
                pos = 0
        equity[i] = cash + pos * prices[i] if pos > 0 else cash
    if pos > 0:
        equity[-1] = pos * prices[-1] * (1 - fee)
    total_ret = equity[-1] / equity[0] - 1
    n_years = (dates[-1] - dates[0]).days / 365.25
    ann_ret = (1 + total_ret) ** (1 / max(n_years, 0.1)) - 1
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = np.min(dd)
    daily_ret = np.diff(equity) / equity[:-1]
    sharpe = np.mean(daily_ret) / (np.std(daily_ret) + 1e-10) * np.sqrt(365)
    win_trades = [t for t in trades if t['ret'] > 0]
    win_rate = len(win_trades) / max(len(trades), 1)
    return {
        'name': name, 'total_ret': total_ret, 'ann_ret': ann_ret,
        'max_dd': max_dd, 'sharpe': sharpe, 'n_trades': len(trades),
        'win_rate': win_rate, 'equity': equity, 'trades': trades,
    }


# ==================== 数据加载 ====================
print("=" * 80)
print("CFX 严格无泄露评估: 滚动窗口 vs 全量训练 vs EMA")
print("=" * 80)

print("\n1. 加载数据...")
fng = pd.read_csv(f'{DATA_DIR}/fear_greed_index.csv', parse_dates=['Date'], index_col='Date')
fng['fng_value'] = pd.to_numeric(fng['fng_value'], errors='coerce')
fng = fng[~fng.index.duplicated(keep='first')].sort_index()
gt = pd.read_csv(f'{DATA_DIR}/google_trends_bitcoin.csv', parse_dates=['date'], index_col='date')
gt = gt.rename(columns={'Bitcoin': 'gtrend'})
gt = gt[~gt.index.duplicated(keep='first')].sort_index()


def load_coin(fname):
    df = pd.read_csv(fname, parse_dates=[0], index_col=0)
    df = df.sort_index().dropna(subset=['Close'])
    df = df.join(fng[['fng_value']], how='left')
    df['fng_value'] = df['fng_value'].ffill().bfill()
    df['fng_norm'] = df['fng_value'] / 100.0
    df['fng_ma30'] = df['fng_norm'].rolling(30, min_periods=5).mean()
    df = df.join(gt[['gtrend']], how='left')
    df['gtrend'] = df['gtrend'].ffill().bfill()
    df['gt_norm'] = df['gtrend'] / 100.0
    df = compute_features(df)
    df = compute_labels(df)
    df_valid = df.dropna(subset=FEATURE_COLS).copy()
    for col in FEATURE_COLS:
        df_valid[col] = df_valid[col].clip(-5, 5)
    return df_valid


coin_data = {}
for coin in TRAIN_COINS:
    fname = f'{DATA_DIR}/{coin}_daily.csv'
    if not os.path.exists(fname):
        fname = f'{DATA_DIR}/BTC_daily_full.csv' if coin == 'BTC' else None
    if fname is None or not os.path.exists(fname):
        continue
    df_v = load_coin(fname)
    if len(df_v) >= SEQ_LEN + 100:
        coin_data[coin] = df_v

print(f"   训练币种: {len(coin_data)}")

cfx = load_coin(f'{DATA_DIR}/CFX_daily.csv')
print(f"   CFX: {len(cfx)} 天  {cfx.index[0].date()} ~ {cfx.index[-1].date()}")


# ==================== 模型训练函数 ====================
def train_model(coin_data, label_cutoff_date, max_train_days=365*3):
    """训练模型, 确保标签不超过 label_cutoff_date"""
    X_all, y_top_all, y_bot_all = [], [], []

    for coin, df in coin_data.items():
        # 关键: 训练样本日期 ≤ label_cutoff - FWD_DAYS
        # 这样标签用到的未来数据 ≤ label_cutoff
        sample_cutoff = label_cutoff_date - pd.Timedelta(days=FWD_DAYS)
        mask = df.index <= sample_cutoff
        df_t = df.loc[mask]

        if max_train_days:
            df_t = df_t.iloc[-max_train_days:]

        if len(df_t) < SEQ_LEN + 50:
            continue

        feats = df_t[FEATURE_COLS].values.astype(np.float32)
        lt = df_t['label_top'].values.astype(np.float32)
        lb = df_t['label_bot'].values.astype(np.float32)

        Xw = build_windows(feats, SEQ_LEN)
        nw = min(len(Xw), len(lt) - SEQ_LEN)
        if nw <= 0:
            continue

        valid = ~np.isnan(lt[SEQ_LEN:SEQ_LEN+nw]) & ~np.isnan(lb[SEQ_LEN:SEQ_LEN+nw])
        X_all.append(Xw[:nw][valid])
        y_top_all.append(lt[SEQ_LEN:SEQ_LEN+nw][valid])
        y_bot_all.append(lb[SEQ_LEN:SEQ_LEN+nw][valid])

    if not X_all:
        return None

    X = np.concatenate(X_all)
    yt = np.concatenate(y_top_all)
    yb = np.concatenate(y_bot_all)
    n = len(X)

    perm = np.random.permutation(n)
    X, yt, yb = X[perm], yt[perm], yb[perm]

    model = RiskTransformer(N_FEAT, D_MODEL, NHEAD, N_LAYERS, DROPOUT).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    crit = nn.MSELoss()

    Xt = torch.FloatTensor(X)
    ytt = torch.FloatTensor(yt)
    ybt = torch.FloatTensor(yb)

    model.train()
    best_loss = float('inf')
    best_state = None
    for epoch in range(EPOCHS):
        pe = torch.randperm(n)
        el = 0.0
        nb = 0
        for s in range(0, n - BATCH + 1, BATCH):
            bi = pe[s:s+BATCH]
            xb = Xt[bi].to(DEVICE)
            opt.zero_grad()
            pt, pb = model(xb)
            loss = crit(pt, ytt[bi].to(DEVICE)) + crit(pb, ybt[bi].to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            el += loss.item()
            nb += 1
        scheduler.step()
        avg = el / max(nb, 1)
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    del Xt, ytt, ybt
    return model, n, best_loss


def predict_cfx(model, cfx_df, start_idx, end_idx):
    """预测 CFX 一段区间"""
    feats = cfx_df[FEATURE_COLS].values.astype(np.float32)

    ctx_start = max(0, start_idx - SEQ_LEN)
    Xp = build_windows(feats[ctx_start:end_idx], SEQ_LEN)
    if len(Xp) == 0:
        return np.array([]), np.array([])

    with torch.no_grad():
        apt, apb = [], []
        for s in range(0, len(Xp), 512):
            e = min(s+512, len(Xp))
            pt, pb = model(torch.FloatTensor(Xp[s:e]).to(DEVICE))
            apt.append(pt.cpu().numpy())
            apb.append(pb.cpu().numpy())

    return np.concatenate(apt), np.concatenate(apb)


# ==================== 2. 严格滚动窗口 ====================
print(f"\n2. 严格滚动窗口评估 (设备: {DEVICE})")
print(f"   规则: 预测 CFX 时间 T 时, 训练标签 ≤ T")

PRED_DAYS = 180
SLIDE_DAYS = 180

cfx_feats = cfx.index
cfx_pred_top = np.full(len(cfx), np.nan)
cfx_pred_bot = np.full(len(cfx), np.nan)

# CFX 有效预测起点: 需要 SEQ_LEN 天的历史特征
pred_start_date = cfx.index[SEQ_LEN]
max_date = cfx.index[-1]

results_rolling = []
cur_date = pred_start_date
window_n = 0

while cur_date < max_date:
    t0 = time.time()
    window_n += 1

    # 严格: 标签截止 = 当前预测起始日
    # 训练样本日期 ≤ cur_date - FWD_DAYS
    label_cutoff = cur_date
    pred_end = min(cur_date + pd.Timedelta(days=PRED_DAYS), max_date)

    res = train_model(coin_data, label_cutoff)
    if res is None:
        cur_date += pd.Timedelta(days=SLIDE_DAYS)
        continue

    model, n_samples, loss = res

    # 预测 CFX 这段区间
    start_idx = cfx.index.searchsorted(cur_date)
    end_idx = cfx.index.searchsorted(pred_end)
    if end_idx > len(cfx):
        end_idx = len(cfx)

    pt, pb = predict_cfx(model, cfx, start_idx, end_idx)

    n_fill = min(end_idx - start_idx, len(pt))
    for j in range(n_fill):
        gi = start_idx + j
        if gi < len(cfx_pred_top):
            cfx_pred_top[gi] = pt[j]
            cfx_pred_bot[gi] = pb[j]

    # 评估
    act_top = cfx['label_top'].values[start_idx:end_idx]
    act_bot = cfx['label_bot'].values[start_idx:end_idx]
    p_top_seg = cfx_pred_top[start_idx:end_idx]
    p_bot_seg = cfx_pred_bot[start_idx:end_idx]

    vm = ~np.isnan(p_top_seg) & ~np.isnan(act_top)
    if vm.sum() > 10:
        ct = np.corrcoef(p_top_seg[vm], act_top[vm])[0, 1]
        cb = np.corrcoef(p_bot_seg[vm], act_bot[vm])[0, 1]
    else:
        ct = cb = float('nan')

    elapsed = time.time() - t0
    print(f"   W{window_n}: label_cutoff={label_cutoff.date()} → 预测{cur_date.date()}~{pred_end.date()} | "
          f"{n_samples:,}样本 loss={loss:.4f} | 逃顶r={ct:.3f} 抄底r={cb:.3f} | {elapsed:.0f}s")

    results_rolling.append({
        'window': window_n, 'pred_start': cur_date.date(), 'pred_end': pred_end.date(),
        'n_samples': n_samples, 'loss': loss, 'corr_top': ct, 'corr_bot': cb
    })

    del model
    if DEVICE == 'mps':
        torch.mps.empty_cache()
    gc.collect()

    cur_date += pd.Timedelta(days=SLIDE_DAYS)

# ==================== 3. 加载之前全量训练的预测 ====================
print("\n3. 加载之前全量训练的预测 (有泄露风险)...")
leaked_df = pd.read_csv(f'{DATA_DIR}/cfx_predictions.csv', parse_dates=['date'], index_col='date')
leaked_top = leaked_df['pred_top'].values
leaked_bot = leaked_df['pred_bot'].values
leaked_dates = leaked_df.index
leaked_close = leaked_df['close'].values

# ==================== 4. 对齐数据做回测对比 ====================
print("\n4. 回测对比...")

# 对齐: 只用严格预测有值的区间
valid_mask = ~np.isnan(cfx_pred_top)
valid_indices = np.where(valid_mask)[0]
if len(valid_indices) > 0:
    vi_start = valid_indices[0]
    vi_end = valid_indices[-1] + 1
else:
    vi_start = SEQ_LEN
    vi_end = len(cfx)

eval_dates = cfx.index[vi_start:vi_end]
eval_close = cfx['Close'].values[vi_start:vi_end]
eval_top_strict = cfx_pred_top[vi_start:vi_end]
eval_bot_strict = cfx_pred_bot[vi_start:vi_end]
eval_label_top = cfx['label_top'].values[vi_start:vi_end]
eval_label_bot = cfx['label_bot'].values[vi_start:vi_end]

# 对齐泄露版预测到相同区间
eval_top_leaked = np.full(len(eval_dates), np.nan)
eval_bot_leaked = np.full(len(eval_dates), np.nan)
for i, d in enumerate(eval_dates):
    if d in leaked_df.index:
        eval_top_leaked[i] = leaked_df.loc[d, 'pred_top']
        eval_bot_leaked[i] = leaked_df.loc[d, 'pred_bot']

n_eval = len(eval_dates)
print(f"   评估区间: {eval_dates[0].date()} ~ {eval_dates[-1].date()} ({n_eval} 天)")

# 相关性对比
vm_s = ~np.isnan(eval_top_strict) & ~np.isnan(eval_label_top)
vm_l = ~np.isnan(eval_top_leaked) & ~np.isnan(eval_label_top)

corr_strict_top = np.corrcoef(eval_top_strict[vm_s], eval_label_top[vm_s])[0, 1] if vm_s.sum() > 10 else 0
corr_strict_bot = np.corrcoef(eval_bot_strict[vm_s], eval_label_bot[vm_s])[0, 1] if vm_s.sum() > 10 else 0
corr_leaked_top = np.corrcoef(eval_top_leaked[vm_l], eval_label_top[vm_l])[0, 1] if vm_l.sum() > 10 else 0
corr_leaked_bot = np.corrcoef(eval_bot_leaked[vm_l], eval_label_bot[vm_l])[0, 1] if vm_l.sum() > 10 else 0

print(f"\n   相关性对比 (同一评估区间):")
print(f"     {'方法':20s} | {'逃顶 r':>8s} | {'抄底 r':>8s}")
print(f"     {'-'*42}")
print(f"     {'严格滚动 (无泄露)':20s} | {corr_strict_top:8.3f} | {corr_strict_bot:8.3f}")
print(f"     {'全量训练 (有泄露)':20s} | {corr_leaked_top:8.3f} | {corr_leaked_bot:8.3f}")

# 回测策略
sm_w = 14

def smooth(arr, w):
    return pd.Series(arr).rolling(w, min_periods=1).mean().values

strict_top_s = smooth(np.nan_to_num(eval_top_strict, nan=0.3), sm_w)
strict_bot_s = smooth(np.nan_to_num(eval_bot_strict, nan=0.3), sm_w)
leaked_top_s = smooth(np.nan_to_num(eval_top_leaked, nan=0.3), sm_w)
leaked_bot_s = smooth(np.nan_to_num(eval_bot_leaked, nan=0.3), sm_w)


def make_signals(top_s, bot_s, buy_th, sell_th):
    signals = np.zeros(len(top_s), dtype=int)
    holding = False
    for i in range(len(top_s)):
        if not holding and bot_s[i] > buy_th and top_s[i] < sell_th:
            holding = True
        elif holding and top_s[i] > sell_th:
            holding = False
        signals[i] = 1 if holding else 0
    return signals


# 扫描严格版最优参数
print("\n   扫描严格版策略参数...")
best_strict = None
for bt in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    for st in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        sigs = make_signals(strict_top_s, strict_bot_s, bt, st)
        if sigs.sum() < 5:
            continue
        r = backtest(eval_close, eval_dates, sigs, f'严格TF s{sm_w} B>{bt:.2f} S>{st:.2f}')
        if r['n_trades'] >= 3 and (best_strict is None or r['sharpe'] > best_strict['sharpe']):
            best_strict = r

# 扫描泄露版同参数
best_leaked = None
for bt in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    for st in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        sigs = make_signals(leaked_top_s, leaked_bot_s, bt, st)
        if sigs.sum() < 5:
            continue
        r = backtest(eval_close, eval_dates, sigs, f'泄露TF s{sm_w} B>{bt:.2f} S>{st:.2f}')
        if r['n_trades'] >= 3 and (best_leaked is None or r['sharpe'] > best_leaked['sharpe']):
            best_leaked = r

# EMA 策略
def ema_crossover(prices, fast, slow):
    ef = pd.Series(prices).ewm(span=fast).mean().values
    es = pd.Series(prices).ewm(span=slow).mean().values
    signals = np.zeros(len(prices), dtype=int)
    holding = False
    for i in range(1, len(prices)):
        if not holding and ef[i] > es[i] and ef[i-1] <= es[i-1]:
            holding = True
        elif holding and ef[i] < es[i] and ef[i-1] >= es[i-1]:
            holding = False
        signals[i] = 1 if holding else 0
    return signals

ema_params = [(9, 21), (12, 26), (15, 45), (20, 50), (30, 90), (50, 200)]
best_ema = None
for f, s in ema_params:
    sigs = ema_crossover(eval_close, f, s)
    r = backtest(eval_close, eval_dates, sigs, f'EMA({f},{s})')
    if best_ema is None or r['sharpe'] > best_ema['sharpe']:
        best_ema = r

# Buy & Hold
bh = backtest(eval_close, eval_dates, np.ones(n_eval, dtype=int), 'Buy & Hold')

# ==================== 5. 输出 ====================
all_results = [bh]
if best_strict:
    all_results.append(best_strict)
if best_leaked:
    all_results.append(best_leaked)
if best_ema:
    all_results.append(best_ema)

# 所有 EMA
for f, s in ema_params:
    sigs = ema_crossover(eval_close, f, s)
    r = backtest(eval_close, eval_dates, sigs, f'EMA({f},{s})')
    if r['name'] != best_ema['name']:
        all_results.append(r)
    else:
        for i, ar in enumerate(all_results):
            if ar['name'] == best_ema['name']:
                all_results[i] = r
                break

print(f"\n  {'策略':38s} | {'收益':>8s} | {'Sharpe':>7s} | {'回撤':>8s} | {'交易':>4s} | {'胜率':>4s}")
print(f"  {'-'*80}")
for r in sorted(all_results, key=lambda x: x['sharpe'], reverse=True):
    print(f"  {r['name']:38s} | {r['total_ret']:+7.1%} | {r['sharpe']:7.2f} | "
          f"{r['max_dd']:+7.1%} | {r['n_trades']:4d} | {r['win_rate']:4.0%}")

# ==================== 6. 可视化 ====================
print("\n5. 生成对比图...")

fig, axes = plt.subplots(4, 1, figsize=(30, 24),
                         gridspec_kw={'height_ratios': [3, 1.5, 1.5, 2.5]})
fig.subplots_adjust(hspace=0.1)

# 面板1: 价格
ax = axes[0]
ax.plot(eval_dates, eval_close, 'black', lw=1)
for i in range(1, n_eval):
    if strict_top_s[i] > 0.3:
        ax.axvspan(eval_dates[i-1], eval_dates[i],
                   alpha=min(0.5, strict_top_s[i]*0.6), color='red', lw=0)
    if strict_bot_s[i] > 0.3:
        ax.axvspan(eval_dates[i-1], eval_dates[i],
                   alpha=min(0.5, strict_bot_s[i]*0.5), color='green', lw=0)
ax.set_yscale('log')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:.3f}' if x < 1 else f'${x:.2f}'))
ax.set_title('CFX 严格无泄露评估 — Transformer 信号 (滚动窗口, 标签不超前)\n'
             '红=逃顶  绿=抄底', fontsize=14, fontweight='bold')
ax.set_ylabel('CFX Price (log)')
ax.grid(True, alpha=0.15, which='both')
ax.tick_params(labelbottom=False)

# 面板2: 逃顶对比
ax = axes[1]
ax.plot(eval_dates, strict_top_s * 100, 'red', lw=1.5, label='严格(无泄露)')
ax.plot(eval_dates, leaked_top_s * 100, 'orange', lw=1, ls='--', alpha=0.7, label='全量(有泄露)')
ax.set_ylabel('逃顶指数 (%)')
ax.set_ylim(0, 100)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.2)
ax.tick_params(labelbottom=False)

# 面板3: 抄底对比
ax = axes[2]
ax.plot(eval_dates, strict_bot_s * 100, 'green', lw=1.5, label='严格(无泄露)')
ax.plot(eval_dates, leaked_bot_s * 100, 'lightgreen', lw=1, ls='--', alpha=0.7, label='全量(有泄露)')
ax.set_ylabel('抄底指数 (%)')
ax.set_ylim(0, 100)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.2)
ax.tick_params(labelbottom=False)

# 面板4: 净值
ax = axes[3]
for r in sorted(all_results, key=lambda x: x['sharpe'], reverse=True):
    lw = 2.5 if '严格' in r['name'] else (1.5 if '泄露' in r['name'] else 1)
    ls = '-' if 'TF' in r['name'] or '严格' in r['name'] or '泄露' in r['name'] else '--'
    c = 'red' if '严格' in r['name'] else ('orange' if '泄露' in r['name'] else
        ('gray' if 'Buy' in r['name'] else 'blue'))
    ax.plot(eval_dates, r['equity'], c=c, lw=lw, ls=ls, alpha=0.85,
            label=f"{r['name']} ({r['total_ret']:+.0%} S={r['sharpe']:.2f})")
ax.set_ylabel('净值')
ax.legend(fontsize=7, ncol=2, loc='upper left')
ax.grid(True, alpha=0.2)
ax.set_title('净值对比: 红=严格TF  橙=泄露TF  蓝=最优EMA  灰=Buy&Hold',
             fontsize=11, fontweight='bold')

plt.savefig(f'{IMG_DIR}/cfx_strict_eval.png', dpi=150, bbox_inches='tight')
print(f"   保存: {IMG_DIR}/cfx_strict_eval.png")

# ==================== 7. 汇总 ====================
print("\n" + "=" * 80)
print("6. 最终结论")
print("=" * 80)

print(f"\n  相关性对比:")
print(f"    严格滚动 (无泄露): 逃顶 r={corr_strict_top:.3f}, 抄底 r={corr_strict_bot:.3f}")
print(f"    全量训练 (有泄露): 逃顶 r={corr_leaked_top:.3f}, 抄底 r={corr_leaked_bot:.3f}")
if abs(corr_leaked_top) > 0:
    drop_top = (1 - corr_strict_top / corr_leaked_top) * 100
    drop_bot = (1 - corr_strict_bot / corr_leaked_bot) * 100
    print(f"    泄露膨胀: 逃顶 {drop_top:+.1f}%, 抄底 {drop_bot:+.1f}%")

print(f"\n  回测对比:")
if best_strict:
    print(f"    严格TF: {best_strict['name']}")
    print(f"      收益 {best_strict['total_ret']:+.1%}, Sharpe {best_strict['sharpe']:.2f}, "
          f"回撤 {best_strict['max_dd']:.1%}, {best_strict['n_trades']}笔, 胜率 {best_strict['win_rate']:.0%}")
if best_leaked:
    print(f"    泄露TF: {best_leaked['name']}")
    print(f"      收益 {best_leaked['total_ret']:+.1%}, Sharpe {best_leaked['sharpe']:.2f}, "
          f"回撤 {best_leaked['max_dd']:.1%}, {best_leaked['n_trades']}笔, 胜率 {best_leaked['win_rate']:.0%}")
if best_ema:
    print(f"    最优EMA: {best_ema['name']}")
    print(f"      收益 {best_ema['total_ret']:+.1%}, Sharpe {best_ema['sharpe']:.2f}, "
          f"回撤 {best_ema['max_dd']:.1%}, {best_ema['n_trades']}笔, 胜率 {best_ema['win_rate']:.0%}")

if best_strict and best_ema:
    if best_strict['sharpe'] > best_ema['sharpe']:
        print(f"\n  → 即使严格无泄露, Transformer 仍然优于 EMA! (Sharpe {best_strict['sharpe']:.2f} vs {best_ema['sharpe']:.2f})")
    else:
        print(f"\n  → 去除泄露后, EMA 更优 (Sharpe {best_ema['sharpe']:.2f} vs {best_strict['sharpe']:.2f})")
        print(f"    说明之前的 Transformer 优势部分来自信息泄露")

print("\n完成!")
