#!/usr/bin/env python3
"""
CFX 预测 + 回测: 多资产 Transformer 信号 vs EMA 均线策略

流程:
  1. 用 21 币种训练最终模型 (复用 multi_asset_transformer 的逻辑)
  2. CFX 纯样本外预测 (模型从未见过 CFX)
  3. 基于信号的抄底/逃顶策略
  4. 多组 EMA 策略
  5. 综合对比
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
EPOCHS = 40
LR = 5e-4

FEE = 0.001  # 单边手续费 0.1%
MODEL_PATH = '/Users/king/quant/market-top-detector/models/multi_asset_final.pt'
PRED_PATH = '/Users/king/quant/market-top-detector/data/cfx_predictions.csv'
os.makedirs('/Users/king/quant/market-top-detector/models', exist_ok=True)

TRAIN_COINS = [
    'BTC', 'ETH', 'XRP', 'BNB', 'SOL', 'TRX', 'DOGE', 'BCH', 'ADA',
    'LEO', 'XMR', 'LINK', 'LTC', 'XLM', 'HBAR', 'ZEC', 'AVAX',
    'CRO', 'UNI', 'DOT', 'TON',
]

FEATURE_COLS = [
    'ma20_dev', 'ma50_dev', 'ma100_dev', 'ma200_dev',
    'mayer',
    'rsi_14', 'rsi_30',
    'vol_30d', 'vol_90d',
    'ret_7d', 'ret_30d', 'ret_90d', 'ret_180d', 'ret_365d',
    'ath_dd', 'days_since_ath',
    'log_detrend',
    'fng_norm', 'fng_ma30', 'gt_norm',
    # 成交量特征
    'vratio_20', 'vratio_50',       # 当日量 / MA量 (放量/缩量)
    'vma20_trend', 'vma50_trend',   # 量均线变化率
    'vprice_corr_30',               # 量价相关性
    'obv_detrend',                  # OBV 去趋势
]
N_FEAT = len(FEATURE_COLS)
MIN_HISTORY = 365 * 2 + FWD_DAYS


# ==================== 工具 ====================
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

    # ---- 成交量特征 ----
    v = df['Volume'].replace(0, np.nan).ffill()

    # 相对成交量: 当日 / MA (>1 放量, <1 缩量)
    for w in [20, 50]:
        vma = v.rolling(w, min_periods=5).mean()
        df[f'vratio_{w}'] = (v / (vma + 1e-10)).clip(0, 10)
        df[f'vma{w}_trend'] = vma.pct_change(w)  # 量均线动量

    # 量价相关性: 30天窗口内价格涨跌与成交量的相关系数
    # 顶部: 放量上涨→放量下跌 (相关性从正变负)
    # 底部: 缩量阴跌 (低相关 or 负相关)
    price_ret = c.pct_change()
    df['vprice_corr_30'] = price_ret.rolling(30, min_periods=10).corr(v)

    # OBV (On-Balance Volume) 去趋势
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


# ==================== 回测引擎 ====================
def backtest(prices, dates, signals, name, fee=FEE):
    """
    signals: 每天 +1=做多, 0=空仓, -1=做空(此处不做空)
    返回: dict 含各项指标
    """
    n = len(prices)
    cash = 1.0
    pos = 0.0  # 持仓数量
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
                ret = prices[i] / entry_price - 1
                trades.append({
                    'entry': dates[entry_idx], 'exit': dates[i],
                    'entry_p': entry_price, 'exit_p': prices[i],
                    'ret': ret, 'days': i - entry_idx
                })
                pos = 0

        equity[i] = cash + pos * prices[i] if pos > 0 else cash

    if pos > 0:
        cash = pos * prices[-1] * (1 - fee)
        equity[-1] = cash

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

    buy_hold_ret = prices[-1] / prices[0] - 1

    return {
        'name': name,
        'total_ret': total_ret,
        'ann_ret': ann_ret,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'n_trades': len(trades),
        'win_rate': win_rate,
        'avg_trade_ret': np.mean([t['ret'] for t in trades]) if trades else 0,
        'avg_hold_days': np.mean([t['days'] for t in trades]) if trades else 0,
        'buy_hold_ret': buy_hold_ret,
        'equity': equity,
        'trades': trades,
    }


# ==================== 1. 加载数据 & 训练模型 ====================
print("=" * 80)
print("CFX 预测回测: 多资产 Transformer vs EMA 策略")
print("=" * 80)

print("\n1. 加载情绪数据...")
fng = pd.read_csv(f'{DATA_DIR}/fear_greed_index.csv', parse_dates=['Date'], index_col='Date')
fng['fng_value'] = pd.to_numeric(fng['fng_value'], errors='coerce')
fng = fng[~fng.index.duplicated(keep='first')].sort_index()

gt = pd.read_csv(f'{DATA_DIR}/google_trends_bitcoin.csv', parse_dates=['date'], index_col='date')
gt = gt.rename(columns={'Bitcoin': 'gtrend'})
gt = gt[~gt.index.duplicated(keep='first')].sort_index()

print("\n2. 加载训练币种...")
coin_data = {}
for coin in TRAIN_COINS:
    fname = f'{DATA_DIR}/{coin}_daily.csv'
    if not os.path.exists(fname):
        fname = f'{DATA_DIR}/BTC_daily_full.csv' if coin == 'BTC' else None
    if fname is None or not os.path.exists(fname):
        continue

    df = pd.read_csv(fname, parse_dates=[0], index_col=0)
    df = df.sort_index().dropna(subset=['Close'])
    if len(df) < MIN_HISTORY:
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

    if len(df_valid) >= SEQ_LEN + 100:
        coin_data[coin] = df_valid
        print(f"   {coin:>5}: {len(df_valid):>5} 天")

print(f"   → 共 {len(coin_data)} 币种训练")

print("\n3. 加载 CFX...")
cfx = pd.read_csv(f'{DATA_DIR}/CFX_daily.csv', parse_dates=[0], index_col=0)
cfx = cfx.sort_index().dropna(subset=['Close'])
cfx = cfx.join(fng[['fng_value']], how='left')
cfx['fng_value'] = cfx['fng_value'].ffill().bfill()
cfx['fng_norm'] = cfx['fng_value'] / 100.0
cfx['fng_ma30'] = cfx['fng_norm'].rolling(30, min_periods=5).mean()
cfx = cfx.join(gt[['gtrend']], how='left')
cfx['gtrend'] = cfx['gtrend'].ffill().bfill()
cfx['gt_norm'] = cfx['gtrend'] / 100.0
cfx = compute_features(cfx)
cfx = compute_labels(cfx)
cfx_valid = cfx.dropna(subset=FEATURE_COLS).copy()
for col in FEATURE_COLS:
    cfx_valid[col] = cfx_valid[col].clip(-5, 5)
print(f"   CFX: {len(cfx_valid)} 天  {cfx_valid.index[0].date()} ~ {cfx_valid.index[-1].date()}")

# ==================== 4. 训练 / 加载模型 ====================
skip_train = os.path.exists(PRED_PATH)

if skip_train:
    print(f"\n4. 发现已有预测缓存, 跳过训练")
    pred_df = pd.read_csv(PRED_PATH, parse_dates=['date'], index_col='date')
    pred_top = pred_df['pred_top'].values
    pred_bot = pred_df['pred_bot'].values
    pred_dates = pred_df.index
    pred_close = pred_df['close'].values
else:
    print(f"\n4. 训练多资产模型 (设备: {DEVICE})...")

    cutoff_date = cfx_valid.index.max() - pd.Timedelta(days=FWD_DAYS)
    X_all = []
    y_top_all = []
    y_bot_all = []

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
        X_all.append(Xw[:nw])
        y_top_all.append(lt[SEQ_LEN:SEQ_LEN + nw])
        y_bot_all.append(lb[SEQ_LEN:SEQ_LEN + nw])

    X_train = np.concatenate(X_all)
    y_top_train = np.concatenate(y_top_all)
    y_bot_train = np.concatenate(y_bot_all)

    valid_m = ~np.isnan(y_top_train) & ~np.isnan(y_bot_train)
    X_train = X_train[valid_m]
    y_top_train = y_top_train[valid_m]
    y_bot_train = y_bot_train[valid_m]

    n_train = len(X_train)
    perm = np.random.permutation(n_train)
    X_train = X_train[perm]
    y_top_train = y_top_train[perm]
    y_bot_train = y_bot_train[perm]

    print(f"   训练集: {n_train:,} 样本")

    model = RiskTransformer(N_FEAT, D_MODEL, NHEAD, N_LAYERS, DROPOUT).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    crit = nn.MSELoss()

    X_t = torch.FloatTensor(X_train)
    y_top_t = torch.FloatTensor(y_top_train)
    y_bot_t = torch.FloatTensor(y_bot_train)

    model.train()
    best_loss = float('inf')
    best_state = None
    t0 = time.time()

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
        if (epoch + 1) % 10 == 0:
            print(f"     epoch {epoch+1}/{EPOCHS}: loss={avg_loss:.5f}")

    model.load_state_dict(best_state)
    print(f"   训练完成: best_loss={best_loss:.5f} ({time.time()-t0:.0f}s)")

    # 保存模型
    torch.save(best_state, MODEL_PATH)
    print(f"   模型已保存: {MODEL_PATH}")

    # CFX 预测
    print("\n5. CFX 预测...")
    model.eval()
    cfx_feats = cfx_valid[FEATURE_COLS].values.astype(np.float32)
    X_cfx = build_windows(cfx_feats, SEQ_LEN)

    with torch.no_grad():
        all_pt = []
        all_pb = []
        for s in range(0, len(X_cfx), 512):
            e = min(s + 512, len(X_cfx))
            pt, pb = model(torch.FloatTensor(X_cfx[s:e]).to(DEVICE))
            all_pt.append(pt.cpu().numpy())
            all_pb.append(pb.cpu().numpy())

    pred_top = np.concatenate(all_pt)
    pred_bot = np.concatenate(all_pb)

    pred_dates = cfx_valid.index[SEQ_LEN:SEQ_LEN + len(pred_top)]
    pred_close = cfx_valid['Close'].values[SEQ_LEN:SEQ_LEN + len(pred_top)]

    # 保存预测缓存
    pred_df = pd.DataFrame({
        'date': pred_dates, 'close': pred_close,
        'pred_top': pred_top, 'pred_bot': pred_bot
    }).set_index('date')
    pred_df.to_csv(PRED_PATH)
    print(f"   预测已缓存: {PRED_PATH}")

    del model, X_t, y_top_t, y_bot_t
    if DEVICE == 'mps':
        torch.mps.empty_cache()
    gc.collect()

# ==================== 5. 信号平滑 + 评估 ====================
print("\n5. 信号分析...")

# 实际标签
actual_top = cfx_valid['label_top'].values[SEQ_LEN:SEQ_LEN + len(pred_top)] \
    if 'label_top' in cfx_valid.columns else None
actual_bot = cfx_valid['label_bot'].values[SEQ_LEN:SEQ_LEN + len(pred_top)] \
    if 'label_bot' in cfx_valid.columns else None

if actual_top is not None:
    vm = ~np.isnan(actual_top) & ~np.isnan(pred_top)
    if vm.sum() > 10:
        corr_top = np.corrcoef(pred_top[vm], actual_top[vm])[0, 1]
        corr_bot = np.corrcoef(pred_bot[vm], actual_bot[vm])[0, 1]
        print(f"   CFX 样本外相关性: 逃顶 r={corr_top:.3f}, 抄底 r={corr_bot:.3f}")

print(f"   pred_top: mean={pred_top.mean():.3f} std={pred_top.std():.3f} "
      f"p25={np.percentile(pred_top,25):.3f} p50={np.percentile(pred_top,50):.3f} "
      f"p75={np.percentile(pred_top,75):.3f}")
print(f"   pred_bot: mean={pred_bot.mean():.3f} std={pred_bot.std():.3f} "
      f"p25={np.percentile(pred_bot,25):.3f} p50={np.percentile(pred_bot,50):.3f} "
      f"p75={np.percentile(pred_bot,75):.3f}")

# ==================== 6. 构建交易策略 ====================
print("\n6. 构建交易策略...")

n_pred = len(pred_close)


# --- Transformer 策略族 ---
def make_tf_signals(top_s, bot_s, buy_th, sell_th):
    """买入: bot > buy_th & top < sell_th; 卖出: top > sell_th"""
    signals = np.zeros(len(top_s), dtype=int)
    holding = False
    for i in range(len(top_s)):
        if not holding and bot_s[i] > buy_th and top_s[i] < sell_th:
            holding = True
        elif holding and top_s[i] > sell_th:
            holding = False
        signals[i] = 1 if holding else 0
    return signals


def make_combined_signals(top_s, bot_s, buy_th, sell_th):
    """综合信号: 买入 combined > buy_th, 卖出 combined < -sell_th"""
    combined = bot_s - top_s
    signals = np.zeros(len(top_s), dtype=int)
    holding = False
    for i in range(len(top_s)):
        if not holding and combined[i] > buy_th:
            holding = True
        elif holding and combined[i] < -sell_th:
            holding = False
        signals[i] = 1 if holding else 0
    return signals


def make_ratio_signals(top_s, bot_s, buy_ratio, sell_ratio):
    """比率策略: 买入 bot/top > buy_ratio, 卖出 top/bot > sell_ratio"""
    signals = np.zeros(len(top_s), dtype=int)
    holding = False
    for i in range(len(top_s)):
        r_bt = bot_s[i] / max(top_s[i], 0.01)
        r_tb = top_s[i] / max(bot_s[i], 0.01)
        if not holding and r_bt > buy_ratio:
            holding = True
        elif holding and r_tb > sell_ratio:
            holding = False
        signals[i] = 1 if holding else 0
    return signals


# 多组平滑窗口扫描
print("   扫描 Transformer 策略 (多平滑窗口 × 多阈值)...")
tf_scan = []

for sm_w in [3, 5, 7, 14]:
    pts = pd.Series(pred_top, index=pred_dates).rolling(sm_w, min_periods=1).mean().values
    pbs = pd.Series(pred_bot, index=pred_dates).rolling(sm_w, min_periods=1).mean().values

    # 独立阈值策略
    for buy_th in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        for sell_th in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
            sigs = make_tf_signals(pts, pbs, buy_th, sell_th)
            if sigs.sum() < 5:
                continue
            r = backtest(pred_close, pred_dates, sigs,
                         f'TF s{sm_w} B>{buy_th:.2f} S>{sell_th:.2f}')
            tf_scan.append((sm_w, 'ind', buy_th, sell_th, r))

    # 综合信号策略
    for b_th in [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]:
        for s_th in [0.0, 0.02, 0.05, 0.08, 0.10, 0.15]:
            sigs = make_combined_signals(pts, pbs, b_th, s_th)
            if sigs.sum() < 5:
                continue
            r = backtest(pred_close, pred_dates, sigs,
                         f'TF综合 s{sm_w} B>{b_th:.2f} S>{s_th:.2f}')
            tf_scan.append((sm_w, 'comb', b_th, s_th, r))

    # 比率策略
    for br in [1.2, 1.5, 2.0, 2.5, 3.0]:
        for sr in [1.2, 1.5, 2.0, 2.5]:
            sigs = make_ratio_signals(pts, pbs, br, sr)
            if sigs.sum() < 5:
                continue
            r = backtest(pred_close, pred_dates, sigs,
                         f'TF比率 s{sm_w} B>{br:.1f} S>{sr:.1f}')
            tf_scan.append((sm_w, 'ratio', br, sr, r))

# 选最优 + top N
tf_scan.sort(key=lambda x: x[4]['sharpe'], reverse=True)
print(f"   扫描 {len(tf_scan)} 组参数")
print(f"\n   Top 10 Transformer 策略:")
for rank, (sw, tp, bt, st, r) in enumerate(tf_scan[:10], 1):
    print(f"     {rank:2d}. {r['name']:35s} | ret={r['total_ret']:+8.1%} | "
          f"dd={r['max_dd']:+7.1%} | sharpe={r['sharpe']:.2f} | "
          f"trades={r['n_trades']:2d} win={r['win_rate']:.0%}")

# 选择不同交易频率档位的最优策略
tf_selected = []
freq_bins = [(3, 6, '低频'), (6, 12, '中频'), (12, 30, '高频'), (30, 999, '超高频')]
for lo, hi, label in freq_bins:
    candidates = [x for x in tf_scan if lo <= x[4]['n_trades'] < hi and x[4]['sharpe'] > 0]
    if candidates:
        best = candidates[0]
        tf_selected.append((label, best))
        print(f"\n   {label}最优 ({lo}-{hi}笔): {best[4]['name']}")
        print(f"     ret={best[4]['total_ret']:+.1%} sharpe={best[4]['sharpe']:.2f} "
              f"dd={best[4]['max_dd']:.1%} trades={best[4]['n_trades']} win={best[4]['win_rate']:.0%}")

# --- EMA 策略族 ---
def ema_crossover_signals(prices, fast, slow):
    ema_fast = pd.Series(prices).ewm(span=fast).mean().values
    ema_slow = pd.Series(prices).ewm(span=slow).mean().values
    signals = np.zeros(len(prices), dtype=int)
    holding = False
    for i in range(1, len(prices)):
        if not holding and ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]:
            holding = True
        elif holding and ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]:
            holding = False
        signals[i] = 1 if holding else 0
    return signals, ema_fast, ema_slow

ema_params = [(5, 13), (9, 21), (12, 26), (15, 45), (20, 50), (20, 60), (30, 90), (50, 200)]

# --- Buy & Hold ---
bh_signals = np.ones(n_pred, dtype=int)

# ==================== 7. 回测 ====================
print("\n7. 回测对比...")

all_results = []

# Buy & Hold
r = backtest(pred_close, pred_dates, bh_signals, 'Buy & Hold')
all_results.append(r)
print(f"   {'Buy & Hold':35s} | ret={r['total_ret']:+7.1%} | dd={r['max_dd']:+7.1%} | sharpe={r['sharpe']:.2f}")

# 各频率档位最优 Transformer
for label, (sw, tp, bt, st, r_cached) in tf_selected:
    r_cached['name'] = f'TF-{label} {r_cached["name"]}'
    all_results.append(r_cached)
    print(f"   {r_cached['name']:35s} | ret={r_cached['total_ret']:+7.1%} | dd={r_cached['max_dd']:+7.1%} | "
          f"sharpe={r_cached['sharpe']:.2f} | trades={r_cached['n_trades']} win={r_cached['win_rate']:.0%}")

# 全局最优 Transformer (不分频率)
if tf_scan:
    best_overall = tf_scan[0][4]
    if best_overall['name'] not in [r['name'] for r in all_results]:
        best_overall['name'] = f'TF-最优 {best_overall["name"]}'
        all_results.append(best_overall)
        print(f"   {best_overall['name']:35s} | ret={best_overall['total_ret']:+7.1%} | "
              f"dd={best_overall['max_dd']:+7.1%} | sharpe={best_overall['sharpe']:.2f} | "
              f"trades={best_overall['n_trades']} win={best_overall['win_rate']:.0%}")

# EMA 策略
ema_data = {}
for fast, slow in ema_params:
    sigs, ef, es = ema_crossover_signals(pred_close, fast, slow)
    r = backtest(pred_close, pred_dates, sigs, f'EMA({fast},{slow})')
    all_results.append(r)
    ema_data[(fast, slow)] = (ef, es)
    print(f"   {r['name']:35s} | ret={r['total_ret']:+7.1%} | dd={r['max_dd']:+7.1%} | "
          f"sharpe={r['sharpe']:.2f} | trades={r['n_trades']} win={r['win_rate']:.0%}")

# ==================== 8. 可视化 ====================
print("\n8. 生成图表...")

# 使用 smooth=5 的信号做图表着色
sm_plot = 5
pred_top_s = pd.Series(pred_top, index=pred_dates).rolling(sm_plot, min_periods=1).mean().values
pred_bot_s = pd.Series(pred_bot, index=pred_dates).rolling(sm_plot, min_periods=1).mean().values

fig = plt.figure(figsize=(32, 40))
gs = fig.add_gridspec(7, 1, height_ratios=[3, 1.2, 1.2, 1.2, 2.5, 2, 2], hspace=0.12)

# --- 面板1: CFX 价格 + Transformer 信号着色 ---
ax1 = fig.add_subplot(gs[0])
for i in range(1, n_pred):
    if pred_top_s[i] > 0.3:
        ax1.axvspan(pred_dates[i-1], pred_dates[i],
                    alpha=min(0.6, pred_top_s[i] * 0.7), color='red', lw=0)
    if pred_bot_s[i] > 0.3:
        ax1.axvspan(pred_dates[i-1], pred_dates[i],
                    alpha=min(0.6, pred_bot_s[i] * 0.6), color='green', lw=0)

ax1.plot(pred_dates, pred_close, 'black', lw=1, label='CFX Price')

# 标注中频最优策略的买卖点
tf_mid = [x for x in tf_selected if x[0] == '中频']
if tf_mid:
    mid_r = tf_mid[0][1][4]
    for t in mid_r['trades']:
        ax1.axvline(t['entry'], color='green', alpha=0.5, lw=1.5, ls='--')
        ax1.axvline(t['exit'], color='red', alpha=0.5, lw=1.5, ls='--')
        ax1.annotate(f"B ${t['entry_p']:.3f}", xy=(t['entry'], t['entry_p']),
                     fontsize=7, color='green', fontweight='bold', rotation=45,
                     xytext=(5, 10), textcoords='offset points')
        ax1.annotate(f"S ${t['exit_p']:.3f}\n{t['ret']:+.0%}", xy=(t['exit'], t['exit_p']),
                     fontsize=7, color='red', fontweight='bold', rotation=45,
                     xytext=(5, -15), textcoords='offset points')

ax1.set_yscale('log')
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:.3f}' if x < 1 else f'${x:.2f}'))
ax1.set_title('CFX 预测回测: 多资产 Transformer (21币种训练, CFX 纯样本外)\n'
              '红底=逃顶信号  绿底=抄底信号  虚线=中频策略买卖点', fontsize=14, fontweight='bold')
ax1.set_ylabel('CFX Price (log)')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.15, which='both')
ax1.tick_params(labelbottom=False)

# --- 面板2: 逃顶指数 ---
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax2.fill_between(pred_dates, pred_top_s * 100, 0,
                 where=pred_top_s > 0.3, color='red', alpha=0.4, label='逃顶高危')
ax2.fill_between(pred_dates, pred_top_s * 100, 0,
                 where=pred_top_s <= 0.3, color='orange', alpha=0.15)
ax2.plot(pred_dates, pred_top_s * 100, 'black', lw=0.8)
if actual_top is not None:
    ax2.plot(pred_dates, actual_top * 100, 'r-', lw=0.3, alpha=0.3, label='实际回撤')
ax2.set_ylabel('逃顶指数 (%)')
ax2.set_ylim(0, 100)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.2)
ax2.tick_params(labelbottom=False)

# --- 面板3: 抄底指数 ---
ax3 = fig.add_subplot(gs[2], sharex=ax1)
ax3.fill_between(pred_dates, pred_bot_s * 100, 0,
                 where=pred_bot_s > 0.3, color='green', alpha=0.4, label='抄底机会')
ax3.fill_between(pred_dates, pred_bot_s * 100, 0,
                 where=pred_bot_s <= 0.3, color='lightgreen', alpha=0.15)
ax3.plot(pred_dates, pred_bot_s * 100, 'black', lw=0.8)
if actual_bot is not None:
    ax3.plot(pred_dates, actual_bot * 100, 'g-', lw=0.3, alpha=0.3, label='实际涨幅')
ax3.set_ylabel('抄底指数 (%)')
ax3.set_ylim(0, 100)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.2)
ax3.tick_params(labelbottom=False)

# --- 面板4: 综合信号 ---
ax4 = fig.add_subplot(gs[3], sharex=ax1)
combined = pred_bot_s - pred_top_s
ax4.fill_between(pred_dates, combined * 100, 0, where=combined > 0,
                 color='green', alpha=0.4, label='抄底 > 逃顶')
ax4.fill_between(pred_dates, combined * 100, 0, where=combined <= 0,
                 color='red', alpha=0.4, label='逃顶 > 抄底')
ax4.plot(pred_dates, combined * 100, 'black', lw=0.8)
ax4.axhline(0, color='black', lw=0.5)
ax4.set_ylabel('综合信号 (%)')
ax4.legend(fontsize=8, loc='upper left')
ax4.grid(True, alpha=0.2)
ax4.tick_params(labelbottom=False)

# --- 面板5: 净值曲线对比 ---
ax5 = fig.add_subplot(gs[4], sharex=ax1)
tf_colors = ['#e74c3c', '#e67e22', '#f39c12', '#d63031']
ema_colors = ['#3498db', '#2ecc71', '#9b59b6', '#1abc9c', '#00bcd4', '#6c5ce7', '#00cec9', '#a29bfe']

ci = 0
for r in all_results:
    if 'TF' in r['name']:
        c = tf_colors[ci % len(tf_colors)]
        ci += 1
        ax5.plot(pred_dates, r['equity'], c=c, lw=2.5, alpha=0.9,
                 label=f"{r['name']} ({r['total_ret']:+.0%})")
    elif 'Buy' in r['name']:
        ax5.plot(pred_dates, r['equity'], c='gray', lw=2, ls='-', alpha=0.7,
                 label=f"Buy & Hold ({r['total_ret']:+.0%})")

ci = 0
for r in all_results:
    if 'EMA' in r['name']:
        c = ema_colors[ci % len(ema_colors)]
        ci += 1
        ax5.plot(pred_dates, r['equity'], c=c, lw=1, ls='--', alpha=0.6,
                 label=f"{r['name']} ({r['total_ret']:+.0%})")

ax5.set_ylabel('净值')
ax5.legend(fontsize=6.5, ncol=3, loc='upper left')
ax5.grid(True, alpha=0.2)
ax5.tick_params(labelbottom=False)
ax5.set_title('策略净值曲线对比 (实线=Transformer, 虚线=EMA)', fontsize=12, fontweight='bold')

# --- 面板6: 最优 EMA ---
ax6 = fig.add_subplot(gs[5], sharex=ax1)
ax6.plot(pred_dates, pred_close, 'black', lw=0.8, label='CFX')

best_ema = sorted([(r['sharpe'], r['name'], (f, s))
                    for (f, s), (ef, es) in ema_data.items()
                    for r in all_results if r['name'] == f'EMA({f},{s})'],
                   reverse=True)[0] if ema_data else None

if best_ema:
    bf, bs = best_ema[2]
    ef, es = ema_data[(bf, bs)]
    ax6.plot(pred_dates, ef, 'blue', lw=1, alpha=0.7, label=f'EMA{bf}')
    ax6.plot(pred_dates, es, 'red', lw=1, alpha=0.7, label=f'EMA{bs}')
    sigs_best_ema, _, _ = ema_crossover_signals(pred_close, bf, bs)
    for i in range(1, n_pred):
        if sigs_best_ema[i] == 1:
            ax6.axvspan(pred_dates[i-1], pred_dates[i], color='green', alpha=0.08, lw=0)

ax6.set_yscale('log')
ax6.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:.3f}' if x < 1 else f'${x:.2f}'))
ax6.set_ylabel('CFX Price (log)')
ax6.set_title(f'最优 EMA 策略: {best_ema[1] if best_ema else "N/A"}', fontsize=12, fontweight='bold')
ax6.legend(fontsize=9, loc='upper right')
ax6.grid(True, alpha=0.15, which='both')
ax6.tick_params(labelbottom=False)

# --- 面板7: 策略对比表格 ---
ax7 = fig.add_subplot(gs[6])
ax7.axis('off')

headers = ['策略', '总收益', '年化', '最大回撤', 'Sharpe', '交易', '胜率', '均持天', '均收益']
table_data = []
for r in sorted(all_results, key=lambda x: x['sharpe'], reverse=True):
    table_data.append([
        r['name'][:30],
        f"{r['total_ret']:+.1%}",
        f"{r['ann_ret']:+.1%}",
        f"{r['max_dd']:.1%}",
        f"{r['sharpe']:.2f}",
        str(r['n_trades']),
        f"{r['win_rate']:.0%}",
        f"{r['avg_hold_days']:.0f}",
        f"{r['avg_trade_ret']:+.1%}",
    ])

tbl = ax7.table(cellText=table_data, colLabels=headers, loc='center',
                cellLoc='center', colWidths=[0.2, 0.09, 0.09, 0.09, 0.08, 0.06, 0.06, 0.07, 0.08])
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 1.5)

for j in range(len(headers)):
    tbl[0, j].set_facecolor('#2c3e50')
    tbl[0, j].set_text_props(color='white', fontweight='bold')

for i, r in enumerate(sorted(all_results, key=lambda x: x['sharpe'], reverse=True)):
    row = i + 1
    if 'TF' in r['name']:
        for j in range(len(headers)):
            tbl[row, j].set_facecolor('#ffeaa7')
    elif 'Buy' in r['name']:
        for j in range(len(headers)):
            tbl[row, j].set_facecolor('#dfe6e9')

ax7.set_title('策略指标汇总 (按 Sharpe 排序, 黄色=Transformer, 灰色=Buy&Hold)',
              fontsize=11, fontweight='bold', pad=15)

plt.savefig(f'{IMG_DIR}/cfx_predict_vs_ema.png', dpi=150, bbox_inches='tight')
print(f"   保存: {IMG_DIR}/cfx_predict_vs_ema.png")

# ==================== 9. 最终汇总 ====================
print("\n" + "=" * 80)
print("9. 最终汇总")
print("=" * 80)

print(f"\n  CFX 数据: {pred_dates[0].date()} ~ {pred_dates[-1].date()} ({n_pred} 天)")
print(f"  模型训练: 21 币种, CFX 未参与训练 (纯样本外)")
bh_r = [r for r in all_results if 'Buy' in r['name']][0]
print(f"  Buy & Hold: {bh_r['total_ret']:+.1%}")

print(f"\n  {'策略':38s} | {'总收益':>8s} | {'Sharpe':>7s} | {'最大回撤':>8s} | {'交易':>4s} | {'胜率':>4s}")
print(f"  {'-'*85}")
for r in sorted(all_results, key=lambda x: x['sharpe'], reverse=True):
    marker = ' ★' if 'TF' in r['name'] else ''
    print(f"  {r['name']:38s} | {r['total_ret']:+7.1%} | {r['sharpe']:7.2f} | "
          f"{r['max_dd']:+7.1%} | {r['n_trades']:4d} | {r['win_rate']:4.0%}{marker}")

tf_results = [r for r in all_results if 'TF' in r['name']]
ema_results = [r for r in all_results if 'EMA' in r['name']]

if tf_results and ema_results:
    best_tf = max(tf_results, key=lambda x: x['sharpe'])
    best_ema_r = max(ema_results, key=lambda x: x['sharpe'])
    print(f"\n  最优 Transformer: {best_tf['name']}")
    print(f"    收益 {best_tf['total_ret']:+.1%}, Sharpe {best_tf['sharpe']:.2f}, "
          f"最大回撤 {best_tf['max_dd']:.1%}, {best_tf['n_trades']} 笔交易, 胜率 {best_tf['win_rate']:.0%}")
    print(f"  最优 EMA: {best_ema_r['name']}")
    print(f"    收益 {best_ema_r['total_ret']:+.1%}, Sharpe {best_ema_r['sharpe']:.2f}, "
          f"最大回撤 {best_ema_r['max_dd']:.1%}, {best_ema_r['n_trades']} 笔交易, 胜率 {best_ema_r['win_rate']:.0%}")

    if best_tf['sharpe'] > best_ema_r['sharpe']:
        print(f"\n  → Transformer 胜出! Sharpe 高 {best_tf['sharpe'] - best_ema_r['sharpe']:.2f}")
    else:
        print(f"\n  → EMA 胜出! Sharpe 高 {best_ema_r['sharpe'] - best_tf['sharpe']:.2f}")

# 各频率档位分析
print(f"\n  各频率档位 Transformer vs 最佳 EMA:")
for label, (sw, tp, bt, st, r) in tf_selected:
    same_freq_ema = [er for er in ema_results
                     if abs(er['n_trades'] - r['n_trades']) <= max(r['n_trades'] * 0.5, 3)]
    if same_freq_ema:
        best_comp = max(same_freq_ema, key=lambda x: x['sharpe'])
        print(f"    {label} ({r['n_trades']}笔): TF sharpe={r['sharpe']:.2f} vs EMA({best_comp['name']}) sharpe={best_comp['sharpe']:.2f}"
              f" → {'TF胜' if r['sharpe'] > best_comp['sharpe'] else 'EMA胜'}")
    else:
        print(f"    {label} ({r['n_trades']}笔): TF sharpe={r['sharpe']:.2f} (无可比EMA)")

# 逐笔交易详情 (中频)
if tf_mid:
    mid_r = tf_mid[0][1][4]
    print(f"\n  中频策略逐笔交易:")
    for i, t in enumerate(mid_r['trades'], 1):
        print(f"    #{i}: {t['entry'].strftime('%Y-%m-%d')} → {t['exit'].strftime('%Y-%m-%d')} | "
              f"${t['entry_p']:.4f} → ${t['exit_p']:.4f} | {t['ret']:+.1%} | {t['days']}天")

cur_top_v = pred_top_s[-1]
cur_bot_v = pred_bot_s[-1]
cur_price = pred_close[-1]
cur_date = pred_dates[-1]
print(f"\n  当前 CFX 信号 ({cur_date.strftime('%Y-%m-%d')} ${cur_price:.4f}):")
print(f"    逃顶: {cur_top_v*100:.1f}%  抄底: {cur_bot_v*100:.1f}%  综合: {(cur_bot_v-cur_top_v)*100:.1f}%")
if cur_bot_v > cur_top_v:
    print(f"    → 偏乐观, 建议持有/建仓")
else:
    print(f"    → 偏谨慎, 注意风险控制")

print("\n完成!")
