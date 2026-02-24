#!/usr/bin/env python3
"""
牛市顶部风险预测模型 v2 (价格 + 情绪)

数据: BTC 日线 + Fear & Greed Index + Google Trends
方法: 历史大顶共性特征提取 → 多因子复合打分 → 顶部风险概率 0~100%
"""
import os, sys, functools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings('ignore')
print = functools.partial(print, flush=True)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

IMG_DIR = '/Users/king/quant/market-top-detector/img'
os.makedirs(IMG_DIR, exist_ok=True)

# ==================== 数据 ====================
print("1. 加载数据...")
df = pd.read_csv('/Users/king/quant/market-top-detector/data/BTC_daily_full.csv',
                  parse_dates=['Date'], index_col='Date')
df = df.sort_index().dropna(subset=['Close'])
print(f"   {len(df)} 天, {df.index[0].date()} ~ {df.index[-1].date()}")

MAJOR_TOPS = [
    pd.Timestamp('2014-11-12'),
    pd.Timestamp('2017-12-16'),
    pd.Timestamp('2019-06-26'),
    pd.Timestamp('2021-11-08'),
]
MAJOR_BOTTOMS = [
    pd.Timestamp('2015-01-14'),
    pd.Timestamp('2018-12-15'),
    pd.Timestamp('2020-03-12'),
    pd.Timestamp('2022-11-21'),
]

# ==================== 加载情绪数据 ====================
print("\n   加载 Fear & Greed Index...")
fng = pd.read_csv('/Users/king/quant/market-top-detector/data/fear_greed_index.csv',
                   parse_dates=['Date'], index_col='Date')
fng['fng_value'] = pd.to_numeric(fng['fng_value'], errors='coerce')
fng = fng[~fng.index.duplicated(keep='first')].sort_index()
df = df.join(fng[['fng_value']], how='left')
df['fng_value'] = df['fng_value'].ffill()
print(f"   FNG: {fng.index[0].date()} ~ {fng.index[-1].date()}, 有效={df['fng_value'].notna().sum()}天")

print("   加载 Google Trends...")
gt = pd.read_csv('/Users/king/quant/market-top-detector/data/google_trends_bitcoin.csv',
                  parse_dates=['date'], index_col='date')
gt = gt.rename(columns={'Bitcoin': 'gtrend'})
gt = gt[~gt.index.duplicated(keep='first')].sort_index()
df = df.join(gt[['gtrend']], how='left')
df['gtrend'] = df['gtrend'].ffill()
print(f"   GT: {gt.index[0].date()} ~ {gt.index[-1].date()}, 有效={df['gtrend'].notna().sum()}天")

# ==================== 特征工程 ====================
print("\n2. 计算周期特征...")
c = df['Close']

# --- A. 均线偏离 ---
for w in [50, 100, 200, 350, 700]:
    df[f'ma{w}'] = c.rolling(w).mean()
    df[f'ma{w}_dev'] = (c - df[f'ma{w}']) / df[f'ma{w}'] * 100

# --- B. Pi Cycle Top (111DMA vs 350DMA×2) ---
df['ma111'] = c.rolling(111).mean()
df['ma350x2'] = c.rolling(350).mean() * 2
df['pi_cycle_ratio'] = df['ma111'] / df['ma350x2']

# --- C. ATH 相关 ---
df['ath'] = c.expanding().max()
df['ath_drawdown'] = (c - df['ath']) / df['ath'] * 100  # 负值=距ATH回撤
df['days_since_ath'] = 0
last_ath_day = 0
for i in range(len(df)):
    if c.iloc[i] >= df['ath'].iloc[i] * 0.999:
        last_ath_day = i
    df.iloc[i, df.columns.get_loc('days_since_ath')] = i - last_ath_day

# --- D. 波动率 ---
df['ret'] = c.pct_change()
for w in [30, 90, 365]:
    df[f'vol_{w}d'] = df['ret'].rolling(w).std() * np.sqrt(365) * 100

# --- E. RSI (多周期) ---
def rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - 100 / (1 + rs)

for w in [14, 30, 90]:
    df[f'rsi_{w}'] = rsi(c, w)

# --- F. 价格加速度 (二阶导数) ---
df['ma50_slope'] = df['ma50'].pct_change(20)  # 20日斜率
df['ma50_accel'] = df['ma50_slope'].diff(20)   # 斜率变化

# --- G. 从上一个大底至今的涨幅 ---
df['rally_from_bottom'] = np.nan
for i, bot in enumerate(MAJOR_BOTTOMS):
    if bot in df.index:
        bot_price = c.loc[bot]
        mask = df.index >= bot
        if i < len(MAJOR_BOTTOMS) - 1:
            next_bot = MAJOR_BOTTOMS[i + 1]
            mask = mask & (df.index < next_bot)
        df.loc[mask, 'rally_from_bottom'] = (c[mask] - bot_price) / bot_price * 100
# 第一段: 从数据起点到第一个底
first_bot = MAJOR_BOTTOMS[0]
mask0 = df.index < first_bot
if mask0.any():
    start_p = c.iloc[0]
    df.loc[mask0, 'rally_from_bottom'] = (c[mask0] - start_p) / start_p * 100

# --- H. Mayer Multiple (price / 200DMA) ---
df['mayer'] = c / df['ma200']

# --- I. 月度收益率序列 ---
df['ret_30d'] = c.pct_change(30) * 100
df['ret_90d'] = c.pct_change(90) * 100
df['ret_365d'] = c.pct_change(365) * 100

# --- J. 对数价格的线性回归残差 (去趋势) ---
log_p = np.log(c.values)
x = np.arange(len(log_p))
valid = ~np.isnan(log_p)
from numpy.polynomial import polynomial as P
# 用前半段拟合长期趋势
coeffs = np.polyfit(x[valid], log_p[valid], 1)
df['log_trend'] = np.polyval(coeffs, x)
df['log_detrend'] = log_p - df['log_trend'].values
df['log_detrend_z'] = (df['log_detrend'] - df['log_detrend'].rolling(365).mean()) / \
                       (df['log_detrend'].rolling(365).std() + 1e-8)

# --- K. 情绪衍生特征 ---
# FNG 滚动统计
df['fng_ma30'] = df['fng_value'].rolling(30, min_periods=5).mean()
df['fng_ma90'] = df['fng_value'].rolling(90, min_periods=10).mean()
df['fng_extreme_greed_pct'] = df['fng_value'].rolling(90, min_periods=10).apply(
    lambda x: (x > 75).mean() * 100, raw=True)  # 近90天极度贪婪占比
df['fng_above60_pct'] = df['fng_value'].rolling(30, min_periods=5).apply(
    lambda x: (x > 60).mean() * 100, raw=True)  # 近30天贪婪占比

# Google Trends 衍生
df['gtrend_z'] = (df['gtrend'] - df['gtrend'].rolling(365, min_periods=30).mean()) / \
                  (df['gtrend'].rolling(365, min_periods=30).std() + 1e-8)
df['gtrend_mom'] = df['gtrend'].pct_change(90) * 100  # 90天搜索量变化率

df = df.dropna(subset=['ma200', 'vol_90d', 'rsi_30'])
c = df['Close']
print(f"   有效数据: {len(df)} 天")

# ==================== 分析历史大顶特征 ====================
print("\n3. 历史大顶特征分析...")
print("=" * 100)

top_features = {}
feat_names = ['ma200_dev', 'ma350_dev', 'mayer', 'pi_cycle_ratio',
              'rsi_14', 'rsi_30', 'rsi_90',
              'vol_30d', 'vol_90d',
              'ret_30d', 'ret_90d', 'ret_365d',
              'rally_from_bottom', 'log_detrend_z',
              'ma50_slope', 'days_since_ath',
              'fng_value', 'fng_ma30', 'fng_ma90',
              'fng_extreme_greed_pct', 'fng_above60_pct',
              'gtrend', 'gtrend_z', 'gtrend_mom']

print(f"{'特征':<20}", end='')
for top in MAJOR_TOPS:
    print(f"{'  ' + top.strftime('%Y-%m'):>14}", end='')
print(f"{'    当前':>14}{'    中位数':>12}")
print("-" * 100)

current_vals = {}
for feat in feat_names:
    if feat not in df.columns:
        continue
    vals = []
    print(f"{feat:<20}", end='')
    for top in MAJOR_TOPS:
        nearest = df.index[df.index.searchsorted(top)]
        v = df.loc[nearest, feat] if nearest in df.index else np.nan
        vals.append(v)
        print(f"{v:>14.1f}", end='')
    cur = df[feat].iloc[-1]
    current_vals[feat] = cur
    med = np.nanmedian(vals)
    print(f"{cur:>14.1f}{med:>12.1f}")
    top_features[feat] = {'tops': vals, 'median': med, 'current': cur}

# ==================== 复合打分模型 ====================
print("\n4. 构建顶部风险打分模型...")

def score_indicator(value, top_median, direction='above', low_threshold=None, high_threshold=None):
    """
    根据当前值与历史大顶中位数的关系打分 0~100
    direction='above': 值越高越危险 (如RSI, 均线偏离)
    """
    if np.isnan(value) or np.isnan(top_median):
        return 30
    if direction == 'above':
        if high_threshold is None:
            high_threshold = top_median
        if low_threshold is None:
            low_threshold = high_threshold * 0.3
        if value <= 0 and low_threshold > 0:
            return max(0, 5)
        if value >= high_threshold:
            extra = min(30, 30 * (value - high_threshold) / (abs(high_threshold) * 0.5 + 1e-8))
            return min(100, max(0, 70 + extra))
        elif value >= low_threshold:
            return max(0, min(70, 20 + 50 * (value - low_threshold) / (high_threshold - low_threshold + 1e-8)))
        else:
            return max(0, min(20, 20 * max(0, value) / (abs(low_threshold) + 1e-8)))
    else:
        return 30

# 定义因子权重和打分规则
scoring_rules = {
    # 价格因子 (60%)
    'ma200_dev':        {'weight': 10, 'direction': 'above'},
    'mayer':            {'weight': 10, 'direction': 'above'},
    'rsi_30':           {'weight': 7,  'direction': 'above'},
    'rsi_90':           {'weight': 7,  'direction': 'above'},
    'ret_90d':          {'weight': 7,  'direction': 'above'},
    'ret_365d':         {'weight': 7,  'direction': 'above'},
    'rally_from_bottom':{'weight': 5,  'direction': 'above'},
    'log_detrend_z':    {'weight': 5,  'direction': 'above'},
    'pi_cycle_ratio':   {'weight': 5,  'direction': 'above'},
    'ma50_slope':       {'weight': 3,  'direction': 'above'},
    # 情绪因子 (40%)  — 大顶时市场极度贪婪 + 搜索量飙升
    'fng_value':           {'weight': 8,  'direction': 'above'},
    'fng_ma30':            {'weight': 8,  'direction': 'above'},
    'fng_extreme_greed_pct':{'weight': 6, 'direction': 'above'},
    'fng_above60_pct':     {'weight': 5,  'direction': 'above'},
    'gtrend':              {'weight': 6,  'direction': 'above'},
    'gtrend_z':            {'weight': 4,  'direction': 'above'},
    'gtrend_mom':          {'weight': 3,  'direction': 'above'},
}

# 计算每日的综合风险评分
print("   计算每日风险评分...")
risk_scores = np.zeros(len(df))

for i in range(len(df)):
    weighted_sum = 0
    total_weight = 0
    for feat, rule in scoring_rules.items():
        if feat not in df.columns:
            continue
        val = df[feat].iloc[i]
        med = top_features.get(feat, {}).get('median', np.nan)
        if np.isnan(val) or np.isnan(med):
            continue

        tops_vals = top_features.get(feat, {}).get('tops', [])
        tops_min = np.nanmin(tops_vals) if tops_vals else med * 0.5
        tops_max = np.nanmax(tops_vals) if tops_vals else med * 1.5

        score = score_indicator(val, med, rule['direction'],
                                low_threshold=tops_min * 0.3 if rule['direction'] == 'above' else tops_min,
                                high_threshold=med if rule['direction'] == 'above' else tops_max)
        weighted_sum += score * rule['weight']
        total_weight += rule['weight']

    risk_scores[i] = weighted_sum / total_weight if total_weight > 0 else 50

df['top_risk'] = risk_scores

# 平滑
df['top_risk_smooth'] = df['top_risk'].rolling(14, min_periods=1).mean()

# ==================== 抄底指数 (Bottom Opportunity Index) ====================
print("\n5. 构建抄底指数...")

# 分析历史大底特征
print("\n  历史大底特征:")
print(f"  {'特征':<24}", end='')
for bot in MAJOR_BOTTOMS:
    print(f"{'  ' + bot.strftime('%Y-%m'):>14}", end='')
print(f"{'    当前':>14}{'    中位数':>12}")
print("  " + "-" * 96)

bottom_features = {}
bot_feat_names = ['ath_drawdown', 'days_since_ath', 'ma200_dev', 'mayer',
                  'rsi_14', 'rsi_30', 'rsi_90',
                  'vol_90d', 'ret_90d', 'ret_365d', 'log_detrend_z',
                  'fng_value', 'fng_ma30', 'fng_ma90',
                  'fng_extreme_greed_pct', 'fng_above60_pct',
                  'gtrend', 'gtrend_z']

for feat in bot_feat_names:
    if feat not in df.columns:
        continue
    vals = []
    print(f"  {feat:<24}", end='')
    for bot in MAJOR_BOTTOMS:
        nearest = df.index[df.index.searchsorted(bot)]
        v = df.loc[nearest, feat] if nearest in df.index else np.nan
        vals.append(v)
        print(f"{v:>14.1f}", end='')
    cur = df[feat].iloc[-1]
    med = np.nanmedian(vals)
    print(f"{cur:>14.1f}{med:>12.1f}")
    bottom_features[feat] = {'bottoms': vals, 'median': med, 'current': cur}

# 抄底打分: 值越接近历史大底 → 抄底机会越高
def score_bottom(value, bot_median, higher_is_bottom=False):
    """
    值越接近/超过历史大底中位数 → 越适合抄底 (0~100)
    higher_is_bottom=False: 值越低越像底部 (如 RSI, Mayer, FNG)
    higher_is_bottom=True:  值越高越像底部 (如 ATH回撤幅度, days_since_ath)
    """
    if np.isnan(value) or np.isnan(bot_median):
        return 30
    if higher_is_bottom:
        # 如 ath_drawdown=-77%, days_since_ath=378
        # 值越大(绝对值越大) → 越像底部
        if bot_median == 0:
            return 30
        ratio = abs(value) / abs(bot_median)
    else:
        # 如 mayer=0.52, rsi=20, fng=10
        # 值越低 → 越像底部; 超过中位一倍以上 → 不是底部
        if bot_median == 0:
            return 30
        if value <= bot_median:
            ratio = 1.0 + (bot_median - value) / (abs(bot_median) + 1e-8) * 0.3
        else:
            ratio = max(0, 1.0 - (value - bot_median) / (abs(bot_median) + 1e-8))

    ratio = max(0, min(2.0, ratio))
    if ratio >= 1.0:
        return min(100, 60 + 40 * (ratio - 1.0))
    elif ratio >= 0.5:
        return 20 + 40 * (ratio - 0.5) / 0.5
    else:
        return max(0, 20 * ratio / 0.5)

bottom_scoring = {
    # 价格因子 — 底部特征: 大幅回撤、低估值、低动量
    'ath_drawdown':    {'weight': 12, 'higher_is_bottom': True},   # 回撤越深越像底部
    'days_since_ath':  {'weight': 8,  'higher_is_bottom': True},   # 距ATH越久越像底部
    'mayer':           {'weight': 12, 'higher_is_bottom': False},  # Mayer越低越像底部
    'ma200_dev':       {'weight': 8,  'higher_is_bottom': False},  # 均线偏离越负越像底部
    'rsi_30':          {'weight': 8,  'higher_is_bottom': False},  # RSI越低越像底部
    'rsi_90':          {'weight': 6,  'higher_is_bottom': False},
    'ret_90d':         {'weight': 6,  'higher_is_bottom': False},  # 近期跌幅越大越像底部
    'ret_365d':        {'weight': 5,  'higher_is_bottom': False},
    'log_detrend_z':   {'weight': 5,  'higher_is_bottom': False},  # 去趋势越负越像底部
    # 情绪因子 — 底部特征: 极度恐惧、市场冷清
    'fng_value':       {'weight': 10, 'higher_is_bottom': False},  # FNG越低越像底部
    'fng_ma30':        {'weight': 8,  'higher_is_bottom': False},
    'fng_above60_pct': {'weight': 5,  'higher_is_bottom': False},  # 贪婪占比越低越像底部
    'gtrend':          {'weight': 4,  'higher_is_bottom': False},  # 搜索量越低越冷清
    'gtrend_z':        {'weight': 3,  'higher_is_bottom': False},
}

print("\n   计算每日抄底指数...")
bottom_scores = np.zeros(len(df))

for i in range(len(df)):
    ws = 0; tw = 0
    for feat, rule in bottom_scoring.items():
        if feat not in df.columns:
            continue
        val = df[feat].iloc[i]
        med = bottom_features.get(feat, {}).get('median', np.nan)
        if np.isnan(val) or np.isnan(med):
            continue
        s = score_bottom(val, med, rule['higher_is_bottom'])
        ws += s * rule['weight']
        tw += rule['weight']
    bottom_scores[i] = ws / tw if tw > 0 else 30

df['bottom_opp'] = bottom_scores
df['bottom_opp_smooth'] = df['bottom_opp'].rolling(14, min_periods=1).mean()

# ==================== 验证 ====================
print("\n6. 历史验证 (顶部风险 + 抄底指数)...")
print("=" * 100)
print(f"{'事件':<8} {'日期':>12} {'价格':>12} {'逃顶指数':>10} {'评价':>8} {'抄底指数':>10} {'评价':>8}")
print("-" * 100)

for top in MAJOR_TOPS:
    nearest = df.index[df.index.searchsorted(top)]
    if nearest in df.index:
        tr = df.loc[nearest, 'top_risk_smooth']
        br = df.loc[nearest, 'bottom_opp_smooth']
        p = df.loc[nearest, 'Close']
        tg = "极高危" if tr > 70 else "高危" if tr > 55 else "中等" if tr > 40 else "安全"
        bg = "强烈抄底" if br > 70 else "可抄底" if br > 55 else "观望" if br > 40 else "远离"
        print(f"{'大顶':<8} {top.strftime('%Y-%m-%d'):>12} ${p:>10,.0f} {tr:>10.1f} {tg:>8} {br:>10.1f} {bg:>8}")

for bot in MAJOR_BOTTOMS:
    nearest = df.index[df.index.searchsorted(bot)]
    if nearest in df.index:
        tr = df.loc[nearest, 'top_risk_smooth']
        br = df.loc[nearest, 'bottom_opp_smooth']
        p = df.loc[nearest, 'Close']
        tg = "极高危" if tr > 70 else "高危" if tr > 55 else "中等" if tr > 40 else "安全"
        bg = "强烈抄底" if br > 70 else "可抄底" if br > 55 else "观望" if br > 40 else "远离"
        print(f"{'大底':<8} {bot.strftime('%Y-%m-%d'):>12} ${p:>10,.0f} {tr:>10.1f} {tg:>8} {br:>10.1f} {bg:>8}")

cur_risk = df['top_risk_smooth'].iloc[-1]
cur_bot = df['bottom_opp_smooth'].iloc[-1]
cur_price = df['Close'].iloc[-1]
cur_date = df.index[-1]
print("-" * 100)
tg = "极高危" if cur_risk > 70 else "高危" if cur_risk > 55 else "中等" if cur_risk > 40 else "安全"
bg = "强烈抄底" if cur_bot > 70 else "可抄底" if cur_bot > 55 else "观望" if cur_bot > 40 else "远离"
print(f"{'>>> 当前':<8} {cur_date.strftime('%Y-%m-%d'):>12} ${cur_price:>10,.0f} {cur_risk:>10.1f} {tg:>8} {cur_bot:>10.1f} {bg:>8}")

# 当前抄底因子明细
print(f"\n当前抄底因子明细:")
for feat, rule in bottom_scoring.items():
    if feat not in df.columns: continue
    val = df[feat].iloc[-1]
    med = bottom_features.get(feat, {}).get('median', np.nan)
    s = score_bottom(val, med, rule['higher_is_bottom'])
    arrow = "↑" if s >= 60 else "→" if s >= 40 else "↓"
    print(f"  {feat:<22} 当前={val:>8.1f}  大底中位={med:>8.1f}  "
          f"评分={s:>5.0f}  {arrow}  权重={rule['weight']}")

# ==================== 可视化 ====================
print("\n7. 生成图表...")

fig, axes = plt.subplots(5, 1, figsize=(30, 36),
                          gridspec_kw={'height_ratios': [4, 1.8, 1.8, 2, 2]})
fig.subplots_adjust(hspace=0.08)

# --- 面板1: 价格 + 风险着色 ---
ax1 = axes[0]
risk_s = df['top_risk_smooth'].values
for i in range(1, len(df)):
    r = risk_s[i]
    if r > 70:
        ax1.axvspan(df.index[i-1], df.index[i], alpha=0.25, color='red', lw=0)
    elif r > 55:
        ax1.axvspan(df.index[i-1], df.index[i], alpha=0.12, color='orange', lw=0)
    elif r < 25:
        ax1.axvspan(df.index[i-1], df.index[i], alpha=0.08, color='green', lw=0)

ax1.plot(df.index, c, 'black', lw=0.8, alpha=0.7)
ax1.plot(df.index, df['ma200'], color='blue', lw=0.5, alpha=0.3, label='200DMA')
ax1.plot(df.index, df['ma350x2'], color='purple', lw=0.5, alpha=0.3, label='350DMA×2')

for top in MAJOR_TOPS:
    nearest = df.index[df.index.searchsorted(top)]
    if nearest in df.index:
        p = df.loc[nearest, 'Close']
        ax1.plot(nearest, p, 'o', color='red', ms=14, zorder=10,
                 markeredgecolor='darkred', markeredgewidth=2)
        ax1.annotate(f'大顶 ${p:,.0f}\n{nearest.strftime("%Y-%m")}',
                     xy=(nearest, p), xytext=(0, 20),
                     textcoords='offset points', fontsize=9, fontweight='bold',
                     color='red', ha='center', va='bottom',
                     bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.9))

for bot in MAJOR_BOTTOMS:
    nearest = df.index[df.index.searchsorted(bot)]
    if nearest in df.index:
        p = df.loc[nearest, 'Close']
        ax1.plot(nearest, p, 'o', color='green', ms=14, zorder=10,
                 markeredgecolor='darkgreen', markeredgewidth=2)
        ax1.annotate(f'大底 ${p:,.0f}\n{nearest.strftime("%Y-%m")}',
                     xy=(nearest, p), xytext=(0, -22),
                     textcoords='offset points', fontsize=9, fontweight='bold',
                     color='green', ha='center', va='top',
                     bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', alpha=0.9))

ax1.set_yscale('log')
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax1.set_ylabel('BTC Price (USD, log)', fontsize=12)
ax1.set_title('BTC 逃顶 + 抄底 双指数模型 (价格+情绪)\n红底=逃顶高危(>70) 橙底=逃顶警戒(>55) 绿底=抄底机会(>55) 深绿=强烈抄底(>70)',
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(True, alpha=0.15, which='both')
ax1.tick_params(labelbottom=False)

# --- 面板2: 顶部风险评分 ---
ax2 = axes[1]
ax2.fill_between(df.index, risk_s, 0, where=risk_s > 70, color='red', alpha=0.5, label='极高危 (>70)')
ax2.fill_between(df.index, risk_s, 0, where=(risk_s > 55) & (risk_s <= 70), color='orange', alpha=0.4, label='高危 (55-70)')
ax2.fill_between(df.index, risk_s, 0, where=(risk_s > 40) & (risk_s <= 55), color='gold', alpha=0.3, label='中等 (40-55)')
ax2.fill_between(df.index, risk_s, 0, where=risk_s <= 40, color='green', alpha=0.3, label='安全 (<40)')
ax2.plot(df.index, risk_s, 'black', lw=0.8)

for top in MAJOR_TOPS:
    ax2.axvline(top, color='red', ls='--', lw=1, alpha=0.5)
for bot in MAJOR_BOTTOMS:
    ax2.axvline(bot, color='green', ls='--', lw=1, alpha=0.5)

ax2.axhline(70, color='red', ls=':', lw=0.5)
ax2.axhline(55, color='orange', ls=':', lw=0.5)
ax2.axhline(40, color='gold', ls=':', lw=0.5)

# 当前值标注
ax2.annotate(f'当前: {cur_risk:.0f}', xy=(cur_date, cur_risk),
             xytext=(-60, 15), textcoords='offset points',
             fontsize=11, fontweight='bold', color='darkred',
             arrowprops=dict(arrowstyle='->', color='darkred'),
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9))

ax2.set_ylabel('顶部风险评分', fontsize=12)
ax2.set_ylim(0, 100)
ax2.legend(fontsize=7, ncol=4, loc='upper left')
ax2.grid(True, alpha=0.2)
ax2.tick_params(labelbottom=False)

# --- 面板3: 抄底指数 ---
ax3 = axes[2]
bot_s = df['bottom_opp_smooth'].values
ax3.fill_between(df.index, bot_s, 0, where=bot_s > 70, color='#006400', alpha=0.5, label='强烈抄底 (>70)')
ax3.fill_between(df.index, bot_s, 0, where=(bot_s > 55) & (bot_s <= 70), color='green', alpha=0.35, label='可抄底 (55-70)')
ax3.fill_between(df.index, bot_s, 0, where=(bot_s > 40) & (bot_s <= 55), color='lightgreen', alpha=0.25, label='观望 (40-55)')
ax3.fill_between(df.index, bot_s, 0, where=bot_s <= 40, color='gray', alpha=0.15, label='远离 (<40)')
ax3.plot(df.index, bot_s, 'black', lw=0.8)

for top in MAJOR_TOPS:
    ax3.axvline(top, color='red', ls='--', lw=0.8, alpha=0.4)
for bot in MAJOR_BOTTOMS:
    ax3.axvline(bot, color='green', ls='--', lw=1, alpha=0.6)

ax3.axhline(70, color='darkgreen', ls=':', lw=0.5)
ax3.axhline(55, color='green', ls=':', lw=0.5)

ax3.annotate(f'当前: {cur_bot:.0f}', xy=(cur_date, cur_bot),
             xytext=(-60, 15), textcoords='offset points',
             fontsize=11, fontweight='bold', color='darkgreen',
             arrowprops=dict(arrowstyle='->', color='darkgreen'),
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

ax3.set_ylabel('抄底指数', fontsize=12)
ax3.set_ylim(0, 100)
ax3.legend(fontsize=7, ncol=4, loc='upper left')
ax3.grid(True, alpha=0.2)
ax3.tick_params(labelbottom=False)

# --- 面板4: RSI + 年化波动率 ---
ax4 = axes[3]
ax4.plot(df.index, df['rsi_30'], color='blue', lw=0.6, alpha=0.7, label='RSI-30')
ax4.axhline(75, color='red', ls=':', lw=0.5)
ax4.axhline(25, color='green', ls=':', lw=0.5)
for top in MAJOR_TOPS:
    ax4.axvline(top, color='red', ls='--', lw=0.8, alpha=0.4)
for bot in MAJOR_BOTTOMS:
    ax4.axvline(bot, color='green', ls='--', lw=0.8, alpha=0.4)

ax4b = ax4.twinx()
ax4b.plot(df.index, df['vol_90d'], color='orange', lw=0.6, alpha=0.5, label='90日年化波动率')
ax4b.set_ylabel('波动率 (%)', fontsize=10, color='orange')

ax4.set_ylabel('RSI-30', fontsize=10, color='blue')
ax4.legend(fontsize=8, loc='upper left')
ax4b.legend(fontsize=8, loc='upper right')
ax4.grid(True, alpha=0.2)

# --- 面板5: Fear & Greed + Google Trends ---
ax5 = axes[4]
fng_valid = df['fng_value'].dropna()
if len(fng_valid) > 0:
    ax5.fill_between(df.index, df['fng_value'], 50,
                     where=df['fng_value'] > 75, color='red', alpha=0.3, label='极度贪婪(>75)')
    ax5.fill_between(df.index, df['fng_value'], 50,
                     where=(df['fng_value'] > 50) & (df['fng_value'] <= 75), color='orange', alpha=0.15)
    ax5.fill_between(df.index, df['fng_value'], 50,
                     where=(df['fng_value'] < 25), color='green', alpha=0.3, label='极度恐惧(<25)')
    ax5.fill_between(df.index, df['fng_value'], 50,
                     where=(df['fng_value'] >= 25) & (df['fng_value'] < 50), color='lightgreen', alpha=0.15)
    ax5.plot(df.index, df['fng_ma30'], color='darkblue', lw=1, label='FNG 30日均值')

for top in MAJOR_TOPS:
    ax5.axvline(top, color='red', ls='--', lw=0.8, alpha=0.4)
for bot in MAJOR_BOTTOMS:
    ax5.axvline(bot, color='green', ls='--', lw=0.8, alpha=0.4)

ax5.axhline(75, color='red', ls=':', lw=0.5); ax5.axhline(25, color='green', ls=':', lw=0.5)
ax5.axhline(50, color='gray', ls='-', lw=0.3)
ax5.set_ylim(0, 100); ax5.set_ylabel('Fear & Greed', fontsize=10)
ax5.legend(fontsize=7, ncol=4, loc='upper left')
ax5.grid(True, alpha=0.2)

ax5b = ax5.twinx()
gt_valid = df['gtrend'].dropna()
if len(gt_valid) > 0:
    ax5b.plot(df.index, df['gtrend'], color='purple', lw=0.8, alpha=0.5, label='Google Trends "Bitcoin"')
    ax5b.set_ylabel('Google Trends', fontsize=10, color='purple')
    ax5b.legend(fontsize=7, loc='upper right')

plt.savefig(f'{IMG_DIR}/top_risk_model_v2.png', dpi=150, bbox_inches='tight')
print(f"  保存: {IMG_DIR}/top_risk_model_v2.png")

# ==================== 当前详细评估 ====================
print("\n" + "=" * 80)
print("8. 当前市场综合评估")
print("=" * 80)

print(f"\n  日期: {cur_date.strftime('%Y-%m-%d')}")
print(f"  价格: ${cur_price:,.0f}")
print(f"  ┌──────────────────────────────────────────┐")
print(f"  │  逃顶指数: {cur_risk:>5.1f}/100  →  {tg:<8}         │")
print(f"  │  抄底指数: {cur_bot:>5.1f}/100  →  {bg:<8}         │")
print(f"  └──────────────────────────────────────────┘")

# 综合判断
if cur_risk > 70:
    verdict = "极度危险，强烈建议清仓离场"
elif cur_risk > 55:
    verdict = "高位风险，建议逐步减仓"
elif cur_bot > 70:
    verdict = "极度低估，强烈建议抄底建仓"
elif cur_bot > 55:
    verdict = "底部区域，建议分批建仓"
elif cur_bot > 40 and cur_risk < 35:
    verdict = "偏低估，可考虑小额加仓"
elif cur_risk > 40:
    verdict = "中性偏高，谨慎持有"
else:
    verdict = "中性区间，正常持有"

print(f"\n  >>> 综合判断: {verdict}")

print(f"\n  === 逃顶因子 (当前 vs 大顶中位) ===")
for feat in ['ma200_dev', 'mayer', 'rsi_30', 'ret_90d', 'ret_365d', 'fng_value', 'fng_ma30']:
    cv = current_vals.get(feat, 0)
    tm = top_features.get(feat, {}).get('median', 0)
    pct = cv / tm * 100 if tm != 0 else 0
    bar = "█" * max(0, min(20, int(pct / 5))) + "░" * max(0, 20 - min(20, int(pct / 5)))
    print(f"    {feat:<22} {cv:>8.1f} / {tm:>8.1f}  [{bar}] {pct:.0f}%")

print(f"\n  === 抄底因子 (当前 vs 大底中位) ===")
for feat, rule in bottom_scoring.items():
    if feat not in df.columns: continue
    cv = df[feat].iloc[-1]
    bm = bottom_features.get(feat, {}).get('median', 0)
    s = score_bottom(cv, bm, rule['higher_is_bottom'])
    bar_len = int(s / 5)
    bar = "█" * bar_len + "░" * (20 - bar_len)
    flag = " ★" if s >= 60 else ""
    print(f"    {feat:<22} {cv:>8.1f} / {bm:>8.1f}  [{bar}] {s:.0f}{flag}")

print(f"\n  指数定义:")
print(f"    逃顶 70+: 极高危  55-70: 高危  40-55: 中等  <40: 安全")
print(f"    抄底 70+: 强烈抄底  55-70: 可抄底  40-55: 观望  <40: 远离")

print("\n完成!")
