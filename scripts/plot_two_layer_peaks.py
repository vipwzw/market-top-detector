#!/usr/bin/env python3
"""
绘制 BTC 两层标注高低点 (大顶/大底 + 小顶/小底)
使用 2014-2026 完整日线数据，覆盖多轮牛熊周期
"""
import sys, os
sys.path.insert(0, '/Users/king/quant/btcdata')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from models.bull_bear_detection import (
    multi_level_turning_points, adaptive_multi_level_turning_points,
    TurningPointLevel
)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

IMG_DIR = '/Users/king/quant/market-top-detector/img'
os.makedirs(IMG_DIR, exist_ok=True)

# ==================== 加载数据 ====================
print("加载 BTC 完整日线数据...")
df = pd.read_csv('/Users/king/quant/market-top-detector/data/BTC_daily_full.csv',
                  parse_dates=['Date'], index_col='Date')
df = df.sort_index().dropna(subset=['Close'])
prices = df['Close'].values
timestamps = df.index
print(f"  {len(df)} 天, {timestamps[0].date()} ~ {timestamps[-1].date()}")
print(f"  价格范围: ${prices.min():,.0f} ~ ${prices.max():,.0f}")

# ==================== 两层检测 ====================
print("\n运行自适应多层转折点检测...")
regimes, tps, metrics = adaptive_multi_level_turning_points(
    prices,
    timestamps=timestamps.values,
    volatility_window=252,
    min_major_duration=40,
    min_minor_duration=12,
    window=10,
    verbose=True
)

major_peaks = [t for t in tps if t.is_major and t.point_type == 'peak']
major_troughs = [t for t in tps if t.is_major and t.point_type == 'trough']
minor_peaks = [t for t in tps if not t.is_major and t.point_type == 'peak']
minor_troughs = [t for t in tps if not t.is_major and t.point_type == 'trough']
print(f"\n  大顶: {len(major_peaks)} | 大底: {len(major_troughs)} | 小顶: {len(minor_peaks)} | 小底: {len(minor_troughs)}")

# ==================== 图1: 完整K线 + 两层标注 ====================
print("\n绘制图表...")

fig, axes = plt.subplots(3, 1, figsize=(32, 24),
                          gridspec_kw={'height_ratios': [6, 1.5, 0.8]})
fig.subplots_adjust(hspace=0.06)

ax = axes[0]

# 背景: 牛熊着色
for i in range(1, len(regimes)):
    if regimes[i] == 1:
        ax.axvspan(timestamps[i-1], timestamps[i], alpha=0.06, color='green', lw=0)
    elif regimes[i] == -1:
        ax.axvspan(timestamps[i-1], timestamps[i], alpha=0.06, color='red', lw=0)

# 价格线
ax.plot(timestamps, prices, color='#333333', lw=0.8, alpha=0.6, zorder=2)

# 全部骨架线 (细)
tp_times = [timestamps[t.index] for t in tps]
tp_prices = [t.price for t in tps]
ax.plot(tp_times, tp_prices, color='#aaaaaa', lw=0.5, ls='-', alpha=0.4, zorder=3)

# 大周期骨架线 (粗)
major_tps = [t for t in tps if t.is_major]
if major_tps:
    mt_times = [timestamps[t.index] for t in major_tps]
    mt_prices = [t.price for t in major_tps]
    ax.plot(mt_times, mt_prices, color='#444444', lw=2.5, ls='-', alpha=0.5, zorder=4)

# 绘制转折点
for tp in tps:
    t = timestamps[tp.index]
    p = tp.price

    if tp.is_major:
        if tp.point_type == 'peak':
            ax.plot(t, p, 'o', color='#d62728', ms=16, zorder=10,
                    markeredgecolor='darkred', markeredgewidth=2)
            ax.annotate(f'大顶\n${p:,.0f}\n{t.strftime("%Y-%m")}',
                        xy=(t, p), xytext=(0, 22),
                        textcoords='offset points', fontsize=10, fontweight='bold',
                        color='#d62728', ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff0f0',
                                  edgecolor='#d62728', linewidth=1.5, alpha=0.95))
        else:
            ax.plot(t, p, 'o', color='#2ca02c', ms=16, zorder=10,
                    markeredgecolor='darkgreen', markeredgewidth=2)
            ax.annotate(f'大底\n${p:,.0f}\n{t.strftime("%Y-%m")}',
                        xy=(t, p), xytext=(0, -26),
                        textcoords='offset points', fontsize=10, fontweight='bold',
                        color='#2ca02c', ha='center', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0fff0',
                                  edgecolor='#2ca02c', linewidth=1.5, alpha=0.95))
    else:
        if tp.point_type == 'peak':
            ax.plot(t, p, 'v', color='#ff9999', ms=6, zorder=8,
                    markeredgecolor='#cc4444', markeredgewidth=0.7)
            ax.annotate('顶', xy=(t, p), xytext=(0, 8),
                        textcoords='offset points', fontsize=7, color='#cc4444',
                        ha='center', va='bottom', alpha=0.7)
        else:
            ax.plot(t, p, '^', color='#99ff99', ms=6, zorder=8,
                    markeredgecolor='#44aa44', markeredgewidth=0.7)
            ax.annotate('底', xy=(t, p), xytext=(0, -10),
                        textcoords='offset points', fontsize=7, color='#44aa44',
                        ha='center', va='top', alpha=0.7)

ax.set_yscale('log')
ax.set_ylabel('BTC Price (USD, log scale)', fontsize=12)
ax.set_title('BTC 两层转折点标注 — 大顶/大底 + 小顶/小底\n'
             f'(2014-09 ~ 2026-02, D1, 自适应阈值, 绿底=牛市 红底=熊市)',
             fontsize=15, fontweight='bold')
ax.grid(True, alpha=0.15, which='both')
ax.tick_params(labelbottom=False)

# Y轴价格格式
from matplotlib.ticker import FuncFormatter
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', ms=13,
           markeredgecolor='darkred', markeredgewidth=2, label='大顶 (Major Peak)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', ms=13,
           markeredgecolor='darkgreen', markeredgewidth=2, label='大底 (Major Trough)'),
    Line2D([0], [0], marker='v', color='w', markerfacecolor='#ff9999', ms=7,
           markeredgecolor='#cc4444', label='小顶 (Minor Peak)'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='#99ff99', ms=7,
           markeredgecolor='#44cc44', label='小底 (Minor Trough)'),
    Line2D([0], [0], color='#444444', lw=2.5, alpha=0.5, label='大周期骨架'),
    Line2D([0], [0], color='#aaaaaa', lw=0.5, alpha=0.5, label='全部骨架'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9, ncol=2,
          framealpha=0.95, facecolor='white')

# ==================== 面板2: 大周期涨跌幅 ====================
ax2 = axes[1]
for i in range(1, len(major_tps)):
    t0 = timestamps[major_tps[i-1].index]
    t1 = timestamps[major_tps[i].index]
    p0 = major_tps[i-1].price
    p1 = major_tps[i].price
    chg = (p1 - p0) / p0 * 100
    dur = (t1 - t0).days
    mid_t = t0 + (t1 - t0) / 2
    color = '#2ca02c' if chg > 0 else '#d62728'
    ax2.bar(mid_t, chg, width=(t1-t0)*0.75, color=color, alpha=0.5, edgecolor=color)
    # 标注: 涨跌幅 + 天数
    yoff = 15 if chg > 0 else -15
    va = 'bottom' if chg > 0 else 'top'
    ax2.annotate(f'{chg:+.0f}%\n{dur}天',
                 xy=(mid_t, chg), xytext=(0, yoff),
                 textcoords='offset points', ha='center', va=va,
                 fontsize=8, fontweight='bold', color=color)

ax2.axhline(0, color='black', lw=0.5)
ax2.set_ylabel('涨跌幅 (%)', fontsize=10)
ax2.grid(True, alpha=0.2)
ax2.tick_params(labelbottom=False)

# ==================== 面板3: 牛熊状态 ====================
ax3 = axes[2]
bull = regimes == 1; bear = regimes == -1
ax3.fill_between(timestamps, 0, 1, where=bull, color='green', alpha=0.4, label='牛市')
ax3.fill_between(timestamps, 0, -1, where=bear, color='red', alpha=0.4, label='熊市')
ax3.set_ylim(-1.3, 1.3)
ax3.set_ylabel('牛/熊', fontsize=10)
ax3.set_yticks([-1, 0, 1]); ax3.set_yticklabels(['熊', '', '牛'])
ax3.legend(loc='upper left', fontsize=8, ncol=2)
ax3.grid(True, alpha=0.2)

plt.savefig(f'{IMG_DIR}/btc_two_layer_full.png', dpi=150, bbox_inches='tight')
print(f"  保存: {IMG_DIR}/btc_two_layer_full.png")

# ==================== 图2: 手绘骨架图 ====================
fig2, ax2b = plt.subplots(figsize=(30, 12))

# 骨架线
ax2b.plot(tp_times, tp_prices, color='#555555', lw=1.8, zorder=5)
if major_tps:
    ax2b.plot(mt_times, mt_prices, color='#333333', lw=3.5, zorder=6, alpha=0.3)

for tp in tps:
    t = timestamps[tp.index]
    p = tp.price
    if tp.is_major:
        if tp.point_type == 'peak':
            ax2b.plot(t, p, 'o', color='#d62728', ms=20, zorder=10,
                      markeredgecolor='darkred', markeredgewidth=2.5)
            ax2b.annotate(f'大顶\n${p:,.0f}\n{t.strftime("%Y-%m")}',
                          xy=(t, p), xytext=(0, 24),
                          textcoords='offset points', fontsize=12, fontweight='bold',
                          color='#d62728', ha='center', va='bottom',
                          bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                    edgecolor='#d62728', linewidth=2, alpha=0.95))
        else:
            ax2b.plot(t, p, 'o', color='#2ca02c', ms=20, zorder=10,
                      markeredgecolor='darkgreen', markeredgewidth=2.5)
            ax2b.annotate(f'大底\n${p:,.0f}\n{t.strftime("%Y-%m")}',
                          xy=(t, p), xytext=(0, -28),
                          textcoords='offset points', fontsize=12, fontweight='bold',
                          color='#2ca02c', ha='center', va='top',
                          bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                    edgecolor='#2ca02c', linewidth=2, alpha=0.95))
    else:
        if tp.point_type == 'peak':
            ax2b.plot(t, p, 'o', color='#ff7777', ms=8, zorder=8,
                      markeredgecolor='#cc3333', markeredgewidth=1)
            ax2b.annotate('顶', xy=(t, p), xytext=(0, 12),
                          textcoords='offset points', fontsize=9, color='#cc3333',
                          ha='center', va='bottom', fontweight='bold')
        else:
            ax2b.plot(t, p, 'o', color='#77dd77', ms=8, zorder=8,
                      markeredgecolor='#33aa33', markeredgewidth=1)
            ax2b.annotate('底', xy=(t, p), xytext=(0, -14),
                          textcoords='offset points', fontsize=9, color='#33aa33',
                          ha='center', va='top', fontweight='bold')

# 大周期间涨跌幅标注
for i in range(1, len(major_tps)):
    tp0 = major_tps[i-1]; tp1 = major_tps[i]
    t0 = timestamps[tp0.index]; t1 = timestamps[tp1.index]
    chg = (tp1.price - tp0.price) / tp0.price * 100
    dur = (t1 - t0).days
    mid_t = t0 + (t1 - t0) / 2
    mid_p = np.exp((np.log(tp0.price) + np.log(tp1.price)) / 2)
    color = '#2ca02c' if chg > 0 else '#d62728'
    ax2b.annotate(f'{chg:+.0f}%\n{dur}天',
                  xy=(mid_t, mid_p), fontsize=9, color=color,
                  ha='center', va='center', fontweight='bold',
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85, edgecolor=color))

ax2b.set_yscale('log')
ax2b.set_ylabel('BTC Price (USD, log)', fontsize=12)
ax2b.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax2b.set_title('BTC 两层高低点骨架图 — 大顶/大底/顶/底 (2014 ~ 2026, D1)',
               fontsize=16, fontweight='bold')
ax2b.grid(True, alpha=0.12, which='both')

legend2 = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', ms=15,
           markeredgecolor='darkred', markeredgewidth=2, label='大顶'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', ms=15,
           markeredgecolor='darkgreen', markeredgewidth=2, label='大底'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7777', ms=8,
           markeredgecolor='#cc3333', label='顶'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#77dd77', ms=8,
           markeredgecolor='#33aa33', label='底'),
]
ax2b.legend(handles=legend2, loc='upper left', fontsize=11, framealpha=0.95)

plt.savefig(f'{IMG_DIR}/btc_skeleton_full.png', dpi=150, bbox_inches='tight')
print(f"  保存: {IMG_DIR}/btc_skeleton_full.png")

# ==================== 统计 ====================
print("\n" + "=" * 80)
print("转折点统计 (2014 ~ 2026)")
print("=" * 80)

for tp in tps:
    level = "【大】" if tp.is_major else "    "
    kind = "顶" if tp.point_type == 'peak' else "底"
    dt = timestamps[tp.index]
    print(f"  {level}{kind}  {dt.strftime('%Y-%m-%d')}  ${tp.price:>10,.0f}  "
          f"涨跌幅={tp.drawdown:>+8.1f}%  持续={tp.duration:>4d}天")

print(f"\n大顶: {len(major_peaks)}")
for t in major_peaks:
    print(f"  {timestamps[t.index].strftime('%Y-%m-%d')}  ${t.price:,.0f}")
print(f"大底: {len(major_troughs)}")
for t in major_troughs:
    print(f"  {timestamps[t.index].strftime('%Y-%m-%d')}  ${t.price:,.0f}")

print(f"\n小顶: {len(minor_peaks)} | 小底: {len(minor_troughs)}")
print("完成!")
