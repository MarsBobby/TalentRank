import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os

# ═══════════════════════════════════════════════════
# 直接使用内嵌数据（避免路径问题）
# ═══════════════════════════════════════════════════
data_all = {
    "电商": [
        {"feature": "s_level", "importance": 0.1283},
        {"feature": "career_growth_slope", "importance": 0.1196},
        {"feature": "education_score", "importance": 0.0933},
        {"feature": "span_of_control", "importance": float('nan')},
        {"feature": "kpi_log_score", "importance": float('nan')}
    ],
    "人力资源": [
        {"feature": "s_level", "importance": 0.1361},
        {"feature": "career_growth_slope", "importance": 0.1271},
        {"feature": "span_of_control", "importance": float('nan')},
        {"feature": "stability_index", "importance": 0.1360},
        {"feature": "kpi_log_score", "importance": 0.1243}
    ],
    "销售": [
        {"feature": "career_growth_slope", "importance": 0.1070},
        {"feature": "span_of_control", "importance": float('nan')},
        {"feature": "kpi_log_score", "importance": 0.2486},
        {"feature": "s_level", "importance": 0.0822},
        {"feature": "stability_index", "importance": 0.0789}
    ],
    "生产": [
        {"feature": "s_level", "importance": 0.1659},
        {"feature": "span_of_control", "importance": float('nan')},
        {"feature": "kpi_log_score", "importance": 0.1816},
        {"feature": "stability_index", "importance": 0.1229},
        {"feature": "career_growth_slope", "importance": 0.0916}
    ],
    "研发": [
        {"feature": "career_growth_slope", "importance": 0.1039},
        {"feature": "span_of_control", "importance": float('nan')},
        {"feature": "kpi_log_score", "importance": 0.1703},
        {"feature": "s_level", "importance": 0.1258},
        {"feature": "stability_index", "importance": 0.1039}
    ],
    "品牌": [
        {"feature": "career_growth_slope", "importance": 0.1183},
        {"feature": "span_of_control", "importance": float('nan')},
        {"feature": "kpi_log_score", "importance": 0.1644},
        {"feature": "s_level", "importance": 0.1175},
        {"feature": "education_score", "importance": 0.0985}
    ]
}

# ═══════════════════════════════════════════════════
# 数据准备
# ═══════════════════════════════════════════════════
feature_order = [
    "s_level", "career_growth_slope", "kpi_log_score",
    "stability_index", "education_score", "span_of_control"
]
feature_labels = [
    "管理层级\n(s_level)", "职业成长斜率\n(career_growth_slope)", "KPI对数分\n(kpi_log_score)",
    "稳定性指数\n(stability_index)", "学历评分\n(education_score)", "管理幅度\n(span_of_control)"
]

func_names = list(data_all.keys())
n_funcs = len(func_names)
n_feats = len(feature_order)

# 构建矩阵并验证
matrix = np.full((n_feats, n_funcs), np.nan)
for j, fname in enumerate(func_names):
    feat_map = {d["feature"]: d["importance"] for d in data_all[fname]}
    for i, feat in enumerate(feature_order):
        val = feat_map.get(feat, np.nan)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            matrix[i, j] = val

print(f"矩阵形状: {matrix.shape}")
print(f"有效值数量: {np.sum(~np.isnan(matrix))}")
print("部分矩阵值:")
print(matrix[:3, :3])

# ═══════════════════════════════════════════════════
# 热力图 - 修复版本
# ═══════════════════════════════════════════════════
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(14, 7))
cmap = LinearSegmentedColormap.from_list("custom_blue", ["#E8F1FA", "#4A90D9", "#1B4F8A", "#0A2540"])

# 使用 vmin=0, vmax=0.3 确保数据范围
im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, aspect='auto', vmin=0, vmax=0.3)

# 添加数值标签
for i in range(n_feats):
    for j in range(n_funcs):
        val = matrix[i, j]
        if np.isnan(val):
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                      facecolor='#F0F0F0', edgecolor='#CCCCCC', 
                                      linewidth=0.8, hatch='///'))
            ax.text(j, i, '— 缺失 —', ha='center', va='center', 
                    fontsize=9, color='#999999', fontstyle='italic')
        else:
            text_color = 'white' if val > 0.15 else '#1B4F8A'
            ax.text(j, i, f'{val:.4f}', ha='center', va='center', 
                    fontsize=11, fontweight='bold', color=text_color)

ax.set_xticks(range(n_funcs))
ax.set_xticklabels(func_names, fontsize=13, fontweight='bold')
ax.set_yticks(range(n_feats))
ax.set_yticklabels(feature_labels, fontsize=11)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

ax.set_title('各业务函数 × 核心特征维度 — SHAP 特征重要性（EWM权重代理）热力图\n'
             'Feature Importance (SHAP) as Proxy for EWM Weights Across Functions',
             fontsize=15, fontweight='bold', pad=25)

cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('SHAP 重要性 (Importance)', fontsize=11, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

legend_elements = [
    mpatches.Patch(facecolor='#4A90D9', edgecolor='#1B4F8A', label='有效特征权重'),
    mpatches.Patch(facecolor='#F0F0F0', edgecolor='#CCCCCC', hatch='///', label='数据缺失 (NaN)'),
]
ax.legend(handles=legend_elements, loc='upper center', 
          bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=11, frameon=True, 
          fancybox=True, shadow=True)

for i in range(n_feats + 1):
    ax.axhline(y=i-0.5, color='white', linewidth=1.5)
for j in range(n_funcs + 1):
    ax.axvline(x=j-0.5, color='white', linewidth=1.5)

plt.tight_layout()
plt.savefig('ewm_weights_heatmap_fixed.png', dpi=200, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()
print("✅ 热力图（修复版）已保存")

# ═══════════════════════════════════════════════════
# 柱状图 - 修复版本
# ═══════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 7.5))

valid_features = [f for f in feature_order if not np.all(np.isnan(matrix[feature_order.index(f), :]))]
valid_indices = [feature_order.index(f) for f in valid_features]
valid_labels = [feature_labels[i].replace('\n', ' ') for i in valid_indices]

print(f"有效特征: {valid_features}")
print(f"有效特征数量: {len(valid_features)}")

n_valid = len(valid_features)
bar_width = 0.12
index = np.arange(n_valid)

colors = ['#1B4F8A', '#4A90D9', '#7FB3E0', '#E8923C', '#C0504D', '#7CA982']

for j, fname in enumerate(func_names):
    vals = []
    for i in valid_indices:
        v = matrix[i, j]
        if not np.isnan(v) and v > 0:  # 只保留 >0 的值
            vals.append(v)
        else:
            vals.append(0)  # 保留 0 但确保显示
    
    print(f"{fname} 的 vals: {vals}")
    
    bars = ax.bar(index + j * bar_width, vals, bar_width, 
                  label=fname, color=colors[j], alpha=0.85, edgecolor='white', linewidth=0.5)
    
    for k, (bar, v_idx) in enumerate(zip(bars, valid_indices)):
        actual_val = matrix[v_idx, j]
        if np.isnan(actual_val):
            ax.text(bar.get_x() + bar.get_width()/2, 0.005, 'NaN', 
                    ha='center', va='bottom', fontsize=7, color='#999999', fontstyle='italic')
        else:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.003, 
                        f'{actual_val:.3f}', ha='center', va='bottom', 
                        fontsize=7.5, fontweight='bold', color=colors[j])

ax.set_xlabel('核心特征维度', fontsize=13, fontweight='bold')
ax.set_ylabel('SHAP 特征重要性', fontsize=13, fontweight='bold')
ax.set_title('各业务函数核心特征 SHAP 重要性对比（EWM 权重代理）\nSHAP Feature Importance Comparison as EWM Weight Proxy',
             fontsize=15, fontweight='bold', pad=15)
ax.set_xticks(index + bar_width * (n_funcs - 1) / 2)
ax.set_xticklabels(valid_labels, fontsize=10.5)
ax.legend(loc='upper right', fontsize=10, ncol=2, frameon=True, fancybox=True, shadow=True)
ax.set_ylim(0, 0.32)  # 确保显示 0-0.32 范围
ax.yaxis.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

ax.annotate('注：span_of_control（管理幅度）全维度缺失，已排除；\nNaN 标记表示该函数下该特征未产出有效 SHAP 值',
            xy=(0.01, 0.97), xycoords='axes fraction', fontsize=9, 
            color='#666666', va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF8E1', edgecolor='#FFC107', alpha=0.8))

plt.tight_layout()
plt.savefig('ewm_weights_bar_fixed.png', dpi=200, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()
print("✅ 柱状图（修复版）已保存")

# ═══════════════════════════════════════════════════
# 完整性诊断图 - 修复版本
# ═══════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左图：各特征缺失率
ax1 = axes[0]
missing_rates = []
for i, feat in enumerate(feature_order):
    nan_count = np.isnan(matrix[i, :]).sum()
    missing_rates.append(nan_count / n_funcs * 100)
print(f"各特征缺失率: {missing_rates}")

colors_bar = ['#2ECC71' if r == 0 else ('#F39C12' if r < 50 else '#E74C3C') for r in missing_rates]
bars = ax1.barh(range(n_feats), missing_rates, color=colors_bar, edgecolor='white', linewidth=1.2, height=0.6)
ax1.set_yticks(range(n_feats))
ax1.set_yticklabels(feature_order, fontsize=11)
ax1.set_xlim(0, 110)  # 确保显示 0-100%
ax1.set_xlabel('缺失率 (%)', fontsize=12, fontweight='bold')
ax1.set_title('各特征 SHAP 重要性值缺失率', fontsize=14, fontweight='bold', pad=12)
ax1.invert_yaxis()

for i, (bar, rate) in enumerate(zip(bars, missing_rates)):
    label = f'{rate:.0f}%  ({int(np.isnan(matrix[i,:]).sum())}/{n_funcs} 缺失)'
    ax1.text(rate + 2, bar.get_y() + bar.get_height()/2, label,
             va='center', fontsize=10, fontweight='bold',
             color='#E74C3C' if rate > 0 else '#27AE60')
ax1.xaxis.grid(True, alpha=0.2, linestyle='--')
ax1.set_axisbelow(True)

# 右图：各函数数据完整度
ax2 = axes[1]
completeness = []
for j, fname in enumerate(func_names):
    valid_count = np.sum(~np.isnan(matrix[:, j]))
    completeness.append(valid_count / n_feats * 100)
print(f"各函数完整度: {completeness}")

colors_bar2 = ['#27AE60' if c >= 80 else ('#F39C12' if c >= 50 else '#E74C3C') for c in completeness]
bars2 = ax2.barh(range(n_funcs), completeness, color=colors_bar2, edgecolor='white', linewidth=1.2, height=0.6)
ax2.set_yticks(range(n_funcs))
ax2.set_yticklabels(func_names, fontsize=12, fontweight='bold')
ax2.set_xlim(0, 110)  # 确保显示 0-100%
ax2.set_xlabel('数据完整度 (%)', fontsize=12, fontweight='bold')
ax2.set_title('各业务函数 SHAP 驱动因子完整度', fontsize=14, fontweight='bold', pad=12)
ax2.invert_yaxis()

for i, (bar, comp) in enumerate(zip(bars2, completeness)):
    valid = int(np.sum(~np.isnan(matrix[:, i])))
    label = f'{comp:.0f}%  ({valid}/{n_feats} 有效)'
    ax2.text(comp + 2, bar.get_y() + bar.get_height()/2, label,
             va='center', fontsize=10, fontweight='bold',
             color='#27AE60' if comp >= 80 else '#E67E22')
ax2.xaxis.grid(True, alpha=0.2, linestyle='--')
ax2.set_axisbelow(True)

fig.suptitle('EWM 权重 / SHAP 特征重要性 — 数据完整性诊断面板\nData Completeness Diagnostic Panel for EWM Weight Proxy',
             fontsize=15, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('ewm_completeness_fixed.png', dpi=200, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()
print("✅ 完整性诊断图（修复版）已保存")
