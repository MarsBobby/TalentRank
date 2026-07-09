#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4: 动态混合赋权机制（AHP + EWM）
────────────────────────────────────────
1. AHP 主观赋权（专家判断矩阵 + CR 一致性校验）
2. EWM 客观修正（熵权法，基于候选人池信息熵）
3. 动态触发策略（N<5:纯AHP; 5≤N<20:线性过渡; N≥20:标准混合）
4. 输出候选人综合评分
"""

import os
import sys
import math
import numpy as np
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.phase0_data_assets import CandidateFeatureVector
from config import (AHP_WEIGHTS_BRANDING, AHP_WEIGHTS_PRODUCTION,
                    AHP_PURE_THRESHOLD, AHP_EWM_THRESHOLD,
                    AHP_WEIGHT, EWM_WEIGHT)

# ─────────────────────────────────────────────────────────────
# 特征名顺序（与 AHP 权重字典对齐）
# ─────────────────────────────────────────────────────────────

FEATURE_ORDER = [
    "education_score",
    "s_level",
    "career_growth_slope",
    "span_of_control",
    "kpi_log_score",
    "stability_index",
    "achievement_signal",
    "soft_skill_score",
]

# 各职能 AHP 权重（可扩展）
AHP_PRESETS: Dict[str, Dict[str, float]] = {
    "品牌":   AHP_WEIGHTS_BRANDING,
    "电商":   AHP_WEIGHTS_BRANDING,
    "人力资源": {
        "education_score":      0.12,
        "s_level":              0.15,
        "career_growth_slope":  0.13,
        "span_of_control":      0.12,
        "kpi_log_score":        0.16,
        "stability_index":      0.15,
        "achievement_signal":   0.10,
        "soft_skill_score":     0.07,
    },
    "生产":   AHP_WEIGHTS_PRODUCTION,
    "研发": {
        "education_score":      0.15,
        "s_level":              0.15,
        "career_growth_slope":  0.12,
        "span_of_control":      0.10,
        "kpi_log_score":        0.20,
        "stability_index":      0.12,
        "achievement_signal":   0.08,
        "soft_skill_score":     0.08,
    },
    "销售": {
        "education_score":      0.07,
        "s_level":              0.13,
        "career_growth_slope":  0.12,
        "span_of_control":      0.12,
        "kpi_log_score":        0.28,
        "stability_index":      0.10,
        "achievement_signal":   0.12,
        "soft_skill_score":     0.06,
    },
}


# ─────────────────────────────────────────────────────────────
# AHP 模块
# ─────────────────────────────────────────────────────────────

class AHPWeighter:
    """AHP 主观赋权器"""

    # 随机一致性指标（用于 CR 校验，n=1..10）
    RI = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]

    def get_weights(self, function: str) -> Dict[str, float]:
        """获取指定职能的 AHP 权重"""
        weights = AHP_PRESETS.get(function, AHP_WEIGHTS_BRANDING)
        # 归一化（确保总和=1）
        total = sum(weights.values())
        return {k: round(v / total, 6) for k, v in weights.items()}

    def check_consistency(self, judgment_matrix: np.ndarray) -> Tuple[float, bool]:
        """
        计算判断矩阵的一致性比率 CR
        CR < 0.1 则通过
        """
        n = judgment_matrix.shape[0]
        if n < 2:
            return 0.0, True
        # 计算最大特征值
        col_sum = judgment_matrix.sum(axis=0)
        normalized = judgment_matrix / col_sum
        weights = normalized.mean(axis=1)
        lambda_max = (judgment_matrix @ weights / weights).mean()
        CI = (lambda_max - n) / (n - 1)
        ri = self.RI[n - 1] if n <= len(self.RI) else 1.49
        CR = CI / ri if ri > 0 else 0.0
        return round(CR, 4), CR < 0.1


# ─────────────────────────────────────────────────────────────
# EWM 模块（熵权法）
# ─────────────────────────────────────────────────────────────

class EWMWeighter:
    """EWM 客观赋权器（熵权法）"""

    def compute_weights(self, feature_matrix: np.ndarray,
                        feature_names: List[str]) -> Dict[str, float]:
        """
        输入：候选人特征矩阵 (N × M)，其中 N=候选人数，M=特征数
        输出：各特征的熵权
        """
        N, M = feature_matrix.shape
        if N < 2:
            # 样本过少，无法计算熵，均等权重
            return {f: 1.0 / M for f in feature_names}

        # 处理 NaN：用列均值填充（EWM 阶段仅用于权重计算）
        col_means = np.nanmean(feature_matrix, axis=0)
        for j in range(M):
            mask = np.isnan(feature_matrix[:, j])
            feature_matrix[mask, j] = col_means[j]

        # 归一化（Min-Max）
        col_min = feature_matrix.min(axis=0)
        col_max = feature_matrix.max(axis=0)
        denom = col_max - col_min
        denom[denom == 0] = 1  # 防止除零
        P = (feature_matrix - col_min) / denom
        # 防止 log(0)
        P = np.clip(P, 1e-9, 1.0)
        # 行归一化
        row_sum = P.sum(axis=0, keepdims=True)
        row_sum[row_sum == 0] = 1
        P = P / row_sum

        # 信息熵
        E = -(P * np.log(P)).sum(axis=0) / math.log(N)
        E = np.clip(E, 0, 1)

        # 效用度
        D = 1 - E
        D_sum = D.sum()
        if D_sum == 0:
            weights_arr = np.ones(M) / M
        else:
            weights_arr = D / D_sum

        return {feature_names[j]: round(float(weights_arr[j]), 6) for j in range(M)}


# ─────────────────────────────────────────────────────────────
# 动态混合赋权主类
# ─────────────────────────────────────────────────────────────

class DynamicWeighter:
    """
    动态混合赋权：根据候选人数量 N 自动切换策略
    N < 5   → 纯 AHP
    5≤N<20  → 线性过渡
    N ≥ 20  → 标准混合（0.5 AHP + 0.5 EWM）
    """

    def __init__(self):
        self.ahp = AHPWeighter()
        self.ewm = EWMWeighter()

    def compute_weights(self, vecs: List[CandidateFeatureVector],
                        function: str = "品牌") -> Dict[str, float]:
        N = len(vecs)
        ahp_w = self.ahp.get_weights(function)

        if N < AHP_PURE_THRESHOLD:
            print(f"  [Phase4] N={N} < {AHP_PURE_THRESHOLD} → 纯 AHP 权重")
            return ahp_w

        # 构建特征矩阵
        matrix = self._build_matrix(vecs)

        # 线性混合系数 alpha
        if N < AHP_EWM_THRESHOLD:
            alpha = (N - AHP_PURE_THRESHOLD) / (AHP_EWM_THRESHOLD - AHP_PURE_THRESHOLD)
            print(f"  [Phase4] N={N} → 线性过渡 α={alpha:.2f}")
        else:
            alpha = 1.0
            print(f"  [Phase4] N={N} ≥ {AHP_EWM_THRESHOLD} → 标准混合 (α=1.0)")

        ewm_w = self.ewm.compute_weights(matrix.copy(), FEATURE_ORDER)

        # 混合
        final_w = {}
        for feat in FEATURE_ORDER:
            w_ahp = ahp_w.get(feat, 0.0)
            w_ewm = ewm_w.get(feat, 0.0)
            blended = (1 - alpha * (1 - AHP_WEIGHT)) * w_ahp + alpha * EWM_WEIGHT * w_ewm
            final_w[feat] = blended

        # 归一化
        total = sum(final_w.values())
        if total > 0:
            final_w = {k: round(v / total, 6) for k, v in final_w.items()}

        return final_w

    def _build_matrix(self, vecs: List[CandidateFeatureVector]) -> np.ndarray:
        """构建 N×M 特征矩阵，MNAR NaN 保留"""
        rows = []
        for v in vecs:
            d = v.to_feature_dict()
            row = [d.get(f, float("nan")) for f in FEATURE_ORDER]
            rows.append(row)
        return np.array(rows, dtype=float)

    def score_candidates(self, vecs: List[CandidateFeatureVector],
                         weights: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        使用混合权重计算候选人综合得分
        MNAR 缺失字段：用有数据候选人的中位数作为保守估计
        """
        matrix = self._build_matrix(vecs)
        M = len(FEATURE_ORDER)
        N = len(vecs)

        # 计算列中位数（用于 MNAR 评分时的保守替代）
        col_medians = np.nanmedian(matrix, axis=0)

        # Min-Max 归一化（用于评分，与 EWM 共享逻辑）
        col_min = np.nanmin(matrix, axis=0)
        col_max = np.nanmax(matrix, axis=0)
        denom = col_max - col_min
        denom[denom == 0] = 1

        scores = []
        for i, vec in enumerate(vecs):
            row = matrix[i].copy()
            for j in range(M):
                if np.isnan(row[j]):
                    row[j] = col_medians[j]  # MNAR 保守中位数替代

            # 归一化
            row_norm = (row - col_min) / denom
            row_norm = np.clip(row_norm, 0, 1)

            # 加权求和
            w_arr = np.array([weights.get(f, 0.0) for f in FEATURE_ORDER])
            score = float(np.dot(row_norm, w_arr))

            # 置信度惩罚：低置信度候选人得分打折
            conf_penalty = 0.95 if vec.global_confidence >= 0.6 else 0.85
            final_score = round(score * conf_penalty, 4)
            scores.append((vec.candidate_id, final_score))

        return sorted(scores, key=lambda x: x[1], reverse=True)


def print_weight_report(weights: Dict[str, float], function: str):
    """打印权重分配报告"""
    print("\n" + "=" * 55)
    print(f"  Phase 4 · 动态权重报告 [{function}岗位]")
    print("=" * 55)
    sorted_w = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    for feat, w in sorted_w:
        bar = "█" * int(w * 40)
        print(f"  {feat:<26} {w:.4f}  {bar}")
    print()

# ─────────────────────────────────────────────────────────────
# AHP 可视化（论文级展示）
# ─────────────────────────────────────────────────────────────
"""
import matplotlib.pyplot as plt
import seaborn as sns


def build_ahp_matrix_from_weights(weights: Dict[str, float]) -> Tuple[np.ndarray, List[str]]:
    
    # 根据 AHP 权重反推判断矩阵（用于可视化）
    
    features = list(weights.keys())
    w = np.array([weights[f] for f in features])

    n = len(w)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            matrix[i, j] = w[i] / w[j]

    return matrix, features
"""

#############temporary for visualization testing

"""
def visualize_ahp_convergence(matrix, feature_names, title="AHP 主观矩阵与一致性审计"):
    
    # Phase4 可视化核心函数（论文级图）
    
    n = matrix.shape[0]

    eig_vals, eig_vecs = np.linalg.eig(matrix)
    max_idx = np.argmax(eig_vals.real)
    lambda_max = eig_vals[max_idx].real

    w_ahp = eig_vecs[:, max_idx].real
    w_ahp = w_ahp / np.sum(w_ahp)

    RI_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.89, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41}
    ri = RI_dict.get(n, 1.41)
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0
    cr = ci / ri if ri > 0 else 0

    fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                             gridspec_kw={'width_ratios': [1.2, 1]})

    # 左：判断矩阵
    sns.heatmap(
        np.log2(matrix),
        annot=matrix,
        fmt=".2f",
        cmap="RdYlBu_r",
        cbar=False,
        xticklabels=feature_names,
        yticklabels=feature_names,
        ax=axes[0],
        annot_kws={"size": 9}
    )
    axes[0].set_title(f"{title} - 判断矩阵 A", fontsize=13)

    # 右：权重
    y_pos = np.arange(len(feature_names))
    bars = axes[1].barh(y_pos, w_ahp)

    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(feature_names)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("权重")

    for bar in bars:
        width = bar.get_width()
        axes[1].text(width, bar.get_y() + bar.get_height()/2,
                     f'{width:.3f}', va='center')

    # 一致性标签
    status_color = "green" if cr < 0.1 else "red"
    status_text = f"CR={cr:.4f} {'通过' if cr < 0.1 else '失败'}"

    axes[1].text(0.5, 1.1, status_text,
                 transform=axes[1].transAxes,
                 ha="center",
                 color="white",
                 bbox=dict(facecolor=status_color, boxstyle="round"))

    axes[1].set_title(f"λmax={lambda_max:.3f} | CI={ci:.3f}")

    plt.tight_layout()
    return fig
"""
