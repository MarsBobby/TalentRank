#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 7: SHAP 解释性 Agent
────────────────────────────
基于博弈论 Shapley 值，为每个候选人的排名提供：
1. 局部解释：为何该候选人排名第 X？
2. 全局解释：驱动该岗位排名的核心因子
3. HR 可读报告（中文）
"""

import os
import sys
import numpy as np
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.phase0_data_assets import CandidateFeatureVector
from modules.phase5_lambdamart import FEATURE_ORDER

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# 特征中文名映射
FEATURE_CN = {
    "education_score":         "教育背景得分",
    "school_ranking":          "学校排名（越小越好）",
    "major_match_score":       "专业匹配度",
    "s_level":                 "职级（语义校准）",
    "csi_score":               "公司平台指数",
    "total_work_years":        "总工龄（年）",
    "career_growth_slope":     "职业成长斜率",
    "job_hopping_premium":     "跳槽溢价",
    "span_of_control":         "管理人数（Span of Control）",
    "span_missing":            "管理人数MNAR标记",
    "management_verb_density": "管理动词密度",
    "reporting_layer":         "汇报层级",
    "kpi_log_score":           "KPI对数得分",
    "kpi_missing":             "KPI MNAR标记",
    "achievement_signal":      "成就量化信号（0/1）",
    "major_projects_count":    "战略项目数",
    "stability_index":         "任职稳定性",
    "soft_skill_score":        "软素质综合得分",
}


class SHAPExplainabilityAgent:
    """
    SHAP 解释性 Agent
    支持两种模式：
      A. LightGBM 模型可用 → 真正的 SHAP TreeExplainer
      B. 无模型 → 基于权重的线性近似 SHAP
    """

    def __init__(self, model=None, weights: Dict[str, float] = None):
        """
        model: LightGBM Booster（Phase5 输出）
        weights: Phase4 权重字典（用于近似 SHAP）
        """
        self.model   = model
        self.weights = weights or {}
        self.explainer = None

        if SHAP_AVAILABLE and model is not None:
            try:
                self.explainer = shap.TreeExplainer(model)
            except Exception:
                pass

    def explain_candidate(self, vec: CandidateFeatureVector,
                          all_vecs: List[CandidateFeatureVector]) -> Dict:
        """
        局部解释：返回该候选人各特征的 Shapley 贡献值
        """
        feat_dict = vec.to_feature_dict()
        feat_vals = np.array([feat_dict.get(f, 0.0) or 0.0 for f in FEATURE_ORDER])

        # ── 模式 A: 真实 SHAP ──
        if self.explainer is not None:
            X = feat_vals.reshape(1, -1)
            shap_vals = self.explainer.shap_values(X)[0]
        # ── 模式 B: 线性近似 SHAP ──
        else:
            shap_vals = self._linear_shap(feat_vals, all_vecs)

        contributions = {}
        for i, feat in enumerate(FEATURE_ORDER):
            contributions[feat] = round(float(shap_vals[i]), 4)

        return {
            "candidate_id": vec.candidate_id,
            "feature_contributions": contributions,
            "top_positive": self._top_features(contributions, n=3, positive=True),
            "top_negative": self._top_features(contributions, n=2, positive=False),
        }

    def explain_global(self, vecs: List[CandidateFeatureVector]) -> Dict:
        """
        全局解释：聚合所有候选人的 SHAP 值，识别驱动排名的核心因子
        """
        all_contribs = []
        for v in vecs:
            exp = self.explain_candidate(v, vecs)
            all_contribs.append(exp["feature_contributions"])

        # 平均绝对 SHAP
        global_importance = {}
        for feat in FEATURE_ORDER:
            vals = [abs(c.get(feat, 0.0)) for c in all_contribs]
            global_importance[feat] = round(float(np.mean(vals)), 4)

        return {
            "global_feature_importance": global_importance,
            "top_drivers": sorted(
                global_importance.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }

    def generate_hr_report(self, vec: CandidateFeatureVector,
                            rank: int,
                            all_vecs: List[CandidateFeatureVector]) -> str:
        """
        生成面向 HR 的中文解释报告
        """
        exp = self.explain_candidate(vec, all_vecs)
        feat_dict = vec.to_feature_dict()
        lines = [
            f"  ┌─ SHAP 解释报告 · {vec.candidate_id} (排名 #{rank}) ─",
        ]

        # 正向贡献（加分项）
        pos = exp["top_positive"]
        if pos:
            lines.append("  │  ✅ 排名加分项：")
            for feat, val in pos:
                cn = FEATURE_CN.get(feat, feat)
                raw = feat_dict.get(feat)
                raw_str = f"{raw:.2f}" if isinstance(raw, float) and raw == raw else "N/A"
                lines.append(f"  │    · {cn}：贡献 +{val:.3f}  (实际值: {raw_str})")

        # 负向贡献（扣分项）
        neg = exp["top_negative"]
        if neg:
            lines.append("  │  ⚠ 排名扣分项：")
            for feat, val in neg:
                cn = FEATURE_CN.get(feat, feat)
                lines.append(f"  │    · {cn}：贡献 {val:.3f}")

        # 置信度提示
        if vec.needs_human_review:
            lines.append("  │  🔴 信息残缺警示：综合置信度低，建议补全简历后重评")

        lines.append(f"  └─ 综合置信度: {vec.global_confidence:.2f}")
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────
    # 私有工具
    # ─────────────────────────────────────────────────────────

    def _linear_shap(self, feat_vals: np.ndarray,
                     all_vecs: List[CandidateFeatureVector]) -> np.ndarray:
        """
        线性近似 SHAP：
        φ_j = w_j * (x_j - E[x_j]) / std(x_j)
        """
        if not all_vecs:
            return np.zeros(len(FEATURE_ORDER))

        # 构建基线分布
        matrix = []
        for v in all_vecs:
            d = v.to_feature_dict()
            row = [d.get(f, 0.0) or 0.0 for f in FEATURE_ORDER]
            matrix.append(row)
        matrix = np.array(matrix, dtype=float)

        means = np.nanmean(matrix, axis=0)
        stds  = np.nanstd(matrix, axis=0)
        stds[stds == 0] = 1

        # 权重向量
        w_arr = np.array([self.weights.get(f, 1.0 / len(FEATURE_ORDER))
                          for f in FEATURE_ORDER])

        shap_vals = w_arr * (feat_vals - means) / stds
        return shap_vals

    def _top_features(self, contributions: Dict[str, float],
                      n: int = 3, positive: bool = True) -> List[Tuple[str, float]]:
        if positive:
            filtered = [(k, v) for k, v in contributions.items() if v > 0]
        else:
            filtered = [(k, v) for k, v in contributions.items() if v < 0]
        return sorted(filtered, key=lambda x: abs(x[1]), reverse=True)[:n]


def print_global_explanation(global_exp: Dict):
    """打印全局重要性报告"""
    print("\n  Phase 7 · SHAP 全局特征重要性（驱动排名的核心因子）")
    print("  " + "-" * 55)
    for feat, imp in global_exp["top_drivers"]:
        cn = FEATURE_CN.get(feat, feat)
        if imp != imp:  # NaN check
            continue
        bar = "█" * int(imp * 50)
        print(f"  {cn:<22} {imp:.4f}  {bar}")
    print()
