#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 5: LambdaMART 排序模型
──────────────────────────────
基于 LightGBM 实现 LambdaMART Learning-to-Rank
核心优势：
  1. 原生稀疏感知分裂（NaN 自动寻优）
  2. 直接优化 NDCG@K
  3. 无需手工填补 MNAR 字段
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.phase0_data_assets import CandidateFeatureVector, ALL_FEATURE_NAMES as FEATURE_NAMES
from config import LAMBDAMART_PARAMS

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("  ⚠ LightGBM 未安装，Phase5 使用规则回退模式")

FEATURE_ORDER = [
    "education_score", "school_ranking", "major_match_score",
    "s_level", "csi_score", "total_work_years",
    "career_growth_slope", "job_hopping_premium",
    "span_of_control", "span_missing",
    "management_verb_density", "reporting_layer",
    "kpi_log_score", "kpi_missing",
    "achievement_signal", "major_projects_count",
    "stability_index", "soft_skill_score",
]


class LambdaMARTRanker:
    """
    LambdaMART 排序引擎
    在训练数据不足（< 10条带标签）时自动回退到 Phase4 评分
    """

    def __init__(self):
        self.model: Optional[object] = None
        self.is_trained = False
        self._fallback_weights: Dict[str, float] = {}

    # ────── 训练 ────────────────────────────────────────────

    def train(self, vecs: List[CandidateFeatureVector],
              labels: List[float], query_ids: List[int]) -> bool:
        """
        训练 LambdaMART 模型
        labels: 相关性分数 (0-4)，由 HR 黄金标准集提供
        query_ids: 每个候选人所属的 query 组（同一岗位 = 同组）
        """
        if not LGB_AVAILABLE:
            print("  [Phase5] LightGBM 不可用，跳过训练")
            return False

        X = self._build_feature_matrix(vecs)
        y = np.array(labels, dtype=float)
        groups = self._build_groups(query_ids)

        if X.shape[0] < 5:
            print(f"  [Phase5] 样本数 {X.shape[0]} 不足，跳过训练")
            return False

        dataset = lgb.Dataset(X, label=y, group=groups,
                              feature_name=FEATURE_ORDER)

        params = dict(LAMBDAMART_PARAMS)
        self.model = lgb.train(
            params,
            dataset,
            num_boost_round=params.pop("n_estimators", 200),
            valid_sets=[dataset],
            callbacks=[lgb.early_stopping(20, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
        self.is_trained = True
        print(f"  [Phase5] LambdaMART 训练完成，特征数: {X.shape[1]}")
        return True

    # ────── 推断 ────────────────────────────────────────────

    def predict_scores(self, vecs: List[CandidateFeatureVector]) -> List[float]:
        """返回每个候选人的预测排序分"""
        if self.is_trained and self.model is not None:
            X = self._build_feature_matrix(vecs)
            scores = self.model.predict(X)
            return list(scores)
        else:
            # 回退：使用加权评分（来自 Phase4）
            return self._fallback_score(vecs)

    def rank(self, vecs: List[CandidateFeatureVector]) -> List[Tuple[str, float, int]]:
        """
        返回排序列表：[(candidate_id, score, rank), ...]
        """
        scores = self.predict_scores(vecs)
        indexed = [(v.candidate_id, s) for v, s in zip(vecs, scores)]
        ranked = sorted(indexed, key=lambda x: x[1], reverse=True)
        return [(cid, score, rank + 1) for rank, (cid, score) in enumerate(ranked)]

    # ────── 稀疏感知说明 ────────────────────────────────────

    def explain_sparse_splits(self, feature_importance: bool = True):
        """
        打印 LightGBM 稀疏感知特征重要性
        展示 span_missing / kpi_missing 对排序的贡献
        """
        if not self.is_trained or self.model is None:
            print("  [Phase5] 模型未训练，无法打印特征重要性")
            return

        importance = self.model.feature_importance(importance_type="gain")
        feat_imp = dict(zip(FEATURE_ORDER, importance))
        sorted_imp = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)

        print("\n  Phase 5 · LambdaMART 特征重要性（稀疏感知分裂）")
        print("  " + "-" * 50)
        for feat, imp in sorted_imp[:12]:
            bar = "█" * int(imp / max(importance) * 20)
            mnar_tag = " ← MNAR标记" if "missing" in feat else ""
            print(f"  {feat:<28} {imp:>8.1f}  {bar}{mnar_tag}")

    # ────── 私有方法 ─────────────────────────────────────────

    def _build_feature_matrix(self, vecs: List[CandidateFeatureVector]) -> np.ndarray:
        """构建特征矩阵，NaN 保留（LightGBM 原生处理）"""
        rows = []
        for v in vecs:
            d = v.to_feature_dict()
            row = [d.get(f, float("nan")) for f in FEATURE_ORDER]
            rows.append(row)
        return np.array(rows, dtype=float)

    def _build_groups(self, query_ids: List[int]) -> List[int]:
        """构建 LightGBM group 数组（每组的样本数）"""
        from collections import Counter
        cnt = Counter(query_ids)
        # 保持顺序
        seen, groups = set(), []
        for qid in query_ids:
            if qid not in seen:
                groups.append(cnt[qid])
                seen.add(qid)
        return groups

    def _fallback_score(self, vecs: List[CandidateFeatureVector]) -> List[float]:
        """Phase5 未训练时的回退评分（基于规则权重）"""
        weights = {
            "s_level":               0.20,
            "kpi_log_score":         0.18,
            "achievement_signal":    0.12,
            "management_verb_density": 0.10,
            "career_growth_slope":   0.10,
            "education_score":       0.08,
            "csi_score":             0.08,
            "stability_index":       0.07,
            "soft_skill_score":      0.07,
        }
        scores = []
        for v in vecs:
            d = v.to_feature_dict()
            score = 0.0
            for feat, w in weights.items():
                val = d.get(feat)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    # MNAR 保守估计：中等水平
                    val = 0.4
                score += float(val) * w
            # 置信度折扣
            if v.global_confidence < 0.6:
                score *= 0.9
            scores.append(round(score, 4))
        return scores

    def save_model(self, path: str):
        """保存模型到文件"""
        if self.is_trained and self.model:
            self.model.save_model(path)
            print(f"  [Phase5] 模型已保存: {path}")

    def load_model(self, path: str) -> bool:
        """加载已保存的模型"""
        if not LGB_AVAILABLE:
            return False
        if os.path.exists(path):
            self.model = lgb.Booster(model_file=path)
            self.is_trained = True
            print(f"  [Phase5] 模型已加载: {path}")
            return True
        return False


def print_ranking_result(ranked: List[Tuple[str, float, int]]):
    """打印排序结果"""
    print("\n" + "=" * 55)
    print("  Phase 5 · LambdaMART 排序结果")
    print("=" * 55)
    print(f"  {'排名':<5} {'候选人ID':<20} {'排序分'}")
    print("  " + "-" * 40)
    for cid, score, rank in ranked:
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "  ")
        print(f"  {medal} #{rank:<4} {cid:<20} {score:.4f}")
    print()


# 允许 phase0 导入时的兼容性
try:
    from modules.phase0_data_assets import ALL_FEATURE_NAMES as FEATURE_NAMES
except:
    FEATURE_NAMES = FEATURE_ORDER
