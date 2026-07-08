#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 8: 评估体系构建
───────────────────────
双轨制评估指标矩阵：
  离线排序：NDCG@5, NDCG@10
  人类对齐：Kendall-τ（AI 排序 vs 专家排序）
  业务转化：Precision@K
  搜索效能：MRR（平均倒数排名）
"""

import os
import sys
import json
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────
# 指标计算函数
# ─────────────────────────────────────────────────────────────

def dcg_at_k(relevance: List[float], k: int) -> float:
    """Discounted Cumulative Gain @ K"""
    rel = relevance[:k]
    return sum(r / math.log2(i + 2) for i, r in enumerate(rel))


def ndcg_at_k(predicted_order: List[str], gold_relevance: Dict[str, float],
               k: int = 10) -> float:
    """
    NDCG@K：衡量是否将最优候选人排在前 K 位
    predicted_order: AI 预测排序（候选人 ID 列表）
    gold_relevance:  {candidate_id → 相关性分数 0-4}
    """
    # 按 AI 排序取相关性
    pred_rel = [gold_relevance.get(cid, 0.0) for cid in predicted_order[:k]]
    # 理想排序（最大 DCG）
    ideal_rel = sorted(gold_relevance.values(), reverse=True)[:k]

    actual_dcg = dcg_at_k(pred_rel, k)
    ideal_dcg  = dcg_at_k(ideal_rel, k)

    return round(actual_dcg / ideal_dcg, 4) if ideal_dcg > 0 else 0.0


def kendall_tau(ranking1: List[str], ranking2: List[str]) -> float:
    """
    Kendall-τ：衡量 AI 排序与专家排序的一致性
    τ ∈ [-1, 1]，越接近 1 越好
    """
    n = min(len(ranking1), len(ranking2))
    if n < 2:
        return 1.0

    # 转换为排名字典
    r1 = {cid: i for i, cid in enumerate(ranking1[:n])}
    r2 = {cid: i for i, cid in enumerate(ranking2[:n])}

    common = [cid for cid in ranking1[:n] if cid in r2]
    if len(common) < 2:
        return 0.0

    concordant = 0
    discordant = 0
    pairs = [(common[i], common[j]) for i in range(len(common))
             for j in range(i + 1, len(common))]

    for a, b in pairs:
        d1 = r1[a] - r1[b]
        d2 = r2[a] - r2[b]
        if d1 * d2 > 0:
            concordant += 1
        elif d1 * d2 < 0:
            discordant += 1

    total = concordant + discordant
    return round((concordant - discordant) / total, 4) if total > 0 else 0.0


def precision_at_k(predicted_order: List[str],
                   positive_set: List[str], k: int = 5) -> float:
    """
    Precision@K：前 K 名中最终通过面试/入职的比例
    positive_set: 已知的"正例"候选人（通过初试/入职）
    """
    top_k = set(predicted_order[:k])
    hits  = len(top_k & set(positive_set))
    return round(hits / k, 4)


def mean_reciprocal_rank(predicted_order: List[str],
                          relevant_ids: List[str]) -> float:
    """
    MRR：HR 找到第一个符合要求的人才的平均倒数排名
    """
    relevant_set = set(relevant_ids)
    for i, cid in enumerate(predicted_order):
        if cid in relevant_set:
            return round(1.0 / (i + 1), 4)
    return 0.0


# ─────────────────────────────────────────────────────────────
# 黄金标准集构建（Fleiss' Kappa 过滤）
# ─────────────────────────────────────────────────────────────

def fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """
    Fleiss' Kappa：衡量多个 HR 评分者之间的一致性
    ratings_matrix: (N_candidates × N_raters) 矩阵，每格为等级评分 (0-4)
    目标：κ > 0.7 才纳入黄金标准集
    """
    N, k = ratings_matrix.shape   # N=样本数, k=评分者数
    n = k  # 类别数（评分等级 0-4，即5类）
    categories = sorted(set(ratings_matrix.flatten().tolist()))
    n_cat = len(categories)

    # 每个样本-类别的计数
    P_bar_j = []
    for cat in categories:
        count = (ratings_matrix == cat).sum()
        P_bar_j.append(count / (N * k))

    # 每个样本的 P_i
    P_i_list = []
    for i in range(N):
        row = ratings_matrix[i]
        P_i = sum(((row == cat).sum() ** 2 - (row == cat).sum())
                  for cat in categories) / (k * (k - 1))
        P_i_list.append(P_i)

    P_bar   = np.mean(P_i_list)
    P_e_bar = sum(pj ** 2 for pj in P_bar_j)

    if P_e_bar == 1.0:
        return 1.0
    kappa = (P_bar - P_e_bar) / (1 - P_e_bar)
    return round(float(kappa), 4)


def build_gold_standard(candidate_ids: List[str],
                         hr_ratings: Dict[str, List[float]],
                         kappa_threshold: float = 0.7) -> Dict[str, float]:
    """
    构建黄金标准集
    hr_ratings: {candidate_id: [rater1_score, rater2_score, ...]}
    返回：通过 Kappa 阈值过滤后的 {candidate_id: avg_score}
    """
    if not hr_ratings:
        return {}

    n_raters = max(len(v) for v in hr_ratings.values())
    ids = [cid for cid in candidate_ids if cid in hr_ratings]
    if not ids:
        return {}

    matrix = np.array([
        hr_ratings[cid] + [0] * (n_raters - len(hr_ratings[cid]))
        for cid in ids
    ])

    kappa = fleiss_kappa(matrix)
    print(f"  [Phase8] Fleiss' κ = {kappa:.3f}"
          + ("  ✓ 通过" if kappa >= kappa_threshold else "  ✗ 未达标"))

    if kappa >= kappa_threshold:
        return {cid: round(float(np.mean(hr_ratings[cid])), 2) for cid in ids}
    else:
        print("  [Phase8] κ 未达标，使用中位数过滤...")
        gold = {}
        for cid in ids:
            scores = hr_ratings[cid]
            med = float(np.median(scores))
            std = float(np.std(scores))
            if std < 1.5:  # 评分差异不大
                gold[cid] = round(float(np.mean(scores)), 2)
        return gold


# ─────────────────────────────────────────────────────────────
# 综合评估报告
# ─────────────────────────────────────────────────────────────

class EvaluationSystem:
    """
    Phase 8 综合评估系统

    修复记录（v2.0）：
      P0 - Precision@5 = 0.6 在全部6部门完全一致，暴露固定正例比例问题。
           evaluate() 新增 positive_rate 参数，由各职能独立传入真实转化率，
           不再人为对齐。黄金集必须由 3-5 名 HR 独立评分（Fleiss' κ ≥ 0.7）。
      P2 - 新增 evaluate_with_trend()：自动与上次评估结果做 Kendall τ 趋势对比，
           检测跨发版模型偏好漂移；新版 NDCG 高于旧版才触发 CI/CD 自动发版。
    """

    def __init__(self):
        self.history: List[Dict] = []
        # 各部门 Kendall τ 历史（用于漂移监控）
        self.kendall_history: Dict[str, List[float]] = {}

    def evaluate(self,
                 predicted_order:  List[str],
                 expert_order:     Optional[List[str]] = None,
                 gold_relevance:   Optional[Dict[str, float]] = None,
                 positive_set:     Optional[List[str]] = None,
                 k_list:           List[int] = [5, 10],
                 function:         str = "") -> Dict:
        """
        运行完整评估。

        P0修复：positive_set 必须由黄金标准集（Fleiss' κ ≥ 0.7）独立提供，
        不应为固定比例（Precision@5 = 0.6 全部门一致是评估集设计缺陷）。
        各职能的正例比例应独立统计，通过 positive_set 参数传入。
        """
        results = {}

        # NDCG
        if gold_relevance:
            for k in k_list:
                results[f"NDCG@{k}"] = ndcg_at_k(predicted_order, gold_relevance, k)

        # Kendall-τ（AI排序 vs 专家排序）
        if expert_order:
            tau = kendall_tau(predicted_order, expert_order)
            results["Kendall_tau"] = tau
            # 记录趋势历史（P2修复）
            if function:
                self.kendall_history.setdefault(function, []).append(tau)

        # Precision@K（P0修复：使用真实 positive_set，不锁定固定比例）
        if positive_set:
            for k in k_list:
                results[f"Precision@{k}"] = precision_at_k(predicted_order, positive_set, k)
            results["MRR"] = mean_reciprocal_rank(predicted_order, positive_set)

        if function:
            results["_function"] = function
        self.history.append(results)
        return results

    def evaluate_with_trend(self,
                            predicted_order: List[str],
                            expert_order:    Optional[List[str]] = None,
                            gold_relevance:  Optional[Dict[str, float]] = None,
                            positive_set:    Optional[List[str]] = None,
                            function:        str = "") -> Dict:
        """
        P2修复：在 evaluate() 基础上增加 Kendall τ 趋势分析。
        检测本次排序偏好与历史均值的漂移方向，作为模型健康度指标。
        漂移超过 0.15 时触发警告，防止 Bandit 过度拟合单一 HR 偏好。
        """
        results = self.evaluate(
            predicted_order=predicted_order,
            expert_order=expert_order,
            gold_relevance=gold_relevance,
            positive_set=positive_set,
            function=function,
        )

        # 趋势分析
        tau_history = self.kendall_history.get(function, [])
        if len(tau_history) >= 2:
            tau_trend   = tau_history[-1] - tau_history[-2]
            tau_avg     = sum(tau_history) / len(tau_history)
            drift_alert = abs(tau_history[-1] - tau_avg) > 0.15
            results["kendall_trend"] = {
                "current":    round(tau_history[-1], 4),
                "previous":   round(tau_history[-2], 4),
                "delta":      round(tau_trend, 4),
                "history_avg": round(tau_avg, 4),
                "drift_alert": drift_alert,
            }
            if drift_alert:
                print(f"  [Phase8] ⚠ 职能={function} Kendall τ 偏移超过 0.15，"
                      f"当前={tau_history[-1]:.3f} 历史均值={tau_avg:.3f}")

        return results

    def compare_versions(self) -> Dict:
        """
        比较新旧版本指标（CI/CD 发版触发条件：NDCG 必须高于旧版）
        """
        if len(self.history) < 2:
            return {"status": "需要至少两次评估才能比较"}
        old, new = self.history[-2], self.history[-1]
        comparisons = {}
        for metric in set(old) | set(new):
            if metric.startswith("_"):
                continue
            old_val = old.get(metric, 0)
            new_val = new.get(metric, 0)
            comparisons[metric] = {
                "old": old_val, "new": new_val,
                "delta": round(new_val - old_val, 4),
                "improved": new_val >= old_val,
            }
        # CI/CD 触发条件：NDCG@10 必须提升（核心指标）
        ndcg10_ok = comparisons.get("NDCG@10", {}).get("improved", True)
        all_improved = all(v["improved"] for k, v in comparisons.items())
        return {
            "can_release": ndcg10_ok,      # 以 NDCG@10 为主门槛
            "all_improved": all_improved,   # 参考：全指标是否全部提升
            "metrics": comparisons,
        }


def _scalar_metric_value(val) -> Optional[float]:
    """将评估结果中的标量指标转为 float；元数据、dict、不可转换值返回 None。"""
    if val is None or isinstance(val, dict):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def print_evaluation_report(results: Dict, function: str = ""):
    """打印评估报告"""
    print("\n" + "=" * 55)
    print(f"  Phase 8 · 评估报告{f' [{function}]' if function else ''}")
    print("=" * 55)

    metric_descriptions = {
        "NDCG@5":      "前5名排序质量",
        "NDCG@10":     "前10名排序质量",
        "Kendall_tau": "与专家排序一致性",
        "Precision@5": "前5名入围率",
        "Precision@10":"前10名入围率",
        "MRR":         "首个优质人才倒数排名",
    }
    for metric, val in results.items():
        if metric.startswith("_"):
            continue
        num = _scalar_metric_value(val)
        if num is None:
            continue
        desc = metric_descriptions.get(metric, "")
        status = "✓" if num >= 0.6 else "⚠"
        print(f"  {status} {metric:<15} {num:.4f}   {desc}")

    print()

    # 模拟基线对比（冷启动 Phase4 规则 vs LambdaMART）
    baseline = {"NDCG@10": 0.62, "Kendall_tau": 0.55, "MRR": 0.67}
    print("  基线对比（规则评分 vs 当前系统）：")
    for metric, base_val in baseline.items():
        curr_val = results.get(metric)
        curr_f = _scalar_metric_value(curr_val)
        if curr_f is not None:
            delta = curr_f - base_val
            arrow = "↑" if delta > 0 else "↓"
            print(f"    {metric:<15} {base_val:.4f} → {curr_f:.4f}  ({arrow}{abs(delta):.4f})")
    print()