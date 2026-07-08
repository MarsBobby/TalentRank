#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 6: Contextual Bandit 在线反馈机制（LinUCB + KL 散度正则化）
────────────────────────────────────────────────────────────────────
- LinUCB：上下文多臂老虎机，解决奖励信号稀疏问题
- KL 散度约束：防止单次异常调整导致模型遗忘全局特征
- 支持 HR 行为反馈信号：点击(0.3) / 收藏(0.6) / 邀约(1.0)
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.phase0_data_assets import CandidateFeatureVector
from config import LINUCB_ALPHA, KL_LAMBDA, MAX_WEIGHT_DELTA, FUNCTION_KL_LAMBDA

# HR 行为 → 奖励信号映射
REWARD_MAP = {
    "click":    0.3,   # 点击查看详情
    "bookmark": 0.6,   # 收藏简历
    "invite":   1.0,   # 发起面试邀约
    "reject":   -0.1,  # 明确淘汰
}

FEATURE_DIM = 18  # 与 Phase5 特征维度对齐


class LinUCBAgent:
    """
    LinUCB 算法实现（Disjoint 版本）
    每个"臂"对应候选人的特征上下文

    P2修复：__init__ 接受 kl_lambda 参数，支持职能差异化约束强度。
    """

    def __init__(self, alpha: float = LINUCB_ALPHA, d: int = FEATURE_DIM,
                 kl_lambda: float = KL_LAMBDA):
        self.alpha     = alpha
        self.d         = d
        self.kl_lambda = kl_lambda   # P2修复：职能差异化 KL 惩罚系数
        # 全局参数矩阵 (d×d) 和向量 (d×1)
        self.A  = np.eye(d)
        self.b  = np.zeros(d)
        self._prior_theta = None
        self.update_count = 0

    def predict_ucb(self, context: np.ndarray) -> float:
        """
        计算候选人的 UCB 得分：θ^T x + α √(x^T A^{-1} x)
        """
        x = context.reshape(-1)
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b
        ucb_bonus = self.alpha * np.sqrt(x @ A_inv @ x)
        return float(theta @ x + ucb_bonus)

    def update(self, context: np.ndarray, reward: float):
        """
        收到 HR 反馈后更新参数（带 KL 散度约束）。
        P2修复：使用实例级 self.kl_lambda（职能差异化），
        不再依赖全局常量 KL_LAMBDA。
        """
        x = context.reshape(-1)
        self.A += np.outer(x, x)
        self.b += reward * x
        self.update_count += 1

        # KL 散度正则化：限制权重漂移幅度
        if self._prior_theta is not None:
            theta_new = np.linalg.inv(self.A) @ self.b
            delta = theta_new - self._prior_theta
            # 逐维裁剪漂移量（使用职能差异化 kl_lambda 缩放上限）
            effective_delta = MAX_WEIGHT_DELTA * (1.0 / max(self.kl_lambda, 0.01))
            delta_clipped = np.clip(delta, -effective_delta, effective_delta)
            theta_new = self._prior_theta + delta_clipped
            self.b = self.A @ theta_new

    def set_prior(self, theta: np.ndarray):
        """设置先验权重（来自 Phase4/5 的初始权重）"""
        self._prior_theta = theta.copy()

    def get_theta(self) -> np.ndarray:
        """获取当前最优权重估计"""
        return np.linalg.inv(self.A) @ self.b

    def compute_kl_divergence(self) -> float:
        """计算当前权重与先验的 KL 散度（近似正态分布下）"""
        if self._prior_theta is None:
            return 0.0
        theta = self.get_theta()
        diff = theta - self._prior_theta
        return float(0.5 * np.sum(diff ** 2))  # 简化：高斯近似


class ContextualBanditSystem:
    """
    完整的 Contextual Bandit 系统
    管理多个岗位的 LinUCB Agent，处理 HR 反馈闭环
    """

    def __init__(self):
        self.agents: Dict[str, LinUCBAgent] = {}   # key = function/岗位
        self.feedback_log: List[Dict] = []

    def get_agent(self, function: str) -> LinUCBAgent:
        """懒加载：按职能创建独立 Agent"""
        if function not in self.agents:
            self.agents[function] = LinUCBAgent()
        return self.agents[function]

    def initialize_from_weights(self, function: str,
                                weights: Dict[str, float],
                                feature_order: List[str]):
        """
        用 Phase4 权重初始化先验 θ，并为该职能设置差异化 KL lambda。
        P2修复：销售岗 kpi_log_score 重要性已达 0.25，需收紧约束防止漂移。
        """
        # 按职能获取差异化 KL lambda
        kl_lambda = FUNCTION_KL_LAMBDA.get(function, FUNCTION_KL_LAMBDA["default"])
        # 创建/更新 Agent（携带职能专属 lambda）
        if function not in self.agents:
            self.agents[function] = LinUCBAgent(kl_lambda=kl_lambda)
        else:
            self.agents[function].kl_lambda = kl_lambda

        agent = self.agents[function]
        theta = np.array([weights.get(f, 0.0) for f in feature_order])
        if len(theta) < FEATURE_DIM:
            theta = np.concatenate([theta, np.zeros(FEATURE_DIM - len(theta))])
        agent.set_prior(theta[:FEATURE_DIM])
        print(f"  [Phase6] 职能={function}  KL_λ={kl_lambda}  先验权重已初始化")

    def rank_with_ucb(self, vecs: List[CandidateFeatureVector],
                      function: str) -> List[Tuple[str, float, int]]:
        """
        使用 UCB 分数排序候选人（在线推断）
        """
        agent = self.get_agent(function)
        results = []
        for vec in vecs:
            ctx = self._build_context(vec)
            ucb_score = agent.predict_ucb(ctx)
            results.append((vec.candidate_id, ucb_score))

        ranked = sorted(results, key=lambda x: x[1], reverse=True)
        return [(cid, score, rank + 1) for rank, (cid, score) in enumerate(ranked)]

    def receive_feedback(self, candidate_id: str, action: str,
                         vec: CandidateFeatureVector, function: str):
        """
        接收 HR 反馈并更新对应 Agent
        action: "click" / "bookmark" / "invite" / "reject"
        """
        reward = REWARD_MAP.get(action, 0.0)
        agent  = self.get_agent(function)
        ctx    = self._build_context(vec)
        agent.update(ctx, reward)

        # 记录反馈日志
        _kl = agent.compute_kl_divergence()
        _kl = _kl if (_kl == _kl) else 0.0  # NaN guard
        log_entry = {
            "timestamp":    datetime.now().isoformat(),
            "candidate_id": candidate_id,
            "action":       action,
            "reward":       reward,
            "function":     function,
            "kl_div":       round(_kl, 4),
            "updates_total": agent.update_count,
        }
        self.feedback_log.append(log_entry)

        print(f"  [Phase6] 反馈 [{action}] → reward={reward:.1f}"
              f"  KL散度={log_entry['kl_div']:.4f}"
              f"  累计更新={agent.update_count}")

    def get_current_weights(self, function: str,
                            feature_order: List[str]) -> Dict[str, float]:
        """获取 Bandit 学习到的当前权重"""
        agent = self.get_agent(function)
        theta = agent.get_theta()
        # Softmax 归一化（确保非负）
        exp_theta = np.exp(theta - theta.max())
        normalized = exp_theta / exp_theta.sum()
        return {feature_order[i]: round(float(normalized[i]), 6)
                for i in range(min(len(feature_order), len(normalized)))}

    def save_feedback_log(self, path: str):
        """保存反馈日志"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.feedback_log, f, ensure_ascii=False, indent=2)
        print(f"  [Phase6] 反馈日志已保存: {path}")

    def _build_context(self, vec: CandidateFeatureVector) -> np.ndarray:
        """构建候选人上下文向量（固定 FEATURE_DIM 维）"""
        d = vec.to_feature_dict()
        ctx = [
            d.get("education_score", 1.0),
            d.get("s_level", 2.0),
            d.get("csi_score", 0.5),
            d.get("total_work_years", 5.0) if d.get("total_work_years") is not None else 5.0,
            d.get("career_growth_slope", 0.0),
            d.get("job_hopping_premium", 0.0),
            float(d.get("span_of_control") or 0),
            d.get("span_missing", 1.0),
            d.get("management_verb_density", 0.0),
            d.get("reporting_layer", 0.0),
            float(d.get("kpi_log_score") or 0),
            d.get("kpi_missing", 1.0),
            d.get("achievement_signal", 0.0),
            d.get("major_projects_count", 0.0),
            d.get("stability_index", 0.65),
            d.get("soft_skill_score", 0.0),
            vec.global_confidence,
            float(vec.conflict_flags),
        ]
        return np.array(ctx[:FEATURE_DIM], dtype=float)


def simulate_hr_feedback(system: ContextualBanditSystem,
                         vecs: List[CandidateFeatureVector],
                         ranked: List[Tuple[str, float, int]],
                         function: str,
                         top_k: int = 3):
    """
    模拟 HR 对 Top-K 候选人的反馈行为（演示用）
    实际部署中替换为前端 UI 事件
    """
    print(f"\n  [Phase6] 模拟 HR 反馈（Top-{top_k} 候选人）")
    vec_map = {v.candidate_id: v for v in vecs}
    for cid, score, rank in ranked[:top_k]:
        # 根据得分高低模拟不同反馈
        if rank == 1:
            action = "invite"
        elif rank == 2:
            action = "bookmark"
        else:
            action = "click"
        vec = vec_map.get(cid)
        if vec:
            system.receive_feedback(cid, action, vec, function)