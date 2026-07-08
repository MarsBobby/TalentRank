#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局配置文件 - 工业级人才优选体系

修复记录（v2.0）：
  P0 - CONFIDENCE 中 human_review_threshold 已被 CONFIDENCE_THRESHOLDS 字典替代，
       由 phase3_confidence.get_confidence_threshold(function) 统一调用
  P1 - 新增 CONFIDENCE_THRESHOLDS：研发/生产阈值 0.48，其余 0.60
  P2 - 新增 FUNCTION_KL_LAMBDA：销售岗 kpi_log_score 重要性已达 0.25，
       收紧 KL 约束防止权重漂移
"""

# ─────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "resumes")
LOG_DIR  = os.path.join(BASE_DIR, "logs")
OUT_DIR  = os.path.join(BASE_DIR, "outputs")

# ─────────────────────────────────────────────
# Phase 3 置信度基础配置
# ─────────────────────────────────────────────
CONFIDENCE = {
    "exact_match":    1.0,   # 明确文本直接提取
    "regex_match":    0.7,   # 正则表达式命中
    "context_infer":  0.5,   # 语义/上下文推断
    "bayesian_fill":  0.4,   # 贝叶斯归纳填补
    "default_fill":   0.3,   # 行业默认值兜底
    "not_found":      0.0,   # 完全缺失
    # 保留兜底值；实际路由阈值请使用 CONFIDENCE_THRESHOLDS
    "human_review_threshold": 0.6,
}

# ─────────────────────────────────────────────
# P1修复 — 职能差异化置信度阈值
# ─────────────────────────────────────────────
# 研发/生产简历信息天花板低（技术栈关键词多、管理叙述少），
# 统一使用 0.6 会导致 75%（研发）和 62.5%（生产）候选人被过度路由至人工审核队列。
CONFIDENCE_THRESHOLDS = {
    "研发":     0.48,
    "生产":     0.48,
    "电商":     0.60,
    "品牌":     0.60,
    "销售":     0.60,
    "人力资源": 0.60,
    "default":  0.60,
}

# ─────────────────────────────────────────────
# Phase 4 AHP 默认权重（品牌岗位示例）
# 可按岗位在 main.py 中覆盖
# ─────────────────────────────────────────────
AHP_WEIGHTS_BRANDING = {
    "education_score":      0.10,
    "s_level":              0.15,
    "career_growth_slope":  0.15,
    "span_of_control":      0.10,
    "kpi_log_score":        0.20,
    "stability_index":      0.10,
    "achievement_signal":   0.10,
    "soft_skill_score":     0.10,
}

AHP_WEIGHTS_PRODUCTION = {
    "education_score":      0.08,
    "s_level":              0.18,
    "career_growth_slope":  0.12,
    "span_of_control":      0.15,
    "kpi_log_score":        0.20,
    "stability_index":      0.15,
    "achievement_signal":   0.07,
    "soft_skill_score":     0.05,
}

# ─────────────────────────────────────────────
# Phase 4 EWM 动态混合策略
# ─────────────────────────────────────────────
AHP_PURE_THRESHOLD   = 5    # N < 5 → 纯AHP
AHP_EWM_THRESHOLD    = 20   # N >= 20 → 标准混合
AHP_WEIGHT           = 0.5
EWM_WEIGHT           = 0.5

# ─────────────────────────────────────────────
# Phase 5 LambdaMART 训练参数
# ─────────────────────────────────────────────
LAMBDAMART_PARAMS = {
    "objective":       "lambdarank",
    "metric":          "ndcg",
    "ndcg_eval_at":    [5, 10],
    "num_leaves":      31,
    "learning_rate":   0.05,
    "n_estimators":    200,
    "min_child_samples": 5,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "verbose":         -1,
}

# ─────────────────────────────────────────────
# Phase 6 LinUCB Bandit 全局参数
# ─────────────────────────────────────────────
LINUCB_ALPHA     = 1.0   # 探索系数
KL_LAMBDA        = 0.1   # 全局权重漂移惩罚系数（兜底值）
MAX_WEIGHT_DELTA = 0.2   # 单次更新最大漂移量

# P2修复 — 职能差异化 KL 散度惩罚系数
# 销售岗 kpi_log_score SHAP 重要性已达 0.25，单一 HR 异常调整风险高，
# 需收紧 λ 防止过度拟合单一 HR 偏好，同时保留其他职能的探索空间。
FUNCTION_KL_LAMBDA = {
    "销售":    0.25,   # 收紧：防止 kpi_log_score 权重被异常拉高
    "研发":    0.15,   # 略收紧：tech_level 校准后权重稳定性更重要
    "default": 0.10,   # 其余职能维持原始值
}

# ─────────────────────────────────────────────
# 职能 → 行业映射
# ─────────────────────────────────────────────
FUNCTION_INDUSTRY_MAP = {
    "品牌": "品牌",
    "人力资源": "人力资源",
    "生产": "生产",
    "电商": "电商",
    "研发": "研发",
    "销售": "销售",
}

SUPPORTED_FUNCTIONS = list(FUNCTION_INDUSTRY_MAP.keys())

# ─────────────────────────────────────────────
# Phase 4 AHP 默认权重（品牌岗位示例）
# 可按岗位在 main.py 中覆盖
# ─────────────────────────────────────────────
AHP_WEIGHTS_BRANDING = {
    "education_score":      0.10,
    "s_level":              0.15,
    "career_growth_slope":  0.15,
    "span_of_control":      0.10,
    "kpi_log_score":        0.20,
    "stability_index":      0.10,
    "achievement_signal":   0.10,
    "soft_skill_score":     0.10,
}

AHP_WEIGHTS_PRODUCTION = {
    "education_score":      0.08,
    "s_level":              0.18,
    "career_growth_slope":  0.12,
    "span_of_control":      0.15,
    "kpi_log_score":        0.20,
    "stability_index":      0.15,
    "achievement_signal":   0.07,
    "soft_skill_score":     0.05,
}

# ─────────────────────────────────────────────
# Phase 4 EWM 动态混合策略
# ─────────────────────────────────────────────
AHP_PURE_THRESHOLD   = 5    # N < 5 → 纯AHP
AHP_EWM_THRESHOLD    = 20   # N >= 20 → 标准混合
AHP_WEIGHT           = 0.5
EWM_WEIGHT           = 0.5

# ─────────────────────────────────────────────
# Phase 5 LambdaMART 训练参数
# ─────────────────────────────────────────────
LAMBDAMART_PARAMS = {
    "objective":       "lambdarank",
    "metric":          "ndcg",
    "ndcg_eval_at":    [5, 10],
    "num_leaves":      31,
    "learning_rate":   0.05,
    "n_estimators":    200,
    "min_child_samples": 5,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "verbose":         -1,
}

# ─────────────────────────────────────────────
# Phase 6 LinUCB Bandit 参数
# ─────────────────────────────────────────────
LINUCB_ALPHA     = 1.0   # 探索系数
KL_LAMBDA        = 0.1   # 权重漂移惩罚系数
MAX_WEIGHT_DELTA = 0.2   # 单次更新最大漂移量

# ─────────────────────────────────────────────
# 职能 → 行业映射
# ─────────────────────────────────────────────
FUNCTION_INDUSTRY_MAP = {
    "品牌": "品牌",
    "人力资源": "人力资源",
    "生产": "生产",
    "电商": "电商",
    "研发": "研发",
    "销售": "销售",
}

SUPPORTED_FUNCTIONS = list(FUNCTION_INDUSTRY_MAP.keys())