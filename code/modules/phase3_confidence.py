#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3: 鲁棒性工程与置信度触发器
────────────────────────────────────
1. 置信度字段结构（value/confidence/is_missing/is_imputed）
2. 贝叶斯归纳填补（非 MNAR 字段的最优估算）
3. 低置信度警告 + 人工审核路由（职能差异化阈值）
4. 综合质量报告

修复记录（v2.0）：
  P1 - audit_confidence 接受 function 参数，使用 CONFIDENCE_THRESHOLDS
       差异化阈值（研发/生产 0.48，其余 0.60），消除研发 75%、
       生产 62.5% 的过度人工审核路由问题
  P1 - generate_quality_report 同步接受 function 参数
  P1 - 新增 get_confidence_threshold 公开函数，供外部调用（api.py / main.py）
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.phase0_data_assets import CandidateFeatureVector, MNAR_FIELDS
from config import CONFIDENCE as CONF, CONFIDENCE_THRESHOLDS


# ─────────────────────────────────────────────────────────────
# 贝叶斯后验填补（简化版）
# Prior = 行业分位点统计；Posterior = 已有特征加权估算
# ─────────────────────────────────────────────────────────────

# 各行业先验：(mean, std) per feature
BAYESIAN_PRIORS: Dict[str, Dict[str, tuple]] = {
    "品牌": {
        "total_work_years": (8.0, 3.0),
        "kpi_log_score":    (4.0, 1.5),
        "span_of_control":  (15.0, 20.0),
    },
    "人力资源": {
        "total_work_years": (7.5, 3.5),
        "kpi_log_score":    (3.5, 1.0),
        "span_of_control":  (20.0, 25.0),
    },
    "生产": {
        "total_work_years": (10.0, 4.0),
        "kpi_log_score":    (3.0, 1.0),
        "span_of_control":  (50.0, 40.0),
    },
    "销售": {
        "total_work_years": (7.0, 3.0),
        "kpi_log_score":    (5.0, 2.0),
        "span_of_control":  (10.0, 15.0),
    },
    "研发": {
        "total_work_years": (6.0, 2.5),
        "kpi_log_score":    (2.5, 1.0),
        "span_of_control":  (8.0, 10.0),
    },
    "电商": {
        "total_work_years": (6.0, 2.5),
        "kpi_log_score":    (4.5, 1.5),
        "span_of_control":  (12.0, 15.0),
    },
}
DEFAULT_PRIORS = {
    "total_work_years": (7.5, 3.0),
    "kpi_log_score":    (3.5, 1.5),
    "span_of_control":  (15.0, 20.0),
}


def bayesian_impute(vec: CandidateFeatureVector) -> CandidateFeatureVector:
    """
    贝叶斯归纳填补：仅对非 MNAR 字段（work_years）填补
    MNAR 字段（span, kpi）保留 NaN + Missing Flag
    """
    priors = BAYESIAN_PRIORS.get(vec.function, DEFAULT_PRIORS)

    # work_years：非 MNAR，可填补
    if vec.total_work_years is None:
        mu, sigma = priors.get("total_work_years", (7.5, 3.0))
        # 利用 s_level 作为调节：level 越高 → 工龄越长
        adjusted = mu + (vec.s_level - 3) * sigma * 0.3
        vec.total_work_years = max(1.0, round(adjusted, 1))
        vec.work_years_conf  = CONF["bayesian_fill"]

    # span_of_control：MNAR → 不填补，保留 NaN + flag
    # kpi_log_score：MNAR → 不填补

    return vec


# ─────────────────────────────────────────────────────────────
# 置信度触发器
# ─────────────────────────────────────────────────────────────

@dataclass
class ConfidenceAudit:
    """置信度审核结果"""
    candidate_id:       str
    global_confidence:  float
    needs_review:       bool
    warning_fields:     List[str]
    mnar_fields_present: List[str]   # 有数据的 MNAR 字段
    mnar_fields_missing: List[str]   # 缺失的 MNAR 字段
    quality_tier:       str          # "green" / "yellow" / "red"
    review_reason:      str          = ""


def get_confidence_threshold(function: str = "") -> float:
    """
    P1修复：返回职能差异化置信度阈值。
    研发/生产简历的信息天花板本身低于其他职能，
    统一使用 0.6 会导致过度路由至人工审核队列。

    Args:
        function: 职能名称，如 '研发'、'生产'、'品牌' 等
    Returns:
        该职能的置信度阈值
    """
    return CONFIDENCE_THRESHOLDS.get(function, CONFIDENCE_THRESHOLDS["default"])


def audit_confidence(vec: CandidateFeatureVector,
                     function: str = "") -> "ConfidenceAudit":
    """
    运行置信度审核，输出 ConfidenceAudit。
    P1修复：使用 get_confidence_threshold(function) 替代固定 0.6 阈值。
    """
    warnings = []

    # 检查 MNAR 字段状态
    mnar_present, mnar_missing = [], []
    if vec.span_missing == 0:
        mnar_present.append("span_of_control")
    else:
        mnar_missing.append("span_of_control")

    if vec.kpi_missing == 0:
        mnar_present.append("kpi_log_score")
    else:
        mnar_missing.append("kpi_log_score")

    # 警告字段
    if vec.education_conf < 0.5:
        warnings.append("education_conf 低")
    if vec.s_level_conf < 0.5:
        warnings.append("s_level_conf 低")
    if vec.work_years_conf < 0.5:
        warnings.append("work_years_conf 低")
    if vec.conflict_flags > 0:
        warnings.append(f"时间冲突 {vec.conflict_flags} 处")

    # 质量分级（P1修复：使用职能差异化阈值）
    threshold = get_confidence_threshold(function)
    gc = vec.global_confidence
    if gc >= 0.75:
        tier = "green"
    elif gc >= threshold:
        tier = "yellow"
    else:
        tier = "red"

    reason = ""
    if tier == "red":
        reason = "综合置信度 < 0.60，建议 HR 补全关键字段后重新评估"
    elif tier == "yellow":
        reason = "置信度中等，请重点核验：" + ", ".join(warnings[:3])

    return ConfidenceAudit(
        candidate_id=vec.candidate_id,
        global_confidence=gc,
        needs_review=vec.needs_human_review,
        warning_fields=warnings,
        mnar_fields_present=mnar_present,
        mnar_fields_missing=mnar_missing,
        quality_tier=tier,
        review_reason=reason,
    )


# ─────────────────────────────────────────────────────────────
# 批量质量报告
# ─────────────────────────────────────────────────────────────

def generate_quality_report(vecs: List[CandidateFeatureVector],
                            function: str = "") -> Dict:
    """
    生成整批候选人的数据质量报告。
    P1修复：接受 function 参数，差异化置信度阈值路由。
    """
    total = len(vecs)
    if total == 0:
        return {"error": "空列表"}

    audits = [audit_confidence(v, function) for v in vecs]

    # 统计
    green_count  = sum(1 for a in audits if a.quality_tier == "green")
    yellow_count = sum(1 for a in audits if a.quality_tier == "yellow")
    red_count    = sum(1 for a in audits if a.quality_tier == "red")
    avg_conf     = round(sum(a.global_confidence for a in audits) / total, 3)

    mnar_span_has    = sum(1 for a in audits if "span_of_control" in a.mnar_fields_present)
    mnar_kpi_has     = sum(1 for a in audits if "kpi_log_score"   in a.mnar_fields_present)

    # 需要人工审核的候选人
    review_list = [a.candidate_id for a in audits if a.needs_review]

    report = {
        "total_candidates": total,
        "avg_confidence":   avg_conf,
        "quality_distribution": {
            "green (≥0.75)":  green_count,
            "yellow (0.60-0.75)": yellow_count,
            "red (<0.60)":    red_count,
        },
        "mnar_statistics": {
            "span_of_control_available": f"{mnar_span_has}/{total} ({100*mnar_span_has//total}%)",
            "kpi_log_score_available":   f"{mnar_kpi_has}/{total} ({100*mnar_kpi_has//total}%)",
        },
        "human_review_queue": review_list,
        "detailed_audits": [
            {
                "id": a.candidate_id,
                "confidence": a.global_confidence,
                "tier": a.quality_tier,
                "warnings": a.warning_fields,
                "reason": a.review_reason,
            }
            for a in audits
        ],
    }

    return report


def print_quality_report(report: Dict):
    """美化打印质量报告"""
    print("\n" + "=" * 65)
    print("  Phase 3 · 数据质量报告 & 置信度触发器")
    print("=" * 65)
    print(f"  总候选人数:    {report['total_candidates']}")
    print(f"  平均置信度:    {report['avg_confidence']}")
    dist = report["quality_distribution"]
    print(f"  质量分布:")
    print(f"    🟢 Green  (≥0.75): {dist['green (≥0.75)']} 人")
    print(f"    🟡 Yellow (0.60+): {dist['yellow (0.60-0.75)']} 人")
    print(f"    🔴 Red    (<0.60): {dist['red (<0.60)']} 人")
    mnar = report["mnar_statistics"]
    print(f"  MNAR 统计:")
    print(f"    下属人数有数据: {mnar['span_of_control_available']}")
    print(f"    KPI 有数据:     {mnar['kpi_log_score_available']}")
    if report["human_review_queue"]:
        print(f"  ⚠ 人工审核队列: {report['human_review_queue']}")
    print()
    for a in report["detailed_audits"]:
        tier_icon = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(a["tier"], "⚪")
        print(f"  {tier_icon} {a['id']:<15} conf={a['confidence']:.3f}"
              + (f"  [{', '.join(a['warnings'][:2])}]" if a["warnings"] else ""))
    print()