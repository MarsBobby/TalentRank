#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 0: 数据资产盘点与字段分级体系
────────────────────────────────────
六大评价维度 + Tier 1/2/3 资产分级
输出标准化的字段 Schema，供后续 Phase 使用
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────
# 资产分级定义
# ─────────────────────────────────────────────────────────────

TIER_1 = "core"        # 确定性信号（缺失 → MNAR）
TIER_2 = "derived"     # 演进信号（职业轨迹动态变化）
TIER_3 = "auxiliary"   # 噪声调节信号（微调排序）


@dataclass
class FieldSchema:
    """字段 Schema 描述"""
    name: str
    dimension: str          # 评价维度
    tier: str               # 资产分级 TIER_1/2/3
    processing_logic: str   # 工业级处理逻辑
    is_mnar_prone: bool = False  # 是否 MNAR 高发字段
    default_value: Any = None    # 行业兜底默认值


# ─────────────────────────────────────────────────────────────
# 六大评价维度 & 全量字段 Schema
# ─────────────────────────────────────────────────────────────

FIELD_SCHEMAS: List[FieldSchema] = [
    # ── 教育背景 ──────────────────────────────────────────────
    FieldSchema("education_score",   "教育背景", TIER_1,
                "4阶编码(0-3) + 985/211标记",        default_value=1.0),
    FieldSchema("school_ranking",    "教育背景", TIER_1,
                "QS/USNews分桶(Top50/100/500/未上榜)", default_value=500),
    FieldSchema("major_match_score", "教育背景", TIER_2,
                "SimCSE余弦相似度 vs JD",             default_value=0.6),

    # ── 工作经历 ──────────────────────────────────────────────
    FieldSchema("s_level",           "工作经历", TIER_1,
                "BERT职级标准化(1-6级)，纠正头衔通胀", is_mnar_prone=False, default_value=2),
    FieldSchema("csi_score",         "工作经历", TIER_1,
                "公司平台指数分三档(头部/知名/普通)",   default_value=0.5),
    FieldSchema("total_work_years",  "工作经历", TIER_1,
                "总工龄(月→年)，极值截断120个月",       default_value=None),

    # ── 职场加速 ──────────────────────────────────────────────
    FieldSchema("career_growth_slope","职场加速", TIER_2,
                "拉普拉斯平滑斜率 = Δlevel/(Δt+ε)",   default_value=0.0),
    FieldSchema("job_hopping_premium","职场加速", TIER_2,
                "Σlog(curr_csi+1)-log(prev_csi+1)",    default_value=0.0),

    # ── 管理深度 ──────────────────────────────────────────────
    FieldSchema("span_of_control",   "管理深度", TIER_1,
                "下属人数；缺失保留NaN，建模MNAR",      is_mnar_prone=True, default_value=None),
    FieldSchema("management_verb_density", "管理深度", TIER_1,
                "管理/决策/统筹动词频次/100词",          default_value=0.0),
    FieldSchema("reporting_layer",   "管理深度", TIER_2,
                "汇报层级识别(向VP汇报→高层)",           default_value=0),

    # ── 核心产出 ──────────────────────────────────────────────
    FieldSchema("kpi_log_score",     "核心产出", TIER_1,
                "log(KPI原始得分+1)，压缩长尾",          is_mnar_prone=True, default_value=None),
    FieldSchema("achievement_signal","核心产出", TIER_1,
                "二值化：简历含%/增长率/对比数据→1",     default_value=0),
    FieldSchema("major_projects_count","核心产出", TIER_2,
                "战略/核心/主导项目关键词计数",           default_value=0),

    # ── 综合特质 ──────────────────────────────────────────────
    FieldSchema("stability_index",   "综合特质", TIER_3,
                "长期任职(≥12月)比率",                   default_value=0.65),
    FieldSchema("soft_skill_score",  "综合特质", TIER_3,
                "8类软素质关键词加权评分",                default_value=0.0),

    # ── P1修复新增特征 ────────────────────────────────────────
    FieldSchema("brand_achievement_binary", "核心产出", TIER_2,
                "品牌量级词/奖项词二值信号(0/1)；降低品牌岗对绝对KPI依赖",
                default_value=0),
    FieldSchema("industry_kpi_hit_count", "核心产出", TIER_2,
                "行业专属KPI提取命中数；电商GMV/研发性能等专属指标",
                is_mnar_prone=True, default_value=0),
]

# 快速访问字典
SCHEMA_MAP: Dict[str, FieldSchema] = {s.name: s for s in FIELD_SCHEMAS}

# MNAR 高发字段列表（Phase 1 特殊处理）
MNAR_FIELDS = [s.name for s in FIELD_SCHEMAS if s.is_mnar_prone]

# 按 Tier 分类
TIER1_FIELDS = [s.name for s in FIELD_SCHEMAS if s.tier == TIER_1]
TIER2_FIELDS = [s.name for s in FIELD_SCHEMAS if s.tier == TIER_2]
TIER3_FIELDS = [s.name for s in FIELD_SCHEMAS if s.tier == TIER_3]

# 全部特征名（用于 DataFrame 列顺序）
ALL_FEATURE_NAMES = [s.name for s in FIELD_SCHEMAS]


# ─────────────────────────────────────────────────────────────
# 候选人特征向量（Phase 1 输出标准格式）
# ─────────────────────────────────────────────────────────────

@dataclass
class CandidateFeatureVector:
    """
    候选人特征向量
    每个特征携带：value / confidence / is_missing / is_imputed
    """
    candidate_id:     str = ""
    file_name:        str = ""
    function:         str = ""    # 职能类别

    # 教育背景
    education_score:       float = 1.0
    education_conf:        float = 0.0
    school_ranking:        float = 500.0
    school_ranking_conf:   float = 0.0
    major_match_score:     float = 0.6
    major_match_conf:      float = 0.0

    # 工作经历
    s_level:               float = 2.0
    s_level_conf:          float = 0.0
    csi_score:             float = 0.5
    csi_score_conf:        float = 0.0
    total_work_years:      Optional[float] = None
    work_years_conf:       float = 0.0

    # 职场加速
    career_growth_slope:   float = 0.0
    job_hopping_premium:   float = 0.0

    # 管理深度
    span_of_control:       Optional[float] = None   # NaN → MNAR
    span_missing:          int   = 1                # 0=有数据 1=缺失
    management_verb_density: float = 0.0
    reporting_layer:       float = 0.0

    # 核心产出
    kpi_log_score:         Optional[float] = None
    kpi_missing:           int   = 1
    achievement_signal:    int   = 0
    major_projects_count:  int   = 0

    # 综合特质
    stability_index:       float = 0.65
    soft_skill_score:      float = 0.0

    # 质量元数据
    global_confidence:     float = 0.0   # 综合置信度
    needs_human_review:    bool  = False
    conflict_flags:        int   = 0     # 冲突数量

    # 原始文本（供 Phase 2 NLP 使用）
    raw_text:              str   = ""

    # ─── P1修复新增字段 ───────────────────────────────────────
    # 品牌岗软性成就量级二值信号（量级词/奖项词命中→1）
    brand_achievement_binary: int  = 0
    # 研发岗技术职级是否经过大厂体系校准（P/L/T系列→标准1-6级）
    tech_level_calibrated:    bool = False
    # 行业专属KPI提取命中数（区别于通用 achievement_signal）
    industry_kpi_hit_count:   int  = 0

    def to_feature_dict(self) -> Dict[str, Any]:
        """返回模型可用的特征字典（去除元数据字段）"""
        return {
            "education_score":          self.education_score,
            "school_ranking":           self.school_ranking,
            "major_match_score":        self.major_match_score,
            "s_level":                  self.s_level,
            "csi_score":                self.csi_score,
            "total_work_years":         self.total_work_years if self.total_work_years is not None else float("nan"),
            "career_growth_slope":      self.career_growth_slope,
            "job_hopping_premium":      self.job_hopping_premium,
            "span_of_control":          self.span_of_control if self.span_of_control is not None else float("nan"),
            "span_missing":             self.span_missing,
            "management_verb_density":  self.management_verb_density,
            "reporting_layer":          self.reporting_layer,
            "kpi_log_score":            self.kpi_log_score if self.kpi_log_score is not None else float("nan"),
            "kpi_missing":              self.kpi_missing,
            "achievement_signal":       self.achievement_signal,
            "major_projects_count":     self.major_projects_count,
            "stability_index":          self.stability_index,
            "soft_skill_score":         self.soft_skill_score,
            # P1修复：品牌量级二值信号（接入 SHAP 解释链）
            "brand_achievement_binary": float(self.brand_achievement_binary),
        }


def print_field_inventory():
    """打印字段资产清单（Phase 0 报告）"""
    print("=" * 70)
    print("  Phase 0 · 数据资产盘点报告")
    print("=" * 70)
    print(f"  总字段数: {len(FIELD_SCHEMAS)}")
    print(f"  Tier-1 (核心资产): {len(TIER1_FIELDS)} 个 → {TIER1_FIELDS}")
    print(f"  Tier-2 (衍生资产): {len(TIER2_FIELDS)} 个 → {TIER2_FIELDS}")
    print(f"  Tier-3 (辅助资产): {len(TIER3_FIELDS)} 个 → {TIER3_FIELDS}")
    print(f"  MNAR高发字段:      {MNAR_FIELDS}")
    print("=" * 70)
    print(f"\n  {'字段名':<28} {'维度':<10} {'Tier':<8} {'MNAR'}")
    print("  " + "-" * 60)
    for s in FIELD_SCHEMAS:
        mnar = "⚠ MNAR" if s.is_mnar_prone else ""
        print(f"  {s.name:<28} {s.dimension:<10} {s.tier:<8} {mnar}")
    print()


if __name__ == "__main__":
    print_field_inventory()