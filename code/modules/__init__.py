# -*- coding: utf-8 -*-
"""
智能简历解析与多维排序系统 (Resume Ranking & Parsing System)
─────────────────────────────────────────────────────────────
包含从数据处理 (Phase 0) 到评估体系 (Phase 8) 的全流程模块。
"""

__version__ = "1.0.0"

# ── Phase 0: 数据资产盘点 ──
from .phase0_data_assets import (
    CandidateFeatureVector,
    FieldSchema,
    FIELD_SCHEMAS,
    print_field_inventory
)

# ── Phase 1: 特征工程 (MNAR 感知) ──
from .phase1_feature_engineering import MNARAwareFeatureEngineer

# ── Phase 2: NLP 语义提取 ──
from .phase2_nlp_extractor import (
    ExtractorAgent,
    ExtractionReport,
    print_extraction_report
)

# ── Phase 3: 置信度与鲁棒性 ──
from .phase3_confidence import (
    ConfidenceAudit,
    audit_confidence,
    bayesian_impute,
    generate_quality_report,
    print_quality_report
)

# ── Phase 4: 动态混合赋权 (AHP + EWM) ──
from .phase4_weighting import (
    DynamicWeighter,
    AHPWeighter,
    EWMWeighter,
    print_weight_report
)

# ── Phase 5: LambdaMART 排序 ──
from .phase5_lambdamart import (
    LambdaMARTRanker,
    print_ranking_result
)

# ── Phase 6: Bandit 在线反馈 (LinUCB) ──
from .phase6_bandit import (
    ContextualBanditSystem,
    simulate_hr_feedback
)

# ── Phase 7: SHAP 解释性 ──
from .phase7_explainability import (
    SHAPExplainabilityAgent,
    print_global_explanation
)

# ── Phase 8: 评估体系 ──
from .phase8_evaluation import (
    EvaluationSystem,
    build_gold_standard,
    print_evaluation_report
)

# ── 定义模块对外暴露的 API (__all__) ──
__all__ = [
    # Phase 0
    "CandidateFeatureVector",
    "FieldSchema",
    "FIELD_SCHEMAS",
    "print_field_inventory",
    
    # Phase 1
    "MNARAwareFeatureEngineer",
    
    # Phase 2
    "ExtractorAgent",
    "ExtractionReport",
    "print_extraction_report",
    
    # Phase 3
    "ConfidenceAudit",
    "audit_confidence",
    "bayesian_impute",
    "generate_quality_report",
    "print_quality_report",
    
    # Phase 4
    "DynamicWeighter",
    "AHPWeighter",
    "EWMWeighter",
    "print_weight_report",
    
    # Phase 5
    "LambdaMARTRanker",
    "print_ranking_result",
    
    # Phase 6
    "ContextualBanditSystem",
    "simulate_hr_feedback",
    
    # Phase 7
    "SHAPExplainabilityAgent",
    "print_global_explanation",
    
    # Phase 8
    "EvaluationSystem",
    "build_gold_standard",
    "print_evaluation_report"
]