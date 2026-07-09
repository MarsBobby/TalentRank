#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
api.py — 人才优选系统 · FastAPI 后端服务
启动方式: uvicorn api:app --host 0.0.0.0 --port 8000 --reload

评估体系修复记录（v2.2）：
  原 _mock_ndcg_gold_relevance / _mock_mrr_relevant_ids 使用 MD5 哈希随机分配
  相关性分数，与候选人真实质量无关，NDCG/MRR 结果完全无意义。

  本版本改为三层递进评估策略：

  Layer 1 — HR 黄金标准文件（最可信）
    路径：<BASE_DIR>/gold_standards/<function>.json
    格式：{"LHQ": 4, "XJJ": 3, "LDD": 3, ...}（HR 打分 0-4）
    HR 对该职能简历评过分后保存此文件，系统优先加载。

  Layer 2 — 信号驱动的代理金标准（有业务依据）
    当无黄金文件时，用候选人自身的高置信度特征（s_level、education、
    work_years、achievement_signal、kpi_log_score）以一套独立于
    Phase4 AHP 权重的固定规则打分，映射到 0-4 等级。
    与 Phase4/5 的排序函数使用不同权重，打破循环依赖。
    置信度 < 0.5 的候选人贡献大幅降权，避免低质量特征污染评估基线。

  Layer 3 — 指标可信度标注
    结果中附带 metrics_source 字段，告知前端当前用的是哪一层，
    HR 可据此判断指标的参考价值。
"""

import os
import sys
import uuid
import json
import math
import hashlib
import shutil
import tempfile
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── 引入流水线模块 ────────────────────────────────────────────
from docx_extractor import DocxExtractor
from modules.phase2_nlp_extractor import ExtractorAgent, print_extraction_report
from modules.phase3_confidence import (
    bayesian_impute, audit_confidence, generate_quality_report,
    get_confidence_threshold
)
from modules.phase4_weighting import DynamicWeighter, FEATURE_ORDER
from modules.phase5_lambdamart import LambdaMARTRanker
from modules.phase6_bandit import ContextualBanditSystem, REWARD_MAP
from modules.phase7_explainability import SHAPExplainabilityAgent, FEATURE_CN
from modules.phase0_data_assets import CandidateFeatureVector

# ── 日志 ────────────────────────────────────────────────────
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, f"backend_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger(__name__)
logger.info(f"后端服务日志系统初始化成功，日志文件：{log_file}")

# ── 黄金标准文件目录 ──────────────────────────────────────────
GOLD_DIR = os.path.join(BASE_DIR, "gold_standards")
os.makedirs(GOLD_DIR, exist_ok=True)

# ── 全局单例 ─────────────────────────────────────────────────
_docx_extractor  = DocxExtractor()
_extractor_agent = ExtractorAgent()
_weighter        = DynamicWeighter()
_ranker          = LambdaMARTRanker()
_bandit          = ContextualBanditSystem()
_shap_agent      = SHAPExplainabilityAgent(weights={})

_session_vecs: Dict[str, Any] = {}

FUNCTION_DIR_MAP = {
    "品牌":     "branding",
    "电商":     "ecommerce",
    "人力资源": "hr",
    "生产":     "manufacturing",
    "研发":     "rnd",
    "销售":     "sales",
}

RADAR_DIMS = [
    ("education_score",        "教育背景", 3.0),
    ("s_level",                "职级水平", 6.0),
    ("career_growth_slope",    "成长轨迹", 2.0),
    ("management_verb_density","管理深度", 20.0),
    ("kpi_log_score",          "核心产出", 10.0),
    ("soft_skill_score",       "综合特质", 10.0),
]

# ── FastAPI App ───────────────────────────────────────────────
app = FastAPI(
    title="TalentSystem API",
    description="工业级人才优选体系 · Phase 0-8 全链路服务",
    version="2.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ══════════════════════════════════════════════════════════════
#  Pydantic 数据模型
# ══════════════════════════════════════════════════════════════

class FeedbackRequest(BaseModel):
    candidate_id: str
    action:       str
    function:     str


class HRRatingRequest(BaseModel):
    """HR 为候选人打分的请求体（用于构建黄金标准集）"""
    function:    str
    ratings:     Dict[str, int]   # {candidate_id: 0-4}
    rater_name:  Optional[str] = ""


class CandidateCard(BaseModel):
    rank:              int
    candidate_id:      str
    file_name:         str
    score:             float
    confidence:        float
    needs_review:      bool
    target_position:   str
    current_salary:    Optional[str]
    location:          str
    s_level:           float
    total_work_years:  Optional[float]
    span_missing:      int
    kpi_missing:       int
    radar:             List[Dict]
    shap_positive:     List[Dict]
    shap_negative:     List[Dict]
    achievement_signal: int
    major_projects_count: int
    warnings:          List[str]


class AnalyzeResponse(BaseModel):
    job_id:          str
    function:        str
    total:           int
    run_time:        str
    candidates:      List[CandidateCard]
    weights:         Dict[str, float]
    quality:         Dict
    metrics:         Dict
    metrics_source:  str   # "hr_gold" | "signal_proxy" — 告知前端指标可信度


# ══════════════════════════════════════════════════════════════
#  三层评估金标准体系
# ══════════════════════════════════════════════════════════════

def _load_hr_gold_standard(function: str) -> Optional[Dict[str, float]]:
    """
    Layer 1：加载 HR 黄金标准文件。
    文件路径：gold_standards/<function>.json
    格式：{"LHQ": 4, "XJJ": 3, "LDD": 2, ...}
    返回 None 表示文件不存在或为空。
    """
    gold_path = os.path.join(GOLD_DIR, f"{function}.json")
    if not os.path.exists(gold_path):
        return None
    try:
        with open(gold_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # 过滤：只保留分值在 0-4 范围内的有效条目
        gold = {
            cid: float(max(0, min(4, int(score))))
            for cid, score in raw.items()
            if isinstance(score, (int, float))
        }
        if len(gold) < 2:
            logger.warning(f"[评估] {function} 黄金文件有效条目 < 2，降级到代理模式")
            return None
        logger.info(f"[评估] 加载 HR 黄金标准：{function}，共 {len(gold)} 条")
        return gold
    except Exception as e:
        logger.warning(f"[评估] 读取黄金文件失败：{gold_path}，原因：{e}")
        return None


def _build_proxy_gold_relevance(
    vecs: List[CandidateFeatureVector],
) -> Dict[str, float]:
    """
    Layer 2：信号驱动的代理金标准。

    设计原则：
    1. 权重独立于 Phase4 AHP / Phase5 回退权重，避免循环依赖。
       Phase4 侧重 kpi_log_score(0.20) + s_level(0.15)；
       此处侧重 education + work_years + achievement_signal，
       反映"资历可信度"维度而非"产出表现"维度。
    2. 置信度加权：全局置信度 < 0.5 时，该候选人分数乘以 0.6 的惩罚系数，
       避免低质量简历的噪声特征污染评估基线。
    3. 映射到 0-4 等级（NDCG 标准格式）：
       在批次内做 MinMax 归一化后乘以 4，保留 1 位小数。

    独立权重配置（刻意与 AHP 权重不同）：
      education_score:    0.25  —— 学历是最稳定的可验证信号
      s_level:            0.20  —— 职级语义校准后的层级
      total_work_years:   0.20  —— 总工龄（经贝叶斯填补，几乎无缺失）
      achievement_signal: 0.20  —— 二值信号，有量化 → 1
      kpi_log_score:      0.10  —— MNAR，有则加权，无则不惩罚
      stability_index:    0.05  —— 稳定性
    """
    PROXY_WEIGHTS = {
        "education_score":    0.25,
        "s_level":            0.20,
        "total_work_years":   0.20,
        "achievement_signal": 0.20,
        "kpi_log_score":      0.10,
        "stability_index":    0.05,
    }

    raw_scores: Dict[str, float] = {}
    for vec in vecs:
        d = vec.to_feature_dict()
        score = 0.0
        for feat, w in PROXY_WEIGHTS.items():
            val = d.get(feat)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                # MNAR 缺失：不加分也不扣分，仅跳过（区别于填0惩罚）
                continue
            # MinMax 边界归一化（各特征量纲差异较大，先粗归一化）
            normalizers = {
                "education_score": 3.0,
                "s_level": 6.0,
                "total_work_years": 20.0,
                "achievement_signal": 1.0,
                "kpi_log_score": 10.0,
                "stability_index": 1.0,
            }
            norm_val = min(1.0, float(val) / normalizers.get(feat, 1.0))
            score += norm_val * w

        # 置信度惩罚：低置信度候选人分数打折
        conf = vec.global_confidence
        if conf < 0.5:
            score *= 0.6
        elif conf < 0.65:
            score *= 0.85

        raw_scores[vec.candidate_id] = score

    if not raw_scores:
        return {}

    # MinMax 归一化到 [0, 4]
    min_s = min(raw_scores.values())
    max_s = max(raw_scores.values())
    span  = max_s - min_s

    gold: Dict[str, float] = {}
    for cid, s in raw_scores.items():
        if span > 0:
            gold[cid] = round((s - min_s) / span * 4, 2)
        else:
            gold[cid] = 2.0  # 所有候选人得分相同时，给中间值

    return gold


def _get_relevant_ids_from_gold(
    gold: Dict[str, float],
    threshold: float = 2.5,
) -> List[str]:
    """
    从金标准中取出相关性分数 >= threshold 的候选人作为 MRR 正例集合。
    threshold=2.5 对应 4 分制的"较好"档位，确保正例集合不为空。
    """
    relevant = [cid for cid, score in gold.items() if score >= threshold]
    if not relevant:
        # 若无候选人达标，取前 1/3（至少 1 人）
        sorted_ids = sorted(gold.keys(), key=lambda c: gold[c], reverse=True)
        n = max(1, len(sorted_ids) // 3)
        relevant = sorted_ids[:n]
    return relevant


def _compute_evaluation_metrics(
    pred_order: List[str],
    vecs: List[CandidateFeatureVector],
    function: str,
) -> Tuple[Dict, str]:
    """
    三层递进评估：返回 (metrics_dict, source_label)。
    source_label: "hr_gold"（HR评分） | "signal_proxy"（代理金标准）
    """
    from modules.phase8_evaluation import ndcg_at_k, mean_reciprocal_rank

    # Layer 1：尝试加载 HR 黄金标准
    gold = _load_hr_gold_standard(function)
    source = "hr_gold"

    if gold is None:
        # Layer 2：信号驱动代理金标准
        gold = _build_proxy_gold_relevance(vecs)
        source = "signal_proxy"
        logger.info(f"[评估] {function} 使用代理金标准，共 {len(gold)} 位候选人")

    if not gold:
        return {"NDCG@5": None, "NDCG@10": None, "MRR": None}, "no_data"

    # 只对出现在 gold 里的候选人进行评估（HR 评分不一定覆盖全部）
    relevant = _get_relevant_ids_from_gold(gold)

    metrics = {
        "NDCG@5":  ndcg_at_k(pred_order, gold, 5),
        "NDCG@10": ndcg_at_k(pred_order, gold, 10),
        "MRR":     mean_reciprocal_rank(pred_order, relevant),
    }
    return metrics, source


# ══════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════

def _safe(val, fallback=0.0):
    if val is None: return fallback
    if isinstance(val, float) and math.isnan(val): return fallback
    return val


def _build_radar(vec) -> List[Dict]:
    d = vec.to_feature_dict()
    result = []
    for feat, label, max_val in RADAR_DIMS:
        raw = _safe(d.get(feat), 0.0)
        pct = min(100, round(raw / max_val * 100, 1))
        result.append({"label": label, "value": pct, "raw": round(raw, 2)})
    return result


def _run_pipeline_on_files(files_paths: List[str], function: str):
    """核心流水线，返回结构化结果"""
    vecs, reports = [], []

    for fpath in files_paths:
        fname = os.path.basename(fpath)
        try:
            from urllib.parse import unquote
            fname_display = unquote(fname)
        except Exception:
            fname_display = fname

        extraction = _docx_extractor.extract_from_file(fpath)
        if extraction["status"] == "failure":
            logger.warning(f"提取失败: {fname} → {extraction['error']}")
            continue

        vec, report = _extractor_agent.extract(
            raw_text=extraction["text"],
            file_name=fname_display,
            function=function,
        )
        vec = bayesian_impute(vec)
        vecs.append(vec)
        reports.append(report)

    if not vecs:
        return None, None, None, None, None

    # Phase 3（职能差异化置信度阈值）
    quality_report = generate_quality_report(vecs, function=function)

    # Phase 4
    weights = _weighter.compute_weights(vecs, function=function)

    # Phase 5
    ranked = _ranker.rank(vecs)

    # Phase 6
    _bandit.initialize_from_weights(function, weights, FEATURE_ORDER)

    # Phase 7
    global _shap_agent
    _shap_agent = SHAPExplainabilityAgent(model=_ranker.model, weights=weights)

    _session_vecs[function] = {v.candidate_id: v for v in vecs}

    return vecs, reports, ranked, weights, quality_report


def _build_candidate_card(vec, report, rank: int, score: float,
                           all_vecs, weights) -> CandidateCard:
    shap_exp = _shap_agent.explain_candidate(vec, all_vecs)

    pos_items = [
        {"feature": f, "cn_name": FEATURE_CN.get(f, f), "value": round(v, 3)}
        for f, v in shap_exp["top_positive"]
    ]
    neg_items = [
        {"feature": f, "cn_name": FEATURE_CN.get(f, f), "value": round(v, 3)}
        for f, v in shap_exp["top_negative"]
    ]

    return CandidateCard(
        rank=rank,
        candidate_id=vec.candidate_id,
        file_name=vec.file_name,
        score=round(score, 4),
        confidence=round(vec.global_confidence, 3),
        needs_review=vec.needs_human_review,
        target_position=getattr(report, "target_position", "") or "",
        current_salary=getattr(report, "current_salary", None),
        location=getattr(report, "location", "") or "",
        s_level=vec.s_level,
        total_work_years=vec.total_work_years,
        span_missing=vec.span_missing,
        kpi_missing=vec.kpi_missing,
        radar=_build_radar(vec),
        shap_positive=pos_items,
        shap_negative=neg_items,
        achievement_signal=vec.achievement_signal,
        major_projects_count=vec.major_projects_count,
        warnings=getattr(report, "warnings", []),
    )


# ══════════════════════════════════════════════════════════════
#  路由
# ══════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"message": "TalentSystem API running. 前端文件未找到。"})


@app.get("/api/health")
async def health():
    return {"status": "ok", "time": datetime.now().isoformat()}


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(
    files: List[UploadFile] = File(...),
    function: str = Form("品牌"),
):
    """上传简历文件并运行 Phase 0-8 全链路分析"""
    if not files:
        raise HTTPException(status_code=400, detail="请至少上传一份简历文件")

    tmp_dir = tempfile.mkdtemp(prefix="talent_")
    saved_paths = []
    try:
        for upload in files:
            if not upload.filename.endswith(".docx"):
                continue
            dest = os.path.join(tmp_dir, upload.filename)
            with open(dest, "wb") as f:
                content = await upload.read()
                f.write(content)
            saved_paths.append(dest)

        if not saved_paths:
            raise HTTPException(status_code=400, detail="未找到有效的 .docx 文件")

        logger.info(f"[analyze] function={function}, files={len(saved_paths)}")

        vecs, reports, ranked, weights, quality_report = _run_pipeline_on_files(
            saved_paths, function
        )

        if vecs is None:
            raise HTTPException(status_code=422, detail="所有文件解析失败，请检查文件格式")

        report_map = {}
        for i, vec in enumerate(vecs):
            if i < len(reports):
                report_map[vec.candidate_id] = reports[i]

        vec_map = {v.candidate_id: v for v in vecs}

        cards = []
        for cid, score, rank in ranked:
            vec    = vec_map.get(cid)
            report = report_map.get(cid)
            if vec and report:
                card = _build_candidate_card(vec, report, rank, score, vecs, weights)
                cards.append(card)

        # ── 三层评估体系（替换原 MD5 mock）────────────────────────
        pred_order = [c.candidate_id for c in cards]
        metrics, metrics_source = _compute_evaluation_metrics(pred_order, vecs, function)

        return AnalyzeResponse(
            job_id=str(uuid.uuid4())[:8],
            function=function,
            total=len(cards),
            run_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            candidates=cards,
            weights={k: round(v, 4) for k, v in sorted(
                weights.items(), key=lambda x: x[1], reverse=True)[:8]},
            quality={
                "avg_confidence":     quality_report.get("avg_confidence", 0),
                "distribution":       quality_report.get("quality_distribution", {}),
                "human_review_queue": quality_report.get("human_review_queue", []),
                "mnar":               quality_report.get("mnar_statistics", {}),
            },
            metrics=metrics,
            metrics_source=metrics_source,
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/api/hr_rating")
async def save_hr_rating(req: HRRatingRequest):
    """
    HR 为候选人打分，写入黄金标准文件。
    多次调用会合并评分（同一候选人取最近一次）。
    前端可在评估结果页提供"HR 评分"入口，由招聘负责人对本批次简历打分后提交。

    请求体示例：
    {
      "function": "人力资源",
      "ratings": {"LHQ": 4, "XJJ": 3, "LDD": 3, "DW": 2},
      "rater_name": "张招聘"
    }
    """
    # 校验分值范围
    invalid = {cid: v for cid, v in req.ratings.items()
               if not isinstance(v, int) or not (0 <= v <= 4)}
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"以下候选人分值超出 0-4 范围：{invalid}"
        )

    gold_path = os.path.join(GOLD_DIR, f"{req.function}.json")

    # 读取现有文件（若有）并合并
    existing: Dict[str, Any] = {}
    if os.path.exists(gold_path):
        try:
            with open(gold_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    existing.update(req.ratings)

    # 附加元数据（不影响评分读取）
    existing["_meta"] = {
        "last_updated": datetime.now().isoformat(),
        "last_rater":   req.rater_name or "anonymous",
        "total_rated":  len([k for k in existing if not k.startswith("_")]),
    }

    with open(gold_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    logger.info(
        f"[hr_rating] {req.function} 黄金标准更新：{len(req.ratings)} 条，"
        f"评分人={req.rater_name}，文件={gold_path}"
    )
    return {
        "status": "ok",
        "function": req.function,
        "rated_count": len(req.ratings),
        "total_in_file": existing["_meta"]["total_rated"],
        "gold_path": gold_path,
    }


@app.get("/api/gold_standard/{function}")
async def get_gold_standard(function: str):
    """
    查看指定职能的 HR 黄金标准评分情况。
    """
    gold = _load_hr_gold_standard(function)
    if gold is None:
        return {
            "function": function,
            "status": "no_gold_standard",
            "message": "该职能尚无 HR 黄金标准，当前使用信号代理金标准，"
                       "请通过 POST /api/hr_rating 提交 HR 评分。",
            "ratings": {},
        }
    gold_path = os.path.join(GOLD_DIR, f"{function}.json")
    meta: Dict = {}
    try:
        with open(gold_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        meta = raw.get("_meta", {})
    except Exception:
        pass
    return {
        "function": function,
        "status": "hr_gold",
        "ratings": gold,
        "meta": meta,
    }


@app.post("/api/feedback")
async def feedback(req: FeedbackRequest):
    """接收 HR 反馈，更新 Contextual Bandit"""
    if req.action not in REWARD_MAP:
        raise HTTPException(status_code=400,
                            detail=f"无效 action，支持: {list(REWARD_MAP.keys())}")

    vecs_map = _session_vecs.get(req.function, {})
    vec = vecs_map.get(req.candidate_id)

    if vec is None:
        return {"status": "skipped", "reason": "候选人不在当前会话中"}

    _bandit.receive_feedback(req.candidate_id, req.action, vec, req.function)

    return {
        "status": "ok",
        "candidate_id": req.candidate_id,
        "action": req.action,
        "reward": REWARD_MAP[req.action],
        "kl_div": round(_bandit.agents[req.function].compute_kl_divergence(), 4),
    }


@app.get("/api/functions")
async def get_functions():
    """返回支持的职能列表"""
    return {
        "functions": [
            {"key": "品牌",    "dir": "branding",      "icon": "✦"},
            {"key": "电商",    "dir": "ecommerce",     "icon": "◈"},
            {"key": "人力资源", "dir": "hr",            "icon": "◉"},
            {"key": "生产",    "dir": "manufacturing", "icon": "◧"},
            {"key": "研发",    "dir": "rnd",           "icon": "◎"},
            {"key": "销售",    "dir": "sales",         "icon": "◆"},
        ]
    }


@app.get("/api/weights/{function}")
async def get_current_weights(function: str):
    """返回 Bandit 学习到的当前权重"""
    if function not in _bandit.agents:
        return {"weights": {}, "message": "该职能尚无反馈数据"}
    weights = _bandit.get_current_weights(function, FEATURE_ORDER)
    return {"function": function, "weights": weights}
