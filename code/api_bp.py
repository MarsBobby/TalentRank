#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
api.py — 人才优选系统 · FastAPI 后端服务
启动方式: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
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
from typing import List, Optional, Dict, Any
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

# ── 日志 ────────────────────────────────────────────────────
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# 生成日志文件名（区分后端和流水线）
log_file = os.path.join(LOG_DIR, f"backend_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"), # 确保写入文件
        logging.StreamHandler(sys.stdout)               # 同时在终端打印
    ],
    force=True # 关键：强制覆盖 FastAPI 可能存在的默认配置
)
logger = logging.getLogger(__name__)
logger.info(f"后端服务日志系统初始化成功，日志文件：{log_file}")

# ── 全局单例（跨请求保持 Bandit 状态） ────────────────────────
_docx_extractor  = DocxExtractor()
_extractor_agent = ExtractorAgent()
_weighter        = DynamicWeighter()
_ranker          = LambdaMARTRanker()
_bandit          = ContextualBanditSystem()
_shap_agent      = SHAPExplainabilityAgent(weights={})

# 内存中保存最近一次各职能的向量（供 Bandit 反馈使用）
_session_vecs: Dict[str, Any] = {}

# 职能→数据目录映射
FUNCTION_DIR_MAP = {
    "品牌":     "branding",
    "电商":     "ecommerce",
    "人力资源": "hr",
    "生产":     "manufacturing",
    "研发":     "rnd",
    "销售":     "sales",
}

# 雷达图6维度归一化配置 (feature_key, display_name, max_val)
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
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件（前端 HTML/CSS/JS）
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ══════════════════════════════════════════════════════════════
#  Pydantic 数据模型
# ══════════════════════════════════════════════════════════════

class FeedbackRequest(BaseModel):
    candidate_id: str
    action:       str   # click / bookmark / invite / reject
    function:     str


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
    radar:             List[Dict]        # [{label, value, max}, ...]
    shap_positive:     List[Dict]        # [{feature, cn_name, value}, ...]
    shap_negative:     List[Dict]
    achievement_signal: int
    major_projects_count: int
    warnings:          List[str]


class AnalyzeResponse(BaseModel):
    job_id:     str
    function:   str
    total:      int
    run_time:   str
    candidates: List[CandidateCard]
    weights:    Dict[str, float]
    quality:    Dict
    metrics:    Dict


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


def _mock_ndcg_gold_relevance(candidate_ids: List[str]) -> Dict[str, float]:
    """演示用：由 candidate_id 确定性生成 NDCG 金标准相关性 (0-4)，与模型排序解耦。"""
    gold: Dict[str, float] = {}
    for cid in candidate_ids:
        digest = hashlib.md5(cid.encode("utf-8")).hexdigest()
        gold[cid] = float(int(digest, 16) % 5)
    return gold


def _mock_mrr_relevant_ids(candidate_ids: List[str]) -> List[str]:
    """演示用：由 candidate_id 确定性挑选 MRR 正例集合，与模型排序解耦。"""
    if not candidate_ids:
        return []
    scored = [
        (cid, int(hashlib.md5(cid.encode("utf-8")).hexdigest(), 16))
        for cid in candidate_ids
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    n = max(1, len(candidate_ids) // 3)
    return [cid for cid, _ in scored[:n]]


def _run_pipeline_on_files(files_paths: List[str], function: str):
    """核心流水线，返回结构化结果"""
    vecs, reports = [], []

    # Phase 1 & 2
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

    # Phase 3
    quality_report = generate_quality_report(vecs)

    # Phase 4
    weights = _weighter.compute_weights(vecs, function=function)

    # Phase 5
    ranked = _ranker.rank(vecs)

    # Phase 6 — 初始化 Bandit 先验权重
    _bandit.initialize_from_weights(function, weights, FEATURE_ORDER)

    # Phase 7 — SHAP
    global _shap_agent
    _shap_agent = SHAPExplainabilityAgent(model=_ranker.model, weights=weights)

    # 存入 session
    _session_vecs[function] = {v.candidate_id: v for v in vecs}

    return vecs, reports, ranked, weights, quality_report


def _build_candidate_card(vec, report, rank: int, score: float,
                           all_vecs, weights) -> CandidateCard:
    """将特征向量转换为前端卡片格式"""
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
    """
    上传简历文件并运行 Phase 0-8 全链路分析
    """
    if not files:
        raise HTTPException(status_code=400, detail="请至少上传一份简历文件")

    # 保存上传文件到临时目录
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

        # 构建候选人报告字典 (candidate_id → report)
        report_map = {}
        for i, vec in enumerate(vecs):
            if i < len(reports):
                report_map[vec.candidate_id] = reports[i]

        vec_map = {v.candidate_id: v for v in vecs}

        # 构建候选人卡片
        cards = []
        for cid, score, rank in ranked:
            vec    = vec_map.get(cid)
            report = report_map.get(cid)
            if vec and report:
                card = _build_candidate_card(vec, report, rank, score, vecs, weights)
                cards.append(card)

        # 简化评估指标（演示模式：Mock 金标准 / relevant，与模型输出解耦）
        from modules.phase8_evaluation import ndcg_at_k, mean_reciprocal_rank
        pred_order = [c.candidate_id for c in cards]
        all_ids = [v.candidate_id for v in vecs]
        gold = _mock_ndcg_gold_relevance(all_ids)
        mock_relevant = _mock_mrr_relevant_ids(all_ids)
        metrics = {
            "NDCG@5":  ndcg_at_k(pred_order, gold, 5),
            "NDCG@10": ndcg_at_k(pred_order, gold, 10),
            "MRR":     mean_reciprocal_rank(pred_order, mock_relevant),
        }

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
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/api/feedback")
async def feedback(req: FeedbackRequest):
    """
    接收 HR 反馈，更新 Contextual Bandit
    """
    if req.action not in REWARD_MAP:
        raise HTTPException(status_code=400,
                            detail=f"无效 action，支持: {list(REWARD_MAP.keys())}")

    vecs_map = _session_vecs.get(req.function, {})
    vec = vecs_map.get(req.candidate_id)

    if vec is None:
        # 候选人不在当前会话中，仍记录但不更新
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
            {"key": "品牌",   "dir": "branding",     "icon": "✦"},
            {"key": "电商",   "dir": "ecommerce",    "icon": "◈"},
            {"key": "人力资源","dir": "hr",           "icon": "◉"},
            {"key": "生产",   "dir": "manufacturing","icon": "◧"},
            {"key": "研发",   "dir": "rnd",          "icon": "◎"},
            {"key": "销售",   "dir": "sales",        "icon": "◆"},
        ]
    }


@app.get("/api/weights/{function}")
async def get_current_weights(function: str):
    """返回 Bandit 学习到的当前权重"""
    if function not in _bandit.agents:
        return {"weights": {}, "message": "该职能尚无反馈数据"}
    weights = _bandit.get_current_weights(function, FEATURE_ORDER)
    return {"function": function, "weights": weights}