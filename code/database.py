#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
database.py — PostgreSQL 数据层
────────────────────────────────
使用 SQLAlchemy Core（同步，兼容 FastAPI 线程池）
表结构：
  analysis_sessions  — 每次分析会话
  candidate_features — 候选人特征向量
  hr_feedbacks       — HR 操作记录
  bandit_weights     — Bandit 学习权重快照
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

from sqlalchemy import (
    create_engine, text,
    Column, String, Float, Integer, Boolean, DateTime, Text, JSON,
    Index, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
import uuid

logger = logging.getLogger(__name__)

# ── 连接配置 ──────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://talent:talent123@localhost:5432/talentdb"
)

engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,       # 自动检测断连
    pool_recycle=3600,        # 每小时回收连接
    echo=False,               # 生产环境关闭 SQL 日志
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


# ══════════════════════════════════════════════════════════════
#  ORM 模型
# ══════════════════════════════════════════════════════════════

class AnalysisSession(Base):
    """每次分析会话"""
    __tablename__ = "analysis_sessions"

    id          = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id      = Column(String(8), nullable=False, index=True)
    function    = Column(String(20), nullable=False)          # 职能类别
    total       = Column(Integer, nullable=False, default=0)
    run_time    = Column(DateTime, default=datetime.utcnow)
    ndcg5       = Column(Float, nullable=True)
    ndcg10      = Column(Float, nullable=True)
    mrr         = Column(Float, nullable=True)
    avg_confidence = Column(Float, nullable=True)
    weights_json   = Column(Text, nullable=True)             # JSON 序列化的权重
    quality_json   = Column(Text, nullable=True)             # JSON 质量报告
    created_at  = Column(DateTime, default=datetime.utcnow)


class CandidateFeature(Base):
    """候选人特征向量（持久化替代内存 _session_vecs）"""
    __tablename__ = "candidate_features"

    id             = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id     = Column(String(36), ForeignKey("analysis_sessions.id"), nullable=False, index=True)
    candidate_id   = Column(String(64), nullable=False, index=True)
    function       = Column(String(20), nullable=False)
    file_name      = Column(String(256), nullable=True)

    # 核心特征（用于 Bandit 更新）
    education_score        = Column(Float, nullable=True)
    s_level                = Column(Float, nullable=True)
    csi_score              = Column(Float, nullable=True)
    total_work_years       = Column(Float, nullable=True)
    career_growth_slope    = Column(Float, nullable=True)
    job_hopping_premium    = Column(Float, nullable=True)
    span_of_control        = Column(Float, nullable=True)
    span_missing           = Column(Integer, default=1)
    management_verb_density= Column(Float, nullable=True)
    reporting_layer        = Column(Float, nullable=True)
    kpi_log_score          = Column(Float, nullable=True)
    kpi_missing            = Column(Integer, default=1)
    achievement_signal     = Column(Integer, default=0)
    major_projects_count   = Column(Integer, default=0)
    stability_index        = Column(Float, nullable=True)
    soft_skill_score       = Column(Float, nullable=True)
    global_confidence      = Column(Float, nullable=True)
    needs_human_review     = Column(Boolean, default=False)

    # 排名与得分
    rank    = Column(Integer, nullable=True)
    score   = Column(Float, nullable=True)

    # 完整特征 JSON（供 SHAP 复现）
    feature_json = Column(Text, nullable=True)
    radar_json   = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_cf_session_cid", "session_id", "candidate_id"),
        Index("ix_cf_function_ts", "function", "created_at"),
    )


class HRFeedback(Base):
    """HR 操作记录"""
    __tablename__ = "hr_feedbacks"

    id           = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id   = Column(String(36), ForeignKey("analysis_sessions.id"), nullable=True, index=True)
    candidate_id = Column(String(64), nullable=False, index=True)
    function     = Column(String(20), nullable=False)
    action       = Column(String(20), nullable=False)   # invite/bookmark/click/reject
    reward       = Column(Float, nullable=True)
    kl_div       = Column(Float, nullable=True)
    created_at   = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_hrf_candidate_fn", "candidate_id", "function"),
    )


class BanditWeight(Base):
    """Bandit 权重快照（每次 HR 反馈后记录）"""
    __tablename__ = "bandit_weights"

    id         = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    function   = Column(String(20), nullable=False, index=True)
    weights_json = Column(Text, nullable=False)          # JSON 序列化
    update_count = Column(Integer, default=0)
    kl_div     = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ══════════════════════════════════════════════════════════════
#  数据库初始化
# ══════════════════════════════════════════════════════════════

def init_db():
    """创建所有表（幂等，已存在则跳过）"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("PostgreSQL 表结构初始化完成")
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        raise


def get_db() -> Session:
    """FastAPI 依赖注入：获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ══════════════════════════════════════════════════════════════
#  数据访问层 DAL
# ══════════════════════════════════════════════════════════════

class SessionDAL:
    """分析会话 CRUD"""

    @staticmethod
    def create(db: Session, job_id: str, function: str, total: int,
               metrics: Dict, weights: Dict, quality: Dict) -> AnalysisSession:
        row = AnalysisSession(
            job_id=job_id,
            function=function,
            total=total,
            ndcg5=metrics.get("NDCG@5"),
            ndcg10=metrics.get("NDCG@10"),
            mrr=metrics.get("MRR"),
            avg_confidence=quality.get("avg_confidence"),
            weights_json=json.dumps(weights, ensure_ascii=False),
            quality_json=json.dumps(quality, ensure_ascii=False),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row

    @staticmethod
    def get_latest(db: Session, function: str) -> Optional[AnalysisSession]:
        return (db.query(AnalysisSession)
                .filter_by(function=function)
                .order_by(AnalysisSession.created_at.desc())
                .first())

    @staticmethod
    def list_recent(db: Session, limit: int = 20) -> List[AnalysisSession]:
        return (db.query(AnalysisSession)
                .order_by(AnalysisSession.created_at.desc())
                .limit(limit).all())


class CandidateDAL:
    """候选人特征 CRUD"""

    @staticmethod
    def upsert_batch(db: Session, session_id: str,
                     vecs, ranked_map: Dict[str, tuple],
                     radar_map: Dict[str, list]) -> None:
        """批量写入候选人特征（同一 session 内先删后插）"""
        db.query(CandidateFeature).filter_by(session_id=session_id).delete()
        for vec in vecs:
            cid = vec.candidate_id
            rank_info = ranked_map.get(cid, (None, None))
            rank, score = rank_info
            d = vec.to_feature_dict()
            row = CandidateFeature(
                session_id=session_id,
                candidate_id=cid,
                function=vec.function,
                file_name=vec.file_name,
                education_score=_safe_val(d.get("education_score")),
                s_level=_safe_val(d.get("s_level")),
                csi_score=_safe_val(d.get("csi_score")),
                total_work_years=_safe_val(d.get("total_work_years")),
                career_growth_slope=_safe_val(d.get("career_growth_slope")),
                job_hopping_premium=_safe_val(d.get("job_hopping_premium")),
                span_of_control=_safe_val(d.get("span_of_control")),
                span_missing=int(d.get("span_missing", 1)),
                management_verb_density=_safe_val(d.get("management_verb_density")),
                reporting_layer=_safe_val(d.get("reporting_layer")),
                kpi_log_score=_safe_val(d.get("kpi_log_score")),
                kpi_missing=int(d.get("kpi_missing", 1)),
                achievement_signal=int(d.get("achievement_signal", 0)),
                major_projects_count=int(d.get("major_projects_count", 0)),
                stability_index=_safe_val(d.get("stability_index")),
                soft_skill_score=_safe_val(d.get("soft_skill_score")),
                global_confidence=vec.global_confidence,
                needs_human_review=vec.needs_human_review,
                rank=rank,
                score=score,
                feature_json=json.dumps(d, ensure_ascii=False, default=str),
                radar_json=json.dumps(radar_map.get(cid, []), ensure_ascii=False),
            )
            db.add(row)
        db.commit()

    @staticmethod
    def get_by_session_and_id(db: Session, session_id: str,
                               candidate_id: str) -> Optional[CandidateFeature]:
        return (db.query(CandidateFeature)
                .filter_by(session_id=session_id, candidate_id=candidate_id)
                .first())

    @staticmethod
    def get_latest_by_function(db: Session, function: str,
                                candidate_id: str) -> Optional[CandidateFeature]:
        """获取该职能下最新一次分析的候选人记录"""
        return (db.query(CandidateFeature)
                .filter_by(function=function, candidate_id=candidate_id)
                .order_by(CandidateFeature.created_at.desc())
                .first())

    @staticmethod
    def get_by_function_latest_session(db: Session, function: str) -> List[CandidateFeature]:
        """获取最新 session 的所有候选人"""
        latest = (db.query(AnalysisSession)
                  .filter_by(function=function)
                  .order_by(AnalysisSession.created_at.desc())
                  .first())
        if not latest:
            return []
        return (db.query(CandidateFeature)
                .filter_by(session_id=latest.id)
                .order_by(CandidateFeature.rank)
                .all())


class FeedbackDAL:
    """HR 反馈 CRUD"""

    @staticmethod
    def create(db: Session, session_id: Optional[str], candidate_id: str,
               function: str, action: str, reward: float,
               kl_div: float) -> HRFeedback:
        row = HRFeedback(
            session_id=session_id,
            candidate_id=candidate_id,
            function=function,
            action=action,
            reward=reward,
            kl_div=kl_div,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row

    @staticmethod
    def get_by_candidate(db: Session, candidate_id: str,
                          function: str) -> List[HRFeedback]:
        return (db.query(HRFeedback)
                .filter_by(candidate_id=candidate_id, function=function)
                .order_by(HRFeedback.created_at.desc())
                .all())

    @staticmethod
    def get_session_feedbacks(db: Session, session_id: str) -> List[HRFeedback]:
        return (db.query(HRFeedback)
                .filter_by(session_id=session_id)
                .order_by(HRFeedback.created_at)
                .all())


class BanditDAL:
    """Bandit 权重快照"""

    @staticmethod
    def save_snapshot(db: Session, function: str, weights: Dict,
                      update_count: int, kl_div: float) -> BanditWeight:
        row = BanditWeight(
            function=function,
            weights_json=json.dumps(weights, ensure_ascii=False),
            update_count=update_count,
            kl_div=kl_div,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row

    @staticmethod
    def get_latest(db: Session, function: str) -> Optional[BanditWeight]:
        return (db.query(BanditWeight)
                .filter_by(function=function)
                .order_by(BanditWeight.created_at.desc())
                .first())


# ══════════════════════════════════════════════════════════════
#  工具
# ══════════════════════════════════════════════════════════════

def _safe_val(v):
    """NaN → None，方便写入数据库"""
    import math
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return float(v)
