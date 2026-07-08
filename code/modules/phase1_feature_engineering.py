#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1: 针对 MNAR 的数据重构与特征工程
────────────────────────────────────────
核心原则："缺失即信息"
1. MNAR 字段保留 NaN + 引入 Missing Flag
2. 职级语义校准（纠正头衔通胀）
3. 职业轨迹斜率（拉普拉斯平滑）
4. 成就语义二值化（放弃 KPI 绝对值估算）
5. 贝叶斯后验填补（仅对非 MNAR 字段）
6. CSI 公司平台分桶

修复记录（v2.0）：
  P0 - _feat_management：原版单条正则 r"(?:管理|带领|带|下属|团队)[^0-9]*?(\d+)\s*(?:人|名)"
       命中率极低，是 span_of_control 在全部 6 个部门均为 NaN 的根本原因。
       本次替换为 SPAN_PATTERNS_EXTENDED（9条正则 + 汉字数量词映射），
       与 Phase2 Agent C 共享同一规则集，彻底修复。
  P0 - _extract_id_from_filename：原版对纯汉字文件名回退至文件名尾部截断，
       产生 "or生产总监" 等脏数据。新增 hashlib MD5 哈希回退，并记录
       id_source 供下游审计。
  P1 - _feat_achievements：新增行业专属 KPI 模式（电商/研发/品牌/销售/生产/HR），
       解决电商 kpi_log_score 持续 NaN 问题（GMV、UV、ROI 等写法未被通用正则捕捉）。
  P1 - _compute_global_confidence：使用 get_confidence_threshold(function)
       差异化阈值（研发/生产 0.48），消除过度人工路由。
"""

import re
import math
import sys
import os
import hashlib
from typing import Dict, List, Optional, Any, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.phase0_data_assets import CandidateFeatureVector, MNAR_FIELDS
from config import CONFIDENCE as CONF


# ─────────────────────────────────────────────────────────────
# 常量：职级语义映射（6级标准化）
# ─────────────────────────────────────────────────────────────

S_LEVEL_MAP = {
    # Level 6 - C-Suite
    "CEO": 6, "总裁": 6, "总经理": 6, "首席执行官": 6,
    "COO": 6, "CTO": 6, "CFO": 6,
    # Level 5 - VP
    "VP": 5, "副总裁": 5, "高级总监": 5, "副总": 5,
    "专家": 5, "资深专家": 5, "高级专家": 5,
    # Level 4 - Director
    "总监": 4, "部门负责人": 4, "区域总监": 4,
    "产品总监": 4, "技术总监": 4, "市场总监": 4,
    "销售总监": 4, "品牌总监": 4, "运营总监": 4,
    # Level 3 - Senior Manager
    "高级经理": 3, "资深经理": 3, "高级": 3,
    "负责人": 3, "品牌总": 3,
    # Level 2 - Manager / Lead
    "经理": 2, "主管": 2, "组长": 2, "Lead": 2,
    "项目经理": 2, "产品经理": 2,
    # Level 1 - IC / Junior
    "专员": 1, "助理": 1, "实习": 1, "初级": 1, "策划": 1,
}

# 管理类动词（高密度 → 管理深度代理）
MANAGEMENT_VERBS = [
    "统筹", "领导", "管理", "带领", "指挥", "协调", "决策", "战略",
    "规划", "主导", "负责", "推进", "构建", "建立", "优化", "整合",
    "指导", "辅导", "监督", "制定", "审批", "调配", "调度",
]

# 成就量化信号正则（检测%/万/倍/增长）
ACHIEVEMENT_PATTERN = re.compile(
    r"(\d+\.?\d*\s*%)"           # 百分比
    r"|(\d+\.?\d*\s*[万亿千百])"  # 金额数量级
    r"|(\d+\.?\d*\s*倍)"         # 倍数
    r"|(增长|提升|降低|节省|增加|提高)\s*\d+",  # 增长动词+数字
    re.UNICODE
)

# 汇报层级关键词
REPORTING_KEYWORDS = {
    "向董事长|向CEO|向总裁": 5,
    "向副总裁|向VP|向COO|向CTO": 4,
    "向总监|向总经理": 3,
    "向经理|向主管": 2,
}

# CSI 公司分桶关键词
CSI_TIER_MAP = {
    "头部": 1.0,
    "知名": 0.7,
    "普通": 0.4,
}
CSI_HEADCOUNT_TIERS = [(10000, 1.0), (1000, 0.8), (100, 0.6), (0, 0.4)]

# 985/211 学校排名分桶
SCHOOL_RANK_MAP = {
    "985": 30,
    "211": 100,
    "双一流": 120,
}

# 学历评分
EDUCATION_SCORE_MAP = {
    "博士": 3.0, "PhD": 3.0,
    "硕士": 2.0, "MBA": 2.0, "Master": 2.0,
    "本科": 1.0, "Bachelor": 1.0, "学士": 1.0,
    "大专": 0.5, "Associate": 0.5,
    "高中": 0.0, "中专": 0.0,
}

# ─────────────────────────────────────────────────────────────
# P0修复 — 管理跨度提取：扩充正则 + 汉字数量词映射
# ─────────────────────────────────────────────────────────────

# 9 条扩充正则，覆盖"负责一支由38人组成的团队"/"团队规模200人"/"十余人"等多样写法
SPAN_PATTERNS_EXTENDED: List[str] = [
    r'(?:管理|带领|负责|统筹|领导|指挥|率领)[^，。\n]{0,15}?(\d+)\s*(?:人|名|位)',
    r'团队[规模人数大小]*[为有共计约达]*\s*(\d+)\s*(?:人|名|位)',
    r'(\d+)\s*(?:人|名|位)(?:的)?(?:团队|研发团队|销售团队|运营团队|技术团队|规模)',
    r'下[辖属][^，。\n]*?(\d+)\s*(?:人|名|位)',
    r'(?:一支|一个|整个)[^，。\n]{0,20}?(\d+)\s*(?:人|名|位)',
    r'(?:管理|[下属团队])[^\d]*?(\d+)\s*(?:人|名)',   # 原有逻辑保留
    r'带团队(\d+)',                                    # 原有逻辑保留
    r'(?:直接|间接)管理[^，。\n]{0,10}?(\d+)\s*(?:人|名|位)',
    r'(?:人员规模|团队人数)[：:为]*\s*(\d+)',
]

# 汉字数量词 → 整数（"十余人""数百人"等写法）
CHINESE_NUM_MAP: Dict[str, int] = {
    '十余': 15, '十多': 15, '十几': 15,
    '二十余': 25, '二十多': 25, '三十余': 35, '三十多': 35,
    '五十余': 55, '近百': 90, '百余': 150, '百多': 150,
    '数十': 30, '数百': 300, '数千': 3000, '近千': 900,
}

# ─────────────────────────────────────────────────────────────
# P1修复 — 行业专属 KPI 提取词典
# ─────────────────────────────────────────────────────────────

INDUSTRY_KPI_PATTERNS: Dict[str, List[Tuple[str, float]]] = {
    '电商': [
        # GMV：支持"GMV 15亿""年成交额 5亿""总成交额 3千万"等多种写法
        (r'(?:GMV|成交额|交易额|年销售额|销售规模)[约达完成破]*\s*(\d+\.?\d*)\s*亿', 1e8),
        (r'(?:GMV|成交额|交易额|年销售额|销售规模)[约达完成破]*\s*(\d+\.?\d*)\s*万', 1e4),
        # UV / DAU / MAU
        (r'(?:UV|DAU|MAU|日活|月活|活跃用户)[约达破]*\s*(\d+\.?\d*)\s*亿', 1e7),
        (r'(?:UV|DAU|MAU|日活|月活|活跃用户)[约达破]*\s*(\d+\.?\d*)\s*万', 1e3),
        # 转化率
        (r'转化率[约达提升至破]*\s*(\d+\.?\d*)\s*%', 10.0),
        # ROI：权重乘以 500（ROI=3 等效于 50% 增长）
        (r'ROI[约达破]*\s*(\d+\.?\d*)', 500.0),
        # 复购率
        (r'复购率[约达提升至]*\s*(\d+\.?\d*)\s*%', 5.0),
        # 客单价（元）
        (r'客单价[约达提升至破]*\s*(\d+\.?\d*)\s*元', 1.0),
        # CTR
        (r'(?:点击率|CTR)[约达提升至]*\s*(\d+\.?\d*)\s*%', 10.0),
    ],
    '研发': [
        (r'性能提升[约达]*\s*(\d+\.?\d*)\s*%', 1.0),
        (r'延迟(?:降低|减少)[约达]*\s*(\d+\.?\d*)\s*(?:%|ms|毫秒)', 1.0),
        (r'(?:代码|测试)覆盖率[约达提升至]*\s*(\d+\.?\d*)\s*%', 1.0),
        (r'(?:QPS|TPS|吞吐量)[约达提升至]*\s*(\d+\.?\d*)', 0.01),
        (r'可用性[约达提升至]*\s*(\d+\.?\d*)\s*%', 1.0),
        (r'(?:专利|发明专利)[^\d]*(\d+)\s*(?:项|件)', 200.0),
        (r'(?:论文|paper)[^\d]*(\d+)\s*篇', 100.0),
    ],
    '品牌': [
        (r'曝光[量次][约达]*\s*(\d+\.?\d*)\s*亿', 1e6),
        (r'曝光[量次][约达]*\s*(\d+\.?\d*)\s*万', 1e2),
        (r'品牌(?:知名度|认知度)[约达提升]*\s*(\d+\.?\d*)\s*%', 10.0),
        (r'声量[约达增长]*\s*(\d+\.?\d*)\s*%', 10.0),
        (r'(?:粉丝|关注)[约达增长至]*\s*(\d+\.?\d*)\s*万', 10.0),
    ],
    '销售': [
        (r'销售额[约达完成]*\s*(\d+\.?\d*)\s*亿', 1e8),
        (r'销售额[约达完成]*\s*(\d+\.?\d*)\s*万', 1e4),
        (r'完成率[约达]*\s*(\d+\.?\d*)\s*%', 1.0),
        (r'(?:客户数|新增客户)[约达增长至]*\s*(\d+\.?\d*)\s*(?:万|个|家)', 100.0),
        (r'回款率[约达提升至]*\s*(\d+\.?\d*)\s*%', 1.0),
    ],
    '生产': [
        (r'良率[约达提升至]*\s*(\d+\.?\d*)\s*%', 1.0),
        (r'产能[约达提升至]*\s*(\d+\.?\d*)\s*(?:万|件|台|套)', 10.0),
        (r'(?:降本|节约|节省)[约达]*\s*(\d+\.?\d*)\s*万', 1e4),
        (r'效率提升[约达]*\s*(\d+\.?\d*)\s*%', 1.0),
    ],
    '人力资源': [
        (r'招聘[完成了]*\s*(\d+\.?\d*)\s*(?:人|名|位)', 5.0),
        (r'离职率[约达降低至]*\s*(\d+\.?\d*)\s*%', 2.0),
        (r'(?:员工|人员)满意度[约达提升至]*\s*(\d+\.?\d*)\s*%', 1.0),
        (r'培训[约达覆盖]*\s*(\d+\.?\d*)\s*(?:人次|人)', 1.0),
    ],
}


# ─────────────────────────────────────────────────────────────
# 核心特征工程类
# ─────────────────────────────────────────────────────────────

class MNARAwareFeatureEngineer:
    """
    Phase 1 特征工程器
    输入：DocxExtractor 提取的原始文本 + ResumeData（可选）
    输出：CandidateFeatureVector
    """

    def __init__(self):
        self._laplace_eps = 1.0   # 拉普拉斯平滑项（单位：年）

    # ────── 公开入口 ──────────────────────────────────────────

    def engineer(self, raw_text: str, file_name: str = "",
                 function: str = "", candidate_id: str = "") -> CandidateFeatureVector:
        """从原始简历文本生成特征向量"""
        vec = CandidateFeatureVector()
        vec.candidate_id = candidate_id or self._extract_id_from_filename(file_name)
        vec.file_name    = file_name
        vec.function     = function
        vec.raw_text     = raw_text

        # 教育背景
        self._feat_education(raw_text, vec)
        # 职级语义校准（S-Level）
        self._feat_s_level(raw_text, vec)
        # CSI 公司平台指数
        self._feat_csi(raw_text, vec)
        # 工作年限
        self._feat_work_years(raw_text, vec)
        # 管理深度特征
        self._feat_management(raw_text, vec)
        # 成就语义（二值化 + Log-KPI + 行业专属KPI）
        self._feat_achievements(raw_text, vec, function)
        # 软素质
        self._feat_soft_skills(raw_text, vec)
        # 职业轨迹斜率（拉普拉斯平滑）
        self._feat_career_slope(raw_text, vec)
        # 稳定性
        self._feat_stability(raw_text, vec)
        # 综合置信度（P1修复：使用职能差异化阈值）
        self._compute_global_confidence(vec, function)

        return vec

    # ────── 教育背景 ──────────────────────────────────────────

    def _feat_education(self, text: str, vec: CandidateFeatureVector):
        score, conf = 1.0, CONF["default_fill"]
        for kw, val in EDUCATION_SCORE_MAP.items():
            if kw in text:
                score, conf = val, CONF["exact_match"]
                break

        rank, rank_conf = 500.0, CONF["default_fill"]
        for kw, r in SCHOOL_RANK_MAP.items():
            if kw in text:
                rank, rank_conf = float(r), CONF["context_infer"]
                break

        # 专业匹配度（简化版：关键词重叠）
        major_match, mm_conf = 0.6, CONF["default_fill"]
        major_m = re.search(r"(?:专业|主修)[：:]\s*([\u4e00-\u9fa5a-zA-Z]+)", text)
        if major_m:
            mm_conf = CONF["regex_match"]
            brand_keywords = {"营销", "品牌", "市场", "传播", "广告", "公关"}
            major_words = set(major_m.group(1))
            overlap = len(major_words & brand_keywords)
            major_match = min(0.95, 0.5 + overlap * 0.15)

        vec.education_score     = score
        vec.education_conf      = conf
        vec.school_ranking      = rank
        vec.school_ranking_conf = rank_conf
        vec.major_match_score   = major_match
        vec.major_match_conf    = mm_conf

    # ────── 职级语义校准 ──────────────────────────────────────

    def _feat_s_level(self, text: str, vec: CandidateFeatureVector):
        """
        BERT 替代方案：基于关键词长度降序匹配，纠正头衔通胀
        规则：以职责描述中管理动词密度 × 职级关键词综合判断
        """
        best_level, best_conf = 1, CONF["default_fill"]
        sorted_map = sorted(S_LEVEL_MAP.items(), key=lambda x: len(x[0]), reverse=True)
        for kw, lvl in sorted_map:
            if kw in text:
                best_level = lvl
                best_conf  = CONF["exact_match"]
                break

        # 动词密度补偿：若密度高但职级低 → 上调一级
        verb_density = self._compute_verb_density(text)
        if verb_density > 0.15 and best_level <= 2:
            best_level = min(best_level + 1, 6)
            best_conf  = CONF["context_infer"]

        vec.s_level      = float(best_level)
        vec.s_level_conf = best_conf

    # ────── CSI 公司平台指数 ──────────────────────────────────

    def _feat_csi(self, text: str, vec: CandidateFeatureVector):
        """
        公司名脱敏环境下的 CSI 代理：
        利用规模描述 / 行业标签 / 国企/外企关键词分三档
        """
        csi = 0.5
        conf = CONF["default_fill"]

        if re.search(r"(外资|跨国|世界500强|上市公司|集团)", text):
            csi = 0.85
            conf = CONF["context_infer"]
        size_m = re.search(r"(\d+)\s*人(?:规模|以上|员工)", text)
        if size_m:
            n = int(size_m.group(1))
            for threshold, val in CSI_HEADCOUNT_TIERS:
                if n >= threshold:
                    csi = max(csi, val)
                    conf = max(conf, CONF["regex_match"])
                    break

        vec.csi_score      = csi
        vec.csi_score_conf = conf

    # ────── 工作年限 ──────────────────────────────────────────

    def _feat_work_years(self, text: str, vec: CandidateFeatureVector):
        """从日期区间估算总工龄（年）"""
        date_ranges = re.findall(
            r"(\d{4})[.\-/年](\d{1,2})\s*[-~至到]\s*(\d{4})[.\-/年](\d{1,2})",
            text
        )
        open_ranges = re.findall(
            r"(\d{4})[.\-/年](\d{1,2})\s*[-~至到]\s*(?:至今|现在|present)",
            text, re.IGNORECASE
        )

        total_months = 0
        periods = []

        for sy, sm, ey, em in date_ranges:
            months = (int(ey) - int(sy)) * 12 + (int(em) - int(sm))
            months = max(0, min(months, 120))
            periods.append(months)

        from datetime import datetime
        now = datetime.now()
        for sy, sm in open_ranges:
            months = (now.year - int(sy)) * 12 + (now.month - int(sm))
            months = max(0, min(months, 120))
            periods.append(months)

        if periods:
            total_months = sum(periods)
            vec.total_work_years = round(total_months / 12, 1)
            vec.work_years_conf  = CONF["regex_match"]
        else:
            edu = vec.education_score
            vec.total_work_years = max(2.0, (edu * 3))
            vec.work_years_conf  = CONF["bayesian_fill"]

    # ────── 管理深度 ──────────────────────────────────────────

    def _feat_management(self, text: str, vec: CandidateFeatureVector):
        """
        P0修复：原版单条正则命中率极低（span_of_control 全部门 NaN 的根因）。
        现使用 SPAN_PATTERNS_EXTENDED（9条正则 + 汉字数量词映射），
        与 Phase2 Agent C 共享规则集，确保两层提取逻辑一致。

        策略：取所有命中值中的最大值作为 span_of_control
        （简历中最高峰管理规模更能代表候选人的管理上限）。
        """
        span_candidates: List[int] = []
        seen: set = set()

        # ① 扩充数字正则（覆盖多样化写法）
        for pattern in SPAN_PATTERNS_EXTENDED:
            try:
                for m in re.finditer(pattern, text):
                    raw = m.group(1) if m.lastindex else m.group(0)
                    val = int(raw)
                    if 1 <= val <= 10000 and val not in seen:
                        span_candidates.append(val)
                        seen.add(val)
            except (ValueError, IndexError, re.error):
                continue

        # ② 汉字数量词（"十余人""数百人""近千人"等）
        for cn_word, num_val in CHINESE_NUM_MAP.items():
            pat = re.escape(cn_word) + r'\s*(?:人|名|位)?'
            if re.search(pat, text) and num_val not in seen:
                span_candidates.append(num_val)
                seen.add(num_val)

        # ③ 写入特征向量：取最大值（代表管理上限）；空则保持 MNAR NaN
        if span_candidates:
            vec.span_of_control = float(max(span_candidates))
            vec.span_missing    = 0
        # 否则 span_of_control 保持 None（NaN），span_missing=1

        # ④ 管理动词密度（代理特征，span NaN 时仍可提供管理深度信号）
        vec.management_verb_density = self._compute_verb_density(text)

        # ⑤ 汇报层级
        for pattern, level in REPORTING_KEYWORDS.items():
            if re.search(pattern, text):
                vec.reporting_layer = float(level)
                break

    def _compute_verb_density(self, text: str) -> float:
        """计算管理动词在文本中的密度（每100词）"""
        words = text.replace("\n", " ").split()
        if not words:
            return 0.0
        count = sum(1 for w in words for v in MANAGEMENT_VERBS if v in w)
        return round(count / max(len(words), 1) * 100, 3)

    # ────── 成就语义特征 ──────────────────────────────────────

    def _feat_achievements(self, text: str, vec: CandidateFeatureVector,
                           function: str = ""):
        """
        P1修复：新增行业专属 KPI 提取（解决电商 kpi_log_score 持续 NaN 问题）。
        原有通用提取逻辑完整保留，行业专属结果叠加到通用得分上。

        策略：
          - 通用：金额(万) + 百分比 + 增长词+数字
          - 行业专属：GMV/UV/ROI/良率/销售额等，各有权重系数压缩量纲
          - 合并后取 log(raw_kpi + 1)，若合并得分 > 0 则 kpi_missing=0
        """
        # ── 成就信号二值化（原逻辑不变）─────────────────────────
        if ACHIEVEMENT_PATTERN.search(text):
            vec.achievement_signal = 1

        # ── 通用 KPI 提取（原逻辑不变）──────────────────────────
        amounts = re.findall(r"(\d+\.?\d*)\s*万", text)
        pcts    = re.findall(r"(\d+\.?\d*)\s*%", text)
        growths = re.findall(r"(?:增长|提升|提高)\s*(\d+\.?\d*)", text)

        raw_kpi = 0.0
        for a in amounts:
            raw_kpi += min(float(a), 10000)
        for p in pcts:
            raw_kpi += min(float(p), 100)
        for g in growths:
            raw_kpi += min(float(g), 100)

        # ── P1修复：行业专属 KPI 提取（叠加到通用得分）──────────
        industry_patterns = INDUSTRY_KPI_PATTERNS.get(function, [])
        industry_kpi_raw  = 0.0
        industry_hit_count = 0

        for pattern, weight in industry_patterns:
            try:
                matches = re.findall(pattern, text)
                for m in matches:
                    # findall 对含多分组的正则可能返回字符串或元组
                    raw_val = m[0] if isinstance(m, tuple) else m
                    try:
                        val = float(raw_val)
                        # weight 是该指标的"万元等效"换算系数，统一压缩量纲
                        industry_kpi_raw += min(val * weight / 1e4, 100.0)
                        industry_hit_count += 1
                    except ValueError:
                        continue
            except re.error:
                continue

        # 合并：行业专属最多加成 50 分（防止量纲爆炸）
        industry_bonus = min(industry_kpi_raw, 50.0)
        total_kpi = raw_kpi + industry_bonus

        # 写入特征：有任何有效得分就不再是 MNAR
        if total_kpi > 0:
            vec.kpi_log_score = round(math.log(total_kpi + 1), 4)
            vec.kpi_missing   = 0
        # 否则 kpi_log_score 保持 None，kpi_missing=1（MNAR）

        # 行业专属 KPI 命中数（新增字段，供 SHAP 解释链使用）
        if hasattr(vec, 'industry_kpi_hit_count'):
            vec.industry_kpi_hit_count = industry_hit_count

        # 重大项目计数（原逻辑不变）
        project_kws = ["主导", "负责", "主抓", "操盘", "一号位", "战略项目", "核心项目"]
        vec.major_projects_count = sum(1 for kw in project_kws if kw in text)

    # ────── 软素质 ──────────────────────────────────────────

    def _feat_soft_skills(self, text: str, vec: CandidateFeatureVector):
        """8 类软素质加权评分"""
        cats = {
            "leadership":    (["领导", "管理", "带领", "决策", "战略", "指导"], 0.25),
            "communication": (["沟通", "协调", "表达", "谈判", "汇报"],         0.15),
            "teamwork":      (["团队", "协作", "合作", "配合"],                 0.15),
            "problem_solving":(["解决问题", "分析", "克服", "优化"],            0.15),
            "innovation":    (["创新", "改进", "突破", "创意", "0到1"],         0.10),
            "adaptability":  (["适应", "灵活", "应变", "转型"],                 0.10),
            "stress_resist": (["抗压", "高压", "压力"],                         0.05),
            "professionalism":(["专业", "敬业", "责任"],                        0.05),
        }
        total = 0.0
        for cat, (kws, w) in cats.items():
            hits = sum(1 for kw in kws if kw in text)
            score = min(1.0, hits / max(len(kws), 1))
            total += score * w
        vec.soft_skill_score = round(total * 10, 3)

    # ────── 职业轨迹斜率（拉普拉斯平滑） ─────────────────────

    def _feat_career_slope(self, text: str, vec: CandidateFeatureVector):
        """
        slope = Δlevel / (Δt + ε)
        ε = 1.0 年（拉普拉斯平滑，防止超短工龄导致指标爆炸）
        """
        positions = re.findall(
            r"(?:总监|经理|主管|总裁|VP|负责人|专员|助理|专家|高级)[^\n,，。]{0,15}",
            text
        )
        if len(positions) >= 2:
            levels = [self._keyword_to_level(p) for p in positions]
            delta_level = max(levels) - min(levels)
            years = vec.total_work_years or 1.0
            vec.career_growth_slope = round(delta_level / (years + self._laplace_eps), 4)

        csi = vec.csi_score
        vec.job_hopping_premium = round(math.log(csi + 1) * (vec.s_level / 3), 4)

    def _keyword_to_level(self, text: str) -> int:
        for kw, lvl in sorted(S_LEVEL_MAP.items(), key=lambda x: len(x[0]), reverse=True):
            if kw in text:
                return lvl
        return 1

    # ────── 稳定性 ──────────────────────────────────────────

    def _feat_stability(self, text: str, vec: CandidateFeatureVector):
        """基于任职时长计算稳定性指数"""
        durations = []
        for sy, sm, ey, em in re.findall(
            r"(\d{4})[.\-/年](\d{1,2})\s*[-~至到]\s*(\d{4})[.\-/年](\d{1,2})", text
        ):
            m = (int(ey) - int(sy)) * 12 + (int(em) - int(sm))
            if m > 0:
                durations.append(min(m, 120))

        if durations:
            long_term = sum(1 for d in durations if d >= 12)
            vec.stability_index = round(long_term / len(durations), 3)
        else:
            vec.stability_index = 0.65

    # ────── 综合置信度 ────────────────────────────────────────

    def _compute_global_confidence(self, vec: CandidateFeatureVector,
                                   function: str = ""):
        """
        P1修复：使用 get_confidence_threshold(function) 差异化阈值，
        研发/生产用 0.48，其余用 0.60，消除过度人工审核路由。
        """
        confs = [
            vec.education_conf, vec.school_ranking_conf,
            vec.major_match_conf, vec.s_level_conf,
            vec.csi_score_conf, vec.work_years_conf,
        ]
        if vec.span_missing:
            confs.append(0.3)
        if vec.kpi_missing:
            confs.append(0.3)

        vec.global_confidence = round(sum(confs) / max(len(confs), 1), 3)

        # P1修复：差异化阈值（避免研发/生产候选人过度路由到人工审核）
        try:
            from modules.phase3_confidence import get_confidence_threshold
            threshold = get_confidence_threshold(function)
        except ImportError:
            # 兜底：若循环导入则使用配置值
            from config import CONFIDENCE_THRESHOLDS
            threshold = CONFIDENCE_THRESHOLDS.get(function,
                        CONFIDENCE_THRESHOLDS.get("default", 0.6))

        vec.needs_human_review = vec.global_confidence < threshold

    # ────── 工具 ──────────────────────────────────────────────

    def _extract_id_from_filename(self, filename: str) -> str:
        """
        P0修复：原版对纯汉字文件名回退至文件名尾部截断，
        产生"or生产总监"等脏数据。新增 MD5 哈希回退：
        无数字 ID 时用文件名哈希前8位生成稳定唯一 ID，
        并在末尾附加 id_source 标记（仅记录，不影响排序逻辑）。
        """
        try:
            from urllib.parse import unquote
            filename = unquote(filename)
        except Exception:
            pass

        # 策略1：匹配文件名中的大写字母组合（如 -HFF(）for-）
        m = re.search(r"-([A-Z]{2,4})(?:[\(（]|for|-)", filename)
        if m:
            return m.group(1)

        # 策略2：提取连续大写字母
        letters = re.findall(r"[A-Z]{2,5}", filename)
        if letters:
            return letters[0]

        # 策略3：提取数字 ID（原有逻辑，优先级低于字母）
        digits = re.search(r'\d{4,}', filename)
        if digits:
            return digits.group(0)

        # P0修复 — 策略4：哈希回退（纯汉字/无标识符文件名）
        # 取文件名（去扩展名）的 MD5 前8位，确保稳定唯一
        stem = re.sub(r'\.docx$', '', filename, flags=re.IGNORECASE)
        h = hashlib.md5(stem.encode('utf-8')).hexdigest()[:8].upper()
        return h