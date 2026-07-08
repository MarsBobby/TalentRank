#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2: NLP 语义提取与 Extractor Agent
────────────────────────────────────────
Multi-Agent 架构（规则 + 统计，不依赖 GPU）：
  Agent A: 基本信息 & 时间线提取
  Agent B: 职级标准化 & 矛盾检测（含研发大厂技术职级校准）
  Agent C: 管理动词密度 & 汇报层级 & 管理跨度提取（扩充正则）
  Agent D: 成就量化信号提取（含行业专属KPI词典 & 品牌量级信号）
  Agent E: 仲裁 & 最终输出（修复：将 C/D 结果回写到特征向量）

修复记录（v2.0）：
  P0 Bug修复 — 原版 Agent C/D 的结果（verb_density, reporting_layer,
               achievements）在 _agent_e_arbitrate 中从未被使用，
               现已全部回写至 CandidateFeatureVector。
  P1 - Agent B: 引入 TECH_LEVEL_MAP，研发岗阿里P系/腾讯T系/字节L系 →
                标准1-6级，解决 NLP 职级校准器对技术 Title 低识别率问题
  P1 - Agent C: SPAN_PATTERNS_EXTENDED 扩充至9条正则 + 汉字数量词映射，
                覆盖"负责一支由38人组成的团队"/"十余人"等多样写法
  P1 - Agent D: INDUSTRY_KPI_PATTERNS 行业专属词典（电商/研发/品牌/销售/生产/HR）
                + 品牌量级二值信号，解决电商/研发 kpi_log_score NaN 问题
"""

import re
import os
import sys
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.phase0_data_assets import CandidateFeatureVector
from modules.phase1_feature_engineering import MNARAwareFeatureEngineer, S_LEVEL_MAP
from config import CONFIDENCE as CONF


# ══════════════════════════════════════════════════════════════
#  P1修复 — 常量定义
# ══════════════════════════════════════════════════════════════

# ── 研发技术职级映射（互联网大厂体系 → 标准1-6级）──────────────
TECH_LEVEL_MAP: Dict[str, int] = {
    # 阿里巴巴 P 系列
    "P4": 1, "P5": 2, "P6": 3, "P7": 4, "P8": 4, "P9": 5, "P10": 6,
    # 腾讯 T 系列
    "T1": 1, "T2": 1, "T3": 2, "T4": 2, "T5": 3, "T6": 3, "T7": 4, "T8": 5, "T9": 6,
    # 字节跳动 L 系列
    "L1": 1, "L2": 1, "L3": 2, "L4": 2, "L5": 3, "L6": 3, "L7": 4, "L8": 5, "L9": 6,
    # 华为级别
    "13级": 1, "14级": 1, "15级": 2, "16级": 3, "17级": 4, "18级": 5, "19级": 6,
    # 中文工程师序列
    "初级工程师": 1, "工程师": 2, "高级工程师": 3,
    "资深工程师": 4, "技术专家": 4, "高级技术专家": 5, "首席技术专家": 6,
    # 架构师序列
    "架构师": 4, "高级架构师": 5, "首席架构师": 6,
    # 英文通用
    "junior engineer": 1, "senior engineer": 3,
    "staff engineer": 4, "principal engineer": 5,
}
_TECH_LEVEL_RE = re.compile(r'\b([PLT])(\d{1,2})\b')

# ── 管理跨度提取：扩充正则 ──────────────────────────────────────
SPAN_PATTERNS_EXTENDED: List[str] = [
    r'(?:管理|带领|负责|统筹|领导|指挥|率领)[^，。\n]{0,15}?(\d+)\s*(?:人|名|位)',
    r'团队[规模人数大小]*[为有共计约达]*\s*(\d+)\s*(?:人|名|位)',
    r'(\d+)\s*(?:人|名|位)(?:的)?(?:团队|研发团队|销售团队|运营团队|设计团队|技术团队|规模)',
    r'下[辖属][^，。\n]*?(\d+)\s*(?:人|名|位)',
    r'(?:一支|一个|整个)[^，。\n]{0,20}?(\d+)\s*(?:人|名|位)',
    r'(?:管理|[下属团队])[^\d]*?(\d+)\s*(?:人|名)',
    r'带团队(\d+)',
    r'(?:直接|间接)管理[^，。\n]{0,10}?(\d+)\s*(?:人|名|位)',
    r'(?:人员规模|团队人数)[：:为]*\s*(\d+)',
]

# 汉字数量词 → 整数映射
CHINESE_NUM_MAP: Dict[str, int] = {
    '十余': 15, '十多': 15, '十几': 15,
    '二十余': 25, '二十多': 25, '三十余': 35, '三十多': 35,
    '五十余': 55, '近百': 90, '百余': 150, '百多': 150,
    '数十': 30, '数百': 300, '数千': 3000, '近千': 900,
}

# ── 行业专属KPI提取词典 ─────────────────────────────────────────
INDUSTRY_KPI_PATTERNS: Dict[str, List[Tuple[str, str]]] = {
    '电商': [
        (r'GMV[约达完成]*\s*(\d+\.?\d*)\s*(亿|万)', 'gmv'),
        (r'转化率[约达提升]*\s*(\d+\.?\d*)\s*%', 'conversion_rate'),
        (r'ROI[约达]*\s*(\d+\.?\d*)', 'roi'),
        (r'复购率[约达提升至]*\s*(\d+\.?\d*)\s*%', 'repurchase_rate'),
        (r'(?:月|年|日)活[用户]*[约达]*\s*(\d+\.?\d*)\s*(万|亿)', 'active_users'),
        (r'(?:点击率|CTR)[约达提升至]*\s*(\d+\.?\d*)\s*%', 'ctr'),
    ],
    '研发': [
        (r'性能提升[约达]*\s*(\d+\.?\d*)\s*%', 'perf_improvement'),
        (r'延迟(?:降低|减少)[约达]*\s*(\d+\.?\d*)\s*(?:%|ms|毫秒)', 'latency_reduction'),
        (r'(?:代码|测试)覆盖率[约达提升至]*\s*(\d+\.?\d*)\s*%', 'coverage'),
        (r'(?:QPS|TPS|吞吐量)[约达提升至]*\s*(\d+\.?\d*)', 'throughput'),
        (r'可用性[约达提升至]*\s*(\d+\.?\d*)\s*%', 'availability'),
        (r'(?:专利|发明专利)[^\d]*(\d+)\s*(?:项|件)', 'patents'),
    ],
    '品牌': [
        (r'曝光[量次][约达]*\s*(\d+\.?\d*)\s*(亿|万)', 'exposure'),
        (r'品牌(?:知名度|认知度)[约达提升]*\s*(\d+\.?\d*)\s*%', 'brand_awareness'),
        (r'声量[约达增长]*\s*(\d+\.?\d*)\s*%', 'share_of_voice'),
        (r'(?:联名|合作|campaign|活动)[^\d]*(\d+)\s*(?:个|次|场)', 'campaigns'),
        (r'(?:粉丝|关注)[约达增长至]*\s*(\d+\.?\d*)\s*(万|亿)', 'fans'),
    ],
    '销售': [
        (r'销售额[约达完成]*\s*(\d+\.?\d*)\s*(亿|万)', 'sales_amount'),
        (r'完成率[约达]*\s*(\d+\.?\d*)\s*%', 'achievement_rate'),
        (r'(?:客户数|客户)[约达增长至]*\s*(\d+\.?\d*)\s*(万|个|家)', 'customer_count'),
        (r'新开(?:客|发)[约达]*\s*(\d+\.?\d*)\s*(?:万|个|家)', 'new_customers'),
    ],
    '生产': [
        (r'良率[约达提升至]*\s*(\d+\.?\d*)\s*%', 'yield_rate'),
        (r'产能[约达提升至]*\s*(\d+\.?\d*)\s*(?:万|件|台|套)', 'capacity'),
        (r'(?:降本|节约|节省)[约达]*\s*(\d+\.?\d*)\s*(亿|万)', 'cost_saving'),
        (r'效率提升[约达]*\s*(\d+\.?\d*)\s*%', 'efficiency'),
    ],
    '人力资源': [
        (r'招聘[完成了]*\s*(\d+\.?\d*)\s*(?:人|名|位)', 'recruitment_count'),
        (r'离职率[约达降低至]*\s*(\d+\.?\d*)\s*%', 'attrition_rate'),
        (r'(?:员工|人员)满意度[约达提升至]*\s*(\d+\.?\d*)\s*%', 'satisfaction'),
        (r'覆盖[约]*\s*(\d+\.?\d*)\s*(万|人)', 'coverage_count'),
    ],
}

# ── 品牌量级关键词 ───────────────────────────────────────────────
BRAND_SCALE_KEYWORDS = [
    '百万级', '千万级', '亿级', '亿次', '千万次',
    '全国TOP', '行业第一', '行业领先', '市场第一',
    '全球', '亚太', '全渠道', '破圈', '出圈',
    '爆款', '现象级', '刷屏', '病毒式传播',
]
BRAND_AWARD_KEYWORDS = [
    '艾菲奖', 'Effie', '金鼠标', 'CLIO', '戛纳', 'Cannes',
    '金投赏', '虎啸奖', '年度最佳', '金奖', '银奖',
    '品牌升级', '品牌焕新',
]


@dataclass
class ExtractionReport:
    """Extractor Agent 输出的详细报告"""
    candidate_id:    str = ""
    file_name:       str = ""
    function:        str = ""
    # 基本信息
    name_hint:       str = ""
    target_position: str = ""
    current_salary:  Optional[str] = None
    location:        str = ""
    # 时间线
    timeline_entries: List[Dict] = field(default_factory=list)
    timeline_conflicts: List[str] = field(default_factory=list)
    # 职级
    s_level_raw:     str = ""
    s_level_calibrated: int = 1
    level_inflation_detected: bool = False
    tech_level_calibrated:    bool = False   # P1修复新增
    # 成就
    achievement_snippets: List[str] = field(default_factory=list)
    # P1修复新增：Agent C 结果（供 Agent E 回写）
    verb_density:          float = 0.0
    reporting_layer:       int   = 0
    span_candidates:       List[int] = field(default_factory=list)
    # P1修复新增：Agent D 结果（供 Agent E 回写）
    industry_kpi_achievements: List[Dict] = field(default_factory=list)
    brand_achievement_binary:  int = 0
    # 置信度
    overall_confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)


class ExtractorAgent:
    """
    简历 Extractor Agent — 协调 5 个子 Agent 完成深度语义提取
    """

    def __init__(self):
        self._engineer = MNARAwareFeatureEngineer()

    # ────── 公开入口 ──────────────────────────────────────────

    def extract(self, raw_text: str, file_name: str = "",
                function: str = "") -> Tuple[CandidateFeatureVector, ExtractionReport]:
        """
        输入：原始简历文本
        输出：(特征向量, 详细报告)

        Bug修复（v2.0）：原版 Agent C/D 的结果从未被传入 Agent E，
        现在通过 report 对象携带并在 _agent_e_arbitrate 中回写。
        """
        report = ExtractionReport(file_name=file_name, function=function)

        # Agent A: 基本信息 & 时间线
        self._agent_a_basic_and_timeline(raw_text, report)
        # Agent B: 职级标准化 & 矛盾检测（含研发大厂职级校准）
        self._agent_b_level_calibration(raw_text, report, function)
        # Agent C: 管理动词密度 & 汇报层级 & 跨度提取（扩充版）
        verb_density, reporting_layer, span_candidates = self._agent_c_management(raw_text)
        report.verb_density      = verb_density
        report.reporting_layer   = reporting_layer
        report.span_candidates   = span_candidates
        # Agent D: 成就量化（含行业专属KPI & 品牌量级信号）
        self._agent_d_achievements(raw_text, report, function)
        # Agent E: 仲裁 & 合并（将 A/B/C/D 全部结果回写到向量）
        vec = self._agent_e_arbitrate(raw_text, file_name, function, report)

        return vec, report

    # ────── Agent A: 基本信息 & 时间线 ───────────────────────

    def _agent_a_basic_and_timeline(self, text: str, report: ExtractionReport):
        """提取基本信息和工作时间线"""
        # 目标职位
        pos_m = re.search(
            r"(?:推荐职位|目标岗位|应聘职位|for[\s\-]+)[:：]?\s*([^\n\t，,]{3,30})",
            text, re.IGNORECASE
        )
        if pos_m:
            report.target_position = pos_m.group(1).strip()

        # 候选人姓名（文件名中的大写字母组合）
        name_m = re.search(r"姓\s*名[：:]\s*([A-Za-z\u4e00-\u9fa5]{2,8})", text)
        if name_m:
            report.name_hint = name_m.group(1)

        # 薪资
        sal_m = re.search(r"目前薪资[：:]\s*([^\n]{3,40})", text)
        if sal_m:
            report.current_salary = sal_m.group(1).strip()

        # 地点
        loc_m = re.search(r"(?:现居住地|现居|所在地)[：:]\s*([\u4e00-\u9fa5]{2,8})", text)
        if loc_m:
            report.location = loc_m.group(1)

        # 时间线提取
        entries = []
        # YYYY.MM-YYYY.MM 格式
        for m in re.finditer(
            r"(\d{4})[./-](\d{1,2})\s*[-~至到]\s*(\d{4})[./-](\d{1,2})"
            r"(?:[^\n]{0,60})?",
            text
        ):
            entries.append({
                "start_y": int(m.group(1)), "start_m": int(m.group(2)),
                "end_y":   int(m.group(3)), "end_m":   int(m.group(4)),
                "snippet": m.group(0)[:80],
            })
        # 含"至今"
        for m in re.finditer(
            r"(\d{4})[./-](\d{1,2})\s*[-~至到]\s*(?:至今|现在)(?:[^\n]{0,50})?",
            text
        ):
            from datetime import datetime
            now = datetime.now()
            entries.append({
                "start_y": int(m.group(1)), "start_m": int(m.group(2)),
                "end_y": now.year, "end_m": now.month,
                "snippet": m.group(0)[:80],
            })
        report.timeline_entries = entries

        # 时间重叠检测（超过 6 个月 → 冲突警告）
        # 合并相邻区间（同一段工作经历的多个日期段不应算冲突）
        merged = []
        sorted_entries = sorted(entries, key=lambda e: e["start_y"]*12+e["start_m"])
        for e in sorted_entries:
            s = e["start_y"]*12+e["start_m"]
            en = e["end_y"]*12+e["end_m"]
            if merged and s <= merged[-1][1] + 3:  # 3个月内视为同一段
                merged[-1] = (min(merged[-1][0], s), max(merged[-1][1], en))
            else:
                merged.append((s, en))
        # 检测合并后的区间是否有实质性重叠（> 12个月）
        for i in range(len(merged)):
            for j in range(i+1, len(merged)):
                overlap = min(merged[i][1], merged[j][1]) - max(merged[i][0], merged[j][0])
                if overlap > 12:
                    report.timeline_conflicts.append(
                        f"区间{i+1}与区间{j+1}重叠 {overlap} 个月"
                    )
                    report.warnings.append(f"⚠ 时间重叠 {overlap} 月")

    # ────── Agent B: 职级校准 ─────────────────────────────────

    def _agent_b_level_calibration(self, text: str, report: ExtractionReport,
                                   function: str = ""):
        """
        基于管理动词密度纠正头衔通胀；
        研发岗额外使用 TECH_LEVEL_MAP 校准大厂技术职级（P1修复）。
        """
        best_kw, best_level = "", 1
        for kw, lvl in sorted(S_LEVEL_MAP.items(), key=lambda x: len(x[0]), reverse=True):
            if kw in text:
                best_kw, best_level = kw, lvl
                break
        report.s_level_raw = best_kw

        # ── P1修复：研发岗大厂技术职级优先匹配 ──────────────────
        if function == "研发":
            # 1. 完整职级词匹配（"高级工程师""架构师"等）
            for tech_title, level in TECH_LEVEL_MAP.items():
                if tech_title in text:
                    best_level = level
                    report.level_inflation_detected = False  # 技术职级不触发通胀警告
                    report.warnings.append(f"ℹ 技术职级校准: '{tech_title}' → L{level}")
                    break
            # 2. 大厂 P/L/T+数字模式（P7、L6、T8 等）
            m = _TECH_LEVEL_RE.search(text)
            if m:
                key = f'{m.group(1)}{m.group(2)}'
                if key in TECH_LEVEL_MAP:
                    best_level = TECH_LEVEL_MAP[key]
                    report.warnings.append(f"ℹ 大厂职级校准: '{key}' → L{best_level}")
            report.tech_level_calibrated = True

        # ── 通用动词密度校验（非研发岗头衔通胀检测）─────────────
        words = text.split()
        mgt_count = sum(
            1 for w in words
            for v in ["管理", "统筹", "领导", "决策", "战略", "主导", "规划", "推进"]
            if v in w
        )
        density = mgt_count / max(len(words), 1)

        calibrated = best_level
        if function != "研发" and best_level >= 4 and density < 0.03:
            calibrated = max(2, best_level - 1)
            report.level_inflation_detected = True
            report.warnings.append(
                f"⚠ 头衔通胀：'{best_kw}'(L{best_level}) → 校准为 L{calibrated}"
            )

        report.s_level_calibrated = calibrated

    # ────── Agent C: 管理动词 & 汇报层级 & 管理跨度 ─────────────

    def _agent_c_management(self, text: str) -> Tuple[float, int, List[int]]:
        """
        P1修复：原版仅返回 (density, layer)，现增返回 span_candidates。
        使用 SPAN_PATTERNS_EXTENDED 扩充正则 + 汉字数量词映射。
        """
        words = text.split()
        mgt_verbs = [
            "统筹", "领导", "管理", "带领", "指挥", "协调", "决策",
            "规划", "主导", "负责", "推进", "构建", "优化", "整合",
        ]
        count = sum(1 for w in words for v in mgt_verbs if v in w)
        density = round(count / max(len(words), 1) * 100, 3)

        layer = 0
        for pattern, level in [
            (r"向董事长|向CEO|向总裁", 5),
            (r"向副总裁|向VP", 4),
            (r"向总监|向总经理", 3),
            (r"向经理|向主管", 2),
        ]:
            if re.search(pattern, text):
                layer = level
                break

        # ── P1修复：扩充管理跨度提取 ─────────────────────────────
        span_candidates: List[int] = []
        seen: set = set()

        # 1. 扩充数字正则
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

        # 2. 汉字数量词（"十余人""数百人"等）
        for cn_word, num_val in CHINESE_NUM_MAP.items():
            if re.search(re.escape(cn_word) + r'\s*(?:人|名|位)?', text):
                if num_val not in seen:
                    span_candidates.append(num_val)
                    seen.add(num_val)

        return density, layer, span_candidates

    # ────── Agent D: 成就量化 ────────────────────────────────

    def _agent_d_achievements(self, text: str, report: ExtractionReport,
                              function: str = "") -> List[str]:
        """
        P1修复：新增行业专属KPI提取 + 品牌量级二值信号。
        原有通用量化句提取逻辑完整保留。
        """
        # ── 通用量化成就句提取（原逻辑保留）──────────────────────
        snippets = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            has_number = re.search(r"\d+\.?\d*\s*[%万亿倍]|增长|提升|节省", line)
            has_action = re.search(r"(?:主导|负责|推动|完成|实现|达成|帮助)", line)
            if has_number and has_action and len(line) > 10:
                snippets.append(line[:120])
        report.achievement_snippets = snippets[:10]

        # ── P1修复：行业专属KPI提取 ──────────────────────────────
        industry_kpis: List[Dict] = []
        for pattern, kpi_name in INDUSTRY_KPI_PATTERNS.get(function, []):
            try:
                for m in re.finditer(pattern, text):
                    raw = m.group(1)
                    unit = m.group(2) if m.lastindex and m.lastindex >= 2 else ""
                    try:
                        industry_kpis.append({
                            "kpi_name": kpi_name,
                            "value": float(raw),
                            "unit": unit,
                        })
                    except ValueError:
                        continue
            except re.error:
                continue
        report.industry_kpi_achievements = industry_kpis

        # ── P1修复：品牌岗量级二值信号 ──────────────────────────
        if function == "品牌":
            has_scale = any(kw in text for kw in BRAND_SCALE_KEYWORDS)
            has_award = any(kw in text for kw in BRAND_AWARD_KEYWORDS)
            report.brand_achievement_binary = 1 if (has_scale or has_award) else 0
        else:
            report.brand_achievement_binary = 0

        return snippets

    # ────── Agent E: 仲裁 & 合并到特征向量 ───────────────────

    def _agent_e_arbitrate(self, text: str, file_name: str,
                           function: str, report: ExtractionReport) -> CandidateFeatureVector:
        """
        融合 Agent A-D 的结果，调用 Phase1 FeatureEngineer 生成基础向量，
        再将所有 Agent 的结果回写覆盖。

        Bug修复（v2.0）：原版只回写了 s_level 和 conflict_flags，
        Agent C 的 verb_density/reporting_layer/span_candidates 及
        Agent D 的 industry_kpis/brand_achievement_binary 均未被应用。
        """
        # 1. 调用 Phase1 工程师生成基础特征向量
        vec = self._engineer.engineer(
            raw_text=text,
            file_name=file_name,
            function=function,
        )

        # 2. 回写 Agent B 结果：职级校准
        vec.s_level = float(report.s_level_calibrated)
        if report.level_inflation_detected:
            vec.s_level_conf = CONF["context_infer"]
        if getattr(report, "tech_level_calibrated", False):
            vec.tech_level_calibrated = True

        # 3. 回写 Agent C 结果（Bug修复：原版未回写）
        vec.management_verb_density = getattr(report, "verb_density", 0.0)
        vec.reporting_layer         = float(getattr(report, "reporting_layer", 0))

        # 管理跨度：仅在 Phase1 提取为空时用 Agent C 扩充结果补充
        span_candidates = getattr(report, "span_candidates", [])
        if span_candidates and (vec.span_of_control is None):
            vec.span_of_control = float(max(span_candidates))
            vec.span_missing    = 0

        # 4. 回写 Agent D 结果（Bug修复：原版未回写）
        industry_kpis = getattr(report, "industry_kpi_achievements", [])
        vec.industry_kpi_hit_count = len(industry_kpis)

        # 行业专属KPI对总分的加成（最多10分，防止量纲爆炸）
        if industry_kpis and (vec.kpi_log_score is None):
            bonus_raw = min(len(industry_kpis) * 2.0, 10.0)
            vec.kpi_log_score = round(math.log(bonus_raw + 1), 4)
            vec.kpi_missing   = 0

        # 品牌量级二值信号
        vec.brand_achievement_binary = getattr(report, "brand_achievement_binary", 0)

        # 5. 时间冲突 → 降低综合置信度
        vec.conflict_flags = len(report.timeline_conflicts)
        if report.timeline_conflicts:
            vec.global_confidence = max(
                0.0, vec.global_confidence - 0.1 * len(report.timeline_conflicts)
            )

        # 6. 更新报告
        report.candidate_id      = vec.candidate_id
        report.overall_confidence = vec.global_confidence

        return vec


def print_extraction_report(report: ExtractionReport):
    """打印 Phase 2 提取报告"""
    print(f"\n  {'─'*55}")
    print(f"  📋 候选人: {report.name_hint or report.candidate_id}")
    print(f"  📂 文件:   {report.file_name}")
    print(f"  💼 目标职位: {report.target_position}")
    print(f"  📍 地点: {report.location}")
    print(f"  💰 目前薪资: {report.current_salary or '未披露'}")
    print(f"  🎯 职级校准: {report.s_level_raw} → L{report.s_level_calibrated}"
          + (" ⚠头衔通胀" if report.level_inflation_detected else ""))
    print(f"  📅 时间线条目: {len(report.timeline_entries)} 段"
          + (f"  ⚠冲突:{report.timeline_conflicts}" if report.timeline_conflicts else ""))
    print(f"  🏆 量化成就片段: {len(report.achievement_snippets)} 条")
    if report.achievement_snippets:
        for s in report.achievement_snippets[:3]:
            print(f"      · {s[:80]}")
    print(f"  📊 综合置信度: {report.overall_confidence:.2f}")
    if report.warnings:
        for w in report.warnings:
            print(f"  {w}")