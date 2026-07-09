#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — 工业级人才优选体系 · 全链路主程序
Phase 0 → Phase 8 完整流水线

用法（在项目根目录运行）:
  python main.py --function 品牌
  python main.py --function 生产 --data_dir BASE\_DIR/data/manufacturing
  python main.py --function hr   --data_dir BASE\_DIR/data/hr

修复记录（v2.2）：
  P1 - _quality_audit：generate_quality_report 补传 function 参数，
       启用研发/生产差异化置信度阈值（0.48），消除过度人工路由。
  P0 - Phase 8 gold_relevance：原版用 score/max_score*4 把排序分直接当黄金答案，
       与 api.py 里已修复的循环参考问题完全相同。
       现改为三层递进策略（与 api.py 保持一致）：
         Layer 1 — HR 黄金文件 gold_standards/<function>.json（HR打分后优先加载）
         Layer 2 — 信号代理金标准（独立权重，与 Phase4/5 权重不同，打破循环依赖）
         Layer 3 — metrics_source 字段标注可信度
"""

import os
import sys
import json
import glob
import math
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# ── 路径配置：确保 modules/ 可以被正确导入 ─────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ── 设置 logging ───────────────────────────────────────────────
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ── 引入各 Phase 模块 ──────────────────────────────────────────
from modules.phase0_data_assets  import print_field_inventory, CandidateFeatureVector
from modules.phase1_feature_engineering import MNARAwareFeatureEngineer
from modules.phase2_nlp_extractor       import ExtractorAgent, print_extraction_report
from modules.phase3_confidence          import (bayesian_impute, audit_confidence,
                                                generate_quality_report, print_quality_report)
from modules.phase4_weighting           import DynamicWeighter, print_weight_report
from modules.phase5_lambdamart          import LambdaMARTRanker, print_ranking_result
from modules.phase6_bandit              import ContextualBanditSystem, simulate_hr_feedback
from modules.phase7_explainability      import SHAPExplainabilityAgent, print_global_explanation
from modules.phase8_evaluation          import EvaluationSystem, print_evaluation_report

# ── 引入已有的提取工具 ─────────────────────────────────────────
from docx_extractor import DocxExtractor

# ── 黄金标准文件目录 ──────────────────────────────────────────
GOLD_DIR = os.path.join(BASE_DIR, "gold_standards")
os.makedirs(GOLD_DIR, exist_ok=True)

# ── 职能→数据目录默认映射 ──────────────────────────────────────
FUNCTION_DIR_MAP = {
    "品牌":    "branding",
    "电商":    "ecommerce",
    "人力资源": "hr",
    "hr":      "hr",
    "生产":    "manufacturing",
    "研发":    "rnd",
    "销售":    "sales",
}


# ══════════════════════════════════════════════════════════════
#  三层评估金标准体系（与 api.py 共享同一逻辑，保持一致）
# ══════════════════════════════════════════════════════════════

def _load_hr_gold_standard(function: str) -> Optional[Dict[str, float]]:
    """
    Layer 1：加载 HR 黄金标准文件。
    HR 通过 POST /api/hr_rating 打分后文件写入 gold_standards/<function>.json。
    命令行模式下也优先读取该文件，保证 main.py 与 api.py 评估基准一致。
    """
    gold_path = os.path.join(GOLD_DIR, f"{function}.json")
    if not os.path.exists(gold_path):
        return None
    try:
        with open(gold_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        gold = {
            cid: float(max(0, min(4, int(score))))
            for cid, score in raw.items()
            if isinstance(score, (int, float)) and not str(cid).startswith("_")
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

    权重配置刻意独立于 Phase4 AHP / Phase5 回退权重，避免循环依赖：
      education_score:    0.25  —— 学历是最稳定可验证信号
      s_level:            0.20  —— 职级语义校准后的层级
      total_work_years:   0.20  —— 总工龄（贝叶斯填补，缺失率低）
      achievement_signal: 0.20  —— 二值信号，有量化成就 → 1
      kpi_log_score:      0.10  —— MNAR，有则加权，缺失不惩罚
      stability_index:    0.05  —— 稳定性

    置信度 < 0.5 的候选人得分乘以 0.6 惩罚系数，避免低质量特征污染基线。
    最终映射到 0-4（NDCG 标准格式）。
    """
    PROXY_WEIGHTS = {
        "education_score":    0.25,
        "s_level":            0.20,
        "total_work_years":   0.20,
        "achievement_signal": 0.20,
        "kpi_log_score":      0.10,
        "stability_index":    0.05,
    }
    NORMALIZERS = {
        "education_score":    3.0,
        "s_level":            6.0,
        "total_work_years":   20.0,
        "achievement_signal": 1.0,
        "kpi_log_score":      10.0,
        "stability_index":    1.0,
    }

    raw_scores: Dict[str, float] = {}
    for vec in vecs:
        d = vec.to_feature_dict()
        score = 0.0
        for feat, w in PROXY_WEIGHTS.items():
            val = d.get(feat)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                continue   # MNAR：跳过，不扣分
            norm_val = min(1.0, float(val) / NORMALIZERS.get(feat, 1.0))
            score += norm_val * w

        # 置信度惩罚
        conf = vec.global_confidence
        if conf < 0.5:
            score *= 0.6
        elif conf < 0.65:
            score *= 0.85

        raw_scores[vec.candidate_id] = score

    if not raw_scores:
        return {}

    min_s = min(raw_scores.values())
    max_s = max(raw_scores.values())
    span  = max_s - min_s

    return {
        cid: round((s - min_s) / span * 4, 2) if span > 0 else 2.0
        for cid, s in raw_scores.items()
    }


def _get_relevant_ids_from_gold(
    gold: Dict[str, float],
    threshold: float = 2.5,
) -> List[str]:
    """从金标准中取出相关性 >= threshold 的候选人作为 MRR 正例集合。"""
    relevant = [cid for cid, score in gold.items() if score >= threshold]
    if not relevant:
        sorted_ids = sorted(gold.keys(), key=lambda c: gold[c], reverse=True)
        n = max(1, len(sorted_ids) // 3)
        relevant = sorted_ids[:n]
    return relevant


def _compute_evaluation_metrics(
    predicted_order: List[str],
    vecs: List[CandidateFeatureVector],
    function: str,
    evaluator: EvaluationSystem,
) -> Tuple[Dict, str]:
    """
    三层递进评估，返回 (metrics_dict, source_label)。
    source_label: "hr_gold" | "signal_proxy"
    """
    from modules.phase8_evaluation import ndcg_at_k, mean_reciprocal_rank

    gold = _load_hr_gold_standard(function)
    source = "hr_gold"

    if gold is None:
        gold = _build_proxy_gold_relevance(vecs)
        source = "signal_proxy"
        logger.info(f"[评估] {function} 使用代理金标准，共 {len(gold)} 位候选人")

    if not gold:
        return {"NDCG@5": None, "NDCG@10": None, "MRR": None}, "no_data"

    relevant = _get_relevant_ids_from_gold(gold)

    # 通过 EvaluationSystem 记录历史（支持跨版本趋势对比）
    eval_results = evaluator.evaluate(
        predicted_order=predicted_order,
        gold_relevance=gold,
        positive_set=relevant,
        function=function,
    )
    return eval_results, source


# ══════════════════════════════════════════════════════════════
#  主流水线
# ══════════════════════════════════════════════════════════════

class TalentPipeline:
    """完整人才优选流水线"""

    def __init__(self, data_root: str, function: str, output_dir: str):
        self.data_root  = data_root
        self.function   = function
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.docx_extractor  = DocxExtractor()
        self.extractor_agent = ExtractorAgent()
        self.weighter        = DynamicWeighter()
        self.ranker          = LambdaMARTRanker()
        self.bandit          = ContextualBanditSystem()
        self.evaluator       = EvaluationSystem()

    # ──────────────────────────────────────────────────────────
    # Step 1: 扫描 .docx 文件
    # ──────────────────────────────────────────────────────────
    def _scan_files(self, data_dir: str) -> List[str]:
        patterns = [
            os.path.join(data_dir, "*.docx"),
            os.path.join(data_dir, "**", "*.docx"),
        ]
        files = []
        for pat in patterns:
            files.extend(glob.glob(pat, recursive=True))
        files = list(set(files))
        logger.info(f"[扫描] 发现 {len(files)} 份简历")
        return files

    # ──────────────────────────────────────────────────────────
    # Step 2: DOCX 提取 → Phase2 NLP → Phase1 特征工程
    # ──────────────────────────────────────────────────────────
    def _extract_features(self, files: List[str]) -> List[CandidateFeatureVector]:
        vecs = []
        for fpath in files:
            fname = os.path.basename(fpath)
            logger.info(f"  处理: {fname}")

            extraction = self.docx_extractor.extract_from_file(fpath)
            if extraction["status"] == "failure":
                logger.warning(f"  ⚠ 提取失败: {extraction['error']}")
                continue

            raw_text = extraction["text"]

            try:
                from urllib.parse import unquote
                fname_display = unquote(fname)
            except Exception:
                fname_display = fname

            vec, report = self.extractor_agent.extract(
                raw_text=raw_text,
                file_name=fname_display,
                function=self.function,
            )
            print_extraction_report(report)

            vec = bayesian_impute(vec)
            vecs.append(vec)

        logger.info(f"[提取完成] 有效候选人: {len(vecs)}")
        return vecs

    # ──────────────────────────────────────────────────────────
    # Phase 3: 质量审核
    # ──────────────────────────────────────────────────────────
    def _quality_audit(self, vecs: List[CandidateFeatureVector]) -> Dict:
        # P1修复：传入 function，启用研发/生产差异化阈值（0.48）
        # 原版 generate_quality_report(vecs) 缺少此参数，导致阈值修复失效
        report = generate_quality_report(vecs, function=self.function)
        print_quality_report(report)
        return report

    # ──────────────────────────────────────────────────────────
    # Phase 4: 动态权重
    # ──────────────────────────────────────────────────────────
    def _compute_weights(self, vecs: List[CandidateFeatureVector]) -> Dict:
        weights = self.weighter.compute_weights(vecs, function=self.function)
        print_weight_report(weights, self.function)
        return weights

    # ──────────────────────────────────────────────────────────
    # Phase 5: LambdaMART 排序
    # ──────────────────────────────────────────────────────────
    def _rank(self, vecs: List[CandidateFeatureVector], weights: Dict):
        model_path = os.path.join(self.output_dir, f"model_{self.function}.lgb")
        self.ranker.load_model(model_path)

        ranked = self.ranker.rank(vecs)
        print_ranking_result(ranked)

        phase4_scores = self.weighter.score_candidates(vecs, weights)
        print("\n  Phase 4 权重评分（对比参考）:")
        for cid, score in phase4_scores[:6]:
            print(f"    {cid:<18} {score:.4f}")

        return ranked

    # ──────────────────────────────────────────────────────────
    # Phase 6: Contextual Bandit 模拟反馈
    # ──────────────────────────────────────────────────────────
    def _bandit_feedback(self, vecs: List[CandidateFeatureVector],
                          ranked, weights: Dict):
        from modules.phase5_lambdamart import FEATURE_ORDER
        self.bandit.initialize_from_weights(self.function, weights, FEATURE_ORDER)
        simulate_hr_feedback(self.bandit, vecs, ranked, self.function, top_k=3)

        log_path = os.path.join(self.output_dir, f"feedback_{self.function}.json")
        self.bandit.save_feedback_log(log_path)

    # ──────────────────────────────────────────────────────────
    # Phase 7: SHAP 解释性
    # ──────────────────────────────────────────────────────────
    def _explain(self, vecs: List[CandidateFeatureVector],
                  ranked, weights: Dict):
        shap_agent = SHAPExplainabilityAgent(
            model=self.ranker.model,
            weights=weights,
        )

        global_exp = shap_agent.explain_global(vecs)
        print_global_explanation(global_exp)

        vec_map = {v.candidate_id: v for v in vecs}
        print("\n  Phase 7 · 候选人个人解释报告")
        for cid, score, rank in ranked[:3]:
            vec = vec_map.get(cid)
            if vec:
                report_str = shap_agent.generate_hr_report(vec, rank, vecs)
                print(report_str)

        return global_exp

    # ──────────────────────────────────────────────────────────
    # 保存最终输出 JSON
    # ──────────────────────────────────────────────────────────
    def _save_output(self, vecs: List[CandidateFeatureVector],
                      ranked, quality_report: Dict,
                      eval_results: Dict, global_exp: Dict,
                      metrics_source: str):
        output = {
            "function":         self.function,
            "run_time":         datetime.now().isoformat(),
            "total_candidates": len(vecs),
            "ranking": [
                {"rank": r, "candidate_id": cid, "score": round(score, 4)}
                for cid, score, r in ranked
            ],
            "quality_summary": {
                "avg_confidence":     quality_report.get("avg_confidence"),
                "human_review_queue": quality_report.get("human_review_queue", []),
            },
            "evaluation_metrics": eval_results,
            # metrics_source 告知结果消费方当前指标的可信度级别
            "metrics_source": metrics_source,
            "top_shap_drivers": [
                {"feature": feat, "importance": imp}
                for feat, imp in global_exp.get("top_drivers", [])
            ],
        }

        out_path = os.path.join(self.output_dir, f"result_{self.function}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info(f"\n  ✅ 结果已保存: {out_path}")
        return out_path

    # ──────────────────────────────────────────────────────────
    # 完整 run()
    # ──────────────────────────────────────────────────────────
    def run(self, data_dir: str):
        print("\n" + "█" * 65)
        print(f"  🚀 工业级人才优选体系 · 职能={self.function}")
        print("█" * 65)

        # ── Phase 0 ──
        print("\n【Phase 0】数据资产盘点")
        print_field_inventory()

        # ── 扫描文件 ──
        files = self._scan_files(data_dir)
        if not files:
            logger.error(f"❌ 未在 {data_dir} 找到 .docx 文件，请检查路径")
            return

        # ── Phase 1 & 2 ──
        print("\n【Phase 1 & 2】DOCX 提取 + NLP 特征工程")
        vecs = self._extract_features(files)
        if not vecs:
            logger.error("❌ 所有文件提取失败，流水线终止")
            return

        # ── Phase 3 ──
        print("\n【Phase 3】置信度审核")
        quality_report = self._quality_audit(vecs)

        # ── Phase 4 ──
        print("\n【Phase 4】动态混合赋权（AHP + EWM）")
        weights = self._compute_weights(vecs)

        # ── Phase 5 ──
        print("\n【Phase 5】LambdaMART 排序")
        ranked = self._rank(vecs, weights)

        # ── Phase 6 ──
        print("\n【Phase 6】Contextual Bandit 在线反馈模拟")
        self._bandit_feedback(vecs, ranked, weights)

        # ── Phase 7 ──
        print("\n【Phase 7】SHAP 解释性分析")
        global_exp = self._explain(vecs, ranked, weights)

        # ── Phase 8 ──
        print("\n【Phase 8】系统评估")
        predicted_order = [cid for cid, score, rank in ranked]

        # P0修复：三层递进评估，替换原来的循环参考写法
        # 原版：gold_relevance = {cid: score/max_score*4 ...} 用排序分当黄金答案
        # 修复后：优先 HR 黄金文件，无文件时用独立权重的信号代理金标准
        eval_results, metrics_source = _compute_evaluation_metrics(
            predicted_order, vecs, self.function, self.evaluator
        )
        print_evaluation_report(eval_results, self.function)
        print(f"\n  📌 评估指标来源: {metrics_source}")
        if metrics_source == "signal_proxy":
            print("     当前为代理金标准（基于候选人特征信号估算）。")
            print(f"     如需精确评估，请将 HR 评分写入 gold_standards/{self.function}.json")
            print("     或通过 POST /api/hr_rating 接口提交评分后重新运行。")
        else:
            print("     当前为 HR 黄金标准评分，指标可信。")

        # ── 保存 ──
        out_path = self._save_output(
            vecs, ranked, quality_report, eval_results, global_exp, metrics_source
        )

        print("\n" + "█" * 65)
        print(f"  🏁 流水线完成！输出文件: {out_path}")
        print("█" * 65 + "\n")


# ══════════════════════════════════════════════════════════════
#  命令行入口
# ══════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="工业级人才优选体系 · Phase 0-8 全链路流水线"
    )
    parser.add_argument(
        "--function", "-f",
        type=str, default="品牌",
        choices=["品牌", "电商", "人力资源", "hr", "生产", "研发", "销售"],
        help="职能类别（对应 data/ 子目录）",
    )
    parser.add_argument(
        "--data_root", "-d",
        type=str,
        default=None,
        help="数据根目录（默认：脚本同级的 data/ 文件夹）",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="直接指定数据目录（优先于 --data_root + --function）",
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=None,
        help="输出目录（默认：脚本同级的 outputs/ 文件夹）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_root = args.data_root or os.path.join(BASE_DIR, "data")
        sub_dir   = FUNCTION_DIR_MAP.get(args.function, args.function)
        data_dir  = os.path.join(data_root, sub_dir)

    output_dir = args.output_dir or os.path.join(BASE_DIR, "outputs")

    if not os.path.isdir(data_dir):
        print(f"\n❌ 数据目录不存在: {data_dir}")
        print(f"   请用 --data_dir 指定正确路径，例如：")
        print(f"   python main.py --function 品牌 --data_dir D:/project_TalentSystem/data/branding")
        sys.exit(1)

    pipeline = TalentPipeline(
        data_root=data_dir,
        function=args.function,
        output_dir=output_dir,
    )
    pipeline.run(data_dir)



if __name__ == "__main__":
    main()
