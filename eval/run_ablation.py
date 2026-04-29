#!/usr/bin/env python3
"""
消融实验与 Baseline 对比

实验列表:
  B0  当前系统 (Hybrid BM25+FAISS + CrossEncoder Reranker, recall=50)
  B1  BM25 Only + Reranker
  B2  Dense (FAISS) Only + Reranker
  B3  Hybrid 无 Reranker（RRF 融合排序）
  A1  Hybrid + Reranker，关闭年份硬过滤
  A3a Hybrid + Reranker，recall_candidates=20
  A3b Hybrid + Reranker，recall_candidates=100

用法:
  cd /mnt/disk4/EscLab/VAD/igem-rag
  CUDA_VISIBLE_DEVICES="" conda run -n igem-rag python eval/run_ablation.py
  CUDA_VISIBLE_DEVICES="" conda run -n igem-rag python eval/run_ablation.py --sample 50
  CUDA_VISIBLE_DEVICES="" conda run -n igem-rag python eval/run_ablation.py --exp B0 B1 B2
"""

import json
import sys
import os
import time
import argparse
import random
import re
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv

os.chdir(Path(__file__).parent.parent)
sys.path.insert(0, ".")
load_dotenv()

from QwenRAGSystemOptimized import QwenRAGSystemOptimized

DATASET_PATH = "./eval/eval_dataset.json"
RESULTS_DIR = Path("./eval/ablation_results")
SUMMARY_PATH = "./eval/ablation_summary.json"
TOP_K = 10

EXPERIMENTS = [
    {"tag": "B0", "name": "Hybrid+Reranker（当前系统）",
     "use_bm25": True,  "use_faiss": True,  "use_reranker": True,  "use_year_filter": True,  "recall": 50},
    {"tag": "B1", "name": "BM25 Only",
     "use_bm25": True,  "use_faiss": False, "use_reranker": True,  "use_year_filter": True,  "recall": 50},
    {"tag": "B2", "name": "Dense (FAISS) Only",
     "use_bm25": False, "use_faiss": True,  "use_reranker": True,  "use_year_filter": True,  "recall": 50},
    {"tag": "B3", "name": "Hybrid 无Reranker (RRF)",
     "use_bm25": True,  "use_faiss": True,  "use_reranker": False, "use_year_filter": True,  "recall": 50},
    {"tag": "A1", "name": "Hybrid 无年份过滤",
     "use_bm25": True,  "use_faiss": True,  "use_reranker": True,  "use_year_filter": False, "recall": 50},
    {"tag": "A3a", "name": "Hybrid recall=20",
     "use_bm25": True,  "use_faiss": True,  "use_reranker": True,  "use_year_filter": True,  "recall": 20},
    {"tag": "A3b", "name": "Hybrid recall=100",
     "use_bm25": True,  "use_faiss": True,  "use_reranker": True,  "use_year_filter": True,  "recall": 100},
]


# ========== 命中判断（与 run_eval.py 保持一致） ==========

def normalize_section(section: str):
    s = section.lower()
    if any(k in s for k in ['wet lab', 'wetlab', '湿实验']): return 'Wet Lab'
    if any(k in s for k in ['建模', 'model', 'dry lab', '干实验']): return 'Model'
    if any(k in s for k in ['human practices', 'humanpractices', '人类实践', '人文实践',
                              '人力实践', '人本实践', '公众实践']): return 'Human Practices'
    return None


def normalize_subsection(sub: str, sec: str):
    if sec == 'Wet Lab':
        if '底盘生物' in sub: return '底盘生物的选择'
        if '关键' in sub and ('元件' in sub or '组件' in sub): return '关键生物元件与表达产物'
        if '创新' in sub: return '实验方法创新点'
        if '验证方法' in sub or ('验证' in sub and '结果' not in sub): return '实验验证方法'
        if '验证结果' in sub or ('结果' in sub and '验证' in sub): return '实验验证结果'
        if '技术挑战' in sub: return '技术挑战及解决方案'
        if '实验设计' in sub: return '实验设计分析'
    elif sec == 'Model':
        if '建模目' in sub or '模型目' in sub: return '建模目的'
        if '模型类型' in sub: return '模型类型'
        if '输入参数' in sub or ('假设' in sub and '条件' in sub): return '输入参数与假设'
        if '吻合' in sub or '一致' in sub or '符合' in sub: return '预测与实验数据吻合度'
        if '开源' in sub or '代码' in sub: return '开源代码'
    elif sec == 'Human Practices':
        if '社会需求' in sub: return '社会需求调研方法'
        if '利益相关' in sub: return '利益相关方合作'
        if '公众推广' in sub or '公众科普' in sub or '科普活动' in sub or '推广活动' in sub: return '公众推广活动'
        if '伦理' in sub: return '伦理审查'
        if '具体案例' in sub or '影响力' in sub or '影响数据' in sub: return '具体案例与影响'
    return None


def check_rule_hit(retrieved: list, gt: dict):
    for rank, chunk in enumerate(retrieved, start=1):
        meta = chunk['metadata']
        r_sec = normalize_section(meta.get('Section', '') or '')
        r_sub = normalize_subsection(meta.get('Subsection', '') or '', r_sec) if r_sec else None
        if (meta.get('team_name') == gt['team_name']
                and meta.get('year') == gt['year']
                and r_sub == gt['canon_subsection']):
            return rank
    return None


def check_llm_hit(retrieved: list, gt: dict, text_to_id: dict):
    for rank, chunk in enumerate(retrieved, start=1):
        if text_to_id.get(chunk['text'][:100]) == gt['chunk_id']:
            return rank
    return None


def compute_metrics(hit_ranks: list, k_list=(1, 3, 5)) -> dict:
    n = len(hit_ranks)
    if n == 0:
        return {}
    m = {f'Hit@{k}': round(sum(1 for r in hit_ranks if r and r <= k) / n, 4) for k in k_list}
    m['MRR'] = round(sum(1 / r for r in hit_ranks if r) / n, 4)
    m['n'] = n
    return m


# ========== 单次实验 ==========

def run_experiment(exp: dict, dataset: list, rag: QwenRAGSystemOptimized,
                   text_to_id: dict) -> dict:
    tag = exp['tag']
    out_path = RESULTS_DIR / f"{tag}.json"

    if out_path.exists():
        print(f"  [{tag}] 已有结果，跳过（删除 {out_path} 可重跑）")
        with open(out_path) as f:
            return json.load(f)

    print(f"\n{'='*55}")
    print(f"  [{tag}] {exp['name']}")
    print(f"  BM25={exp['use_bm25']}  FAISS={exp['use_faiss']}  "
          f"Reranker={exp['use_reranker']}  YearFilter={exp['use_year_filter']}  "
          f"recall={exp['recall']}")
    print(f"{'='*55}")

    rule_ranks, llm_ranks = [], []
    t0 = time.time()
    total = len(dataset)

    for i, q in enumerate(dataset):
        retrieved = rag.flexible_retrieve(
            q['question'], k=TOP_K,
            recall_candidates=exp['recall'],
            use_bm25=exp['use_bm25'],
            use_faiss=exp['use_faiss'],
            use_reranker=exp['use_reranker'],
            use_year_filter=exp['use_year_filter'],
        )
        gt = q['ground_truth']
        if q['type'] == 'rule_based':
            rule_ranks.append(check_rule_hit(retrieved, gt))
        else:
            llm_ranks.append(check_llm_hit(retrieved, gt, text_to_id))

        if (i + 1) % 50 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (total - i - 1)
            print(f"  {i+1}/{total}  已用 {elapsed:.0f}s  预计剩余 {eta:.0f}s")

    all_ranks = rule_ranks + llm_ranks
    result = {
        "tag": tag,
        "name": exp['name'],
        "config": exp,
        "metrics": {
            "overall":      compute_metrics(all_ranks),
            "rule_based":   compute_metrics(rule_ranks),
            "llm_generated": compute_metrics(llm_ranks),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  → 结果已保存至 {out_path}")
    return result


# ========== 主流程 ==========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=0,
                        help='随机抽取 N 题快速验证（0=全量）')
    parser.add_argument('--exp', nargs='+', default=None,
                        help='只跑指定实验，如 --exp B1 B2 B3')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATASET_PATH, encoding='utf-8') as f:
        dataset = json.load(f)

    if args.sample > 0:
        random.seed(42)
        dataset = random.sample(dataset, min(args.sample, len(dataset)))
        print(f"[抽样] {len(dataset)} 题")

    exps = EXPERIMENTS
    if args.exp:
        exps = [e for e in EXPERIMENTS if e['tag'] in args.exp]
        if not exps:
            print(f"未找到实验: {args.exp}")
            sys.exit(1)

    print("初始化 RAG 系统...")
    rag = QwenRAGSystemOptimized()
    text_to_id = {item['text'][:100]: item['id'] for item in rag.metadata}

    all_results = []
    t_total = time.time()

    for exp in exps:
        res = run_experiment(exp, dataset, rag, text_to_id)
        all_results.append(res)

    # ========== 汇总表 ==========
    print(f"\n\n{'='*75}")
    print(f"{'消融/对比实验汇总':^75}")
    print(f"{'='*75}")
    header = f"{'实验':5s}  {'配置说明':<28s}  {'Hit@1':>6}  {'Hit@3':>6}  {'Hit@5':>6}  {'MRR':>6}  {'耗时':>6}"
    print(header)
    print('-' * 75)
    for r in all_results:
        m = r['metrics']['overall']
        print(f"{r['tag']:5s}  {r['name']:<28s}  "
              f"{m.get('Hit@1',0):6.3f}  {m.get('Hit@3',0):6.3f}  "
              f"{m.get('Hit@5',0):6.3f}  {m.get('MRR',0):6.3f}  "
              f"{r.get('elapsed_s',0):5.0f}s")

    print(f"\n总耗时: {time.time()-t_total:.0f}s")

    with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
        json.dump({"experiments": all_results}, f, ensure_ascii=False, indent=2)
    print(f"汇总结果已保存至 {SUMMARY_PATH}")


if __name__ == '__main__':
    main()
