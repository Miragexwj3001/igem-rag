#!/usr/bin/env python3
"""
iGEM RAG 评估脚本

指标：Hit@1, Hit@3, Hit@5, MRR
分组：整体 / 规则化题 / LLM生成题 / 各子模块

用法：
  cd /mnt/disk4/EscLab/VAD/igem-rag
  python eval/run_eval.py              # 全量评估
  python eval/run_eval.py --sample 20  # 快速抽样验证
"""

import json
import sys
import os
import time
import argparse
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv

# 切换到项目根目录（RAG 系统依赖相对路径）
os.chdir(Path(__file__).parent.parent)
sys.path.insert(0, ".")
load_dotenv()

from QwenRAGSystemOptimized import QwenRAGSystemOptimized

DATASET_PATH = "./eval/eval_dataset.json"
RESULTS_PATH = "./eval/eval_results.json"
TOP_K = 10  # 检索返回数，评估 @1 @3 @5 都在这个范围内


# ========== 规范化（与生成脚本保持一致） ==========

def normalize_section(section: str) -> str | None:
    s = section.lower()
    if any(k in s for k in ['wet lab', 'wetlab', '湿实验']): return 'Wet Lab'
    if any(k in s for k in ['建模', 'model', 'dry lab', '干实验']): return 'Model'
    if any(k in s for k in ['human practices', 'humanpractices', '人类实践', '人文实践',
                             '人力实践', '人本实践', '公众实践']): return 'Human Practices'
    return None


def normalize_subsection(sub: str, sec: str) -> str | None:
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


# ========== 命中判断 ==========

def check_rule_based_hit(retrieved: list, gt: dict) -> int | None:
    """
    规则化题：在 top-K 里找到 (team_name, year, canon_subsection) 完全匹配的 chunk
    返回命中排名（1-indexed），未命中返回 None
    """
    for rank, chunk in enumerate(retrieved, start=1):
        meta = chunk['metadata']
        r_sec = normalize_section(meta.get('Section', '') or '')
        r_sub = normalize_subsection(meta.get('Subsection', '') or '', r_sec) if r_sec else None
        if (meta.get('team_name') == gt['team_name']
                and meta.get('year') == gt['year']
                and r_sub == gt['canon_subsection']):
            return rank
    return None


def check_llm_hit(retrieved: list, gt: dict, text_to_id: dict) -> int | None:
    """
    LLM生成题：在 top-K 里找到 chunk_id 匹配的 chunk
    返回命中排名（1-indexed），未命中返回 None
    """
    for rank, chunk in enumerate(retrieved, start=1):
        cid = text_to_id.get(chunk['text'][:100])
        if cid == gt['chunk_id']:
            return rank
    return None


# ========== 指标计算 ==========

def compute_metrics(hit_ranks: list, k_list=(1, 3, 5)) -> dict:
    n = len(hit_ranks)
    if n == 0:
        return {}
    metrics = {f'Hit@{k}': round(sum(1 for r in hit_ranks if r and r <= k) / n, 4)
               for k in k_list}
    metrics['MRR'] = round(sum(1 / r for r in hit_ranks if r) / n, 4)
    metrics['n'] = n
    return metrics


def print_metrics(label: str, metrics: dict):
    print(f"  [{label}]  n={metrics.get('n',0)}"
          f"  Hit@1={metrics.get('Hit@1', 0):.3f}"
          f"  Hit@3={metrics.get('Hit@3', 0):.3f}"
          f"  Hit@5={metrics.get('Hit@5', 0):.3f}"
          f"  MRR={metrics.get('MRR', 0):.3f}")


# ========== 主流程 ==========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=0,
                        help='随机抽取 N 道题快速验证（0 = 全量）')
    args = parser.parse_args()

    with open(DATASET_PATH, encoding='utf-8') as f:
        dataset = json.load(f)

    if args.sample > 0:
        import random; random.seed(42)
        dataset = random.sample(dataset, min(args.sample, len(dataset)))
        print(f"[抽样模式] 随机抽取 {len(dataset)} 道题")

    print("初始化 RAG 系统...")
    rag = QwenRAGSystemOptimized()

    # 构建 text[:100] → chunk_id 查找表
    text_to_id = {item['text'][:100]: item['id'] for item in rag.metadata}

    rule_ranks, llm_ranks = [], []
    details = []

    total = len(dataset)
    t0 = time.time()

    for i, q in enumerate(dataset):
        retrieved = rag.hybrid_retrieve(q['question'], k=TOP_K)
        gt = q['ground_truth']

        if q['type'] == 'rule_based':
            rank = check_rule_based_hit(retrieved, gt)
            rule_ranks.append(rank)
        else:
            rank = check_llm_hit(retrieved, gt, text_to_id)
            llm_ranks.append(rank)

        details.append({
            'id': q['id'],
            'type': q['type'],
            'question': q['question'],
            'hit_rank': rank,
            'ground_truth': gt,
        })

        # 进度
        if (i + 1) % 20 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (total - i - 1)
            print(f"  {i+1}/{total}  已用 {elapsed:.0f}s  预计剩余 {eta:.0f}s")

    # ========== 打印结果 ==========
    all_ranks = rule_ranks + llm_ranks
    print("\n" + "=" * 50)
    print("评估结果")
    print("=" * 50)
    print_metrics("整体", compute_metrics(all_ranks))
    print_metrics("规则化题", compute_metrics(rule_ranks))
    print_metrics("LLM生成题", compute_metrics(llm_ranks))

    # 按子模块细分（规则化题）
    rule_qs = [q for q in dataset if q['type'] == 'rule_based']
    sub_ranks = defaultdict(list)
    for q, r in zip(rule_qs, rule_ranks):
        sub_ranks[q['ground_truth']['canon_subsection']].append(r)

    print("\n  --- 规则化题 按子模块 ---")
    for sub in sorted(sub_ranks):
        m = compute_metrics(sub_ranks[sub])
        print(f"  {sub:<18}  n={m['n']:3d}  Hit@5={m['Hit@5']:.3f}  MRR={m['MRR']:.3f}")

    # 按大模块细分（LLM生成题）
    llm_qs = [q for q in dataset if q['type'] == 'llm_generated']
    sec_ranks = defaultdict(list)
    for q, r in zip(llm_qs, llm_ranks):
        sec_ranks[q['ground_truth']['canon_section']].append(r)

    print("\n  --- LLM生成题 按大模块 ---")
    for sec in sorted(sec_ranks):
        m = compute_metrics(sec_ranks[sec])
        print(f"  {sec:<20}  n={m['n']:3d}  Hit@5={m['Hit@5']:.3f}  MRR={m['MRR']:.3f}")

    # ========== 保存 ==========
    output = {
        'summary': {
            'total': len(all_ranks),
            'overall': compute_metrics(all_ranks),
            'rule_based': compute_metrics(rule_ranks),
            'llm_generated': compute_metrics(llm_ranks),
            'by_subsection': {sub: compute_metrics(ranks) for sub, ranks in sub_ranks.items()},
            'by_section_llm': {sec: compute_metrics(ranks) for sec, ranks in sec_ranks.items()},
        },
        'details': details,
    }
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存至 {RESULTS_PATH}")


if __name__ == '__main__':
    main()
