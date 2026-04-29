#!/usr/bin/env python3
"""
iGEM RAG 生成质量评估 (Phase 3)

对 50 道题跑完整 RAG pipeline（检索+生成），用 LLM-as-Judge 评估：
  - Faithfulness（忠实度，1-5）：回答是否有上下文支撑
  - Relevance（相关性，1-5）：回答是否切题
  - Citation Accuracy（引用准确率，%）：引用来源是否实际存在于检索结果中

用法:
  cd /mnt/disk4/EscLab/VAD/igem-rag
  CUDA_VISIBLE_DEVICES="" conda run -n igem-rag python eval/generation_eval.py
"""

import json
import os
import sys
import re
import time
import random
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

os.chdir(Path(__file__).parent.parent)
sys.path.insert(0, ".")
load_dotenv()

from QwenRAGSystemOptimized import QwenRAGSystemOptimized

SEED = 42
SAMPLE_RULE = 25
SAMPLE_LLM = 25
DATASET_PATH = "./eval/eval_dataset.json"
OUTPUT_PATH = "./eval/generation_results.json"

random.seed(SEED)

# ========== LLM 评审 Prompt ==========

JUDGE_PROMPT = """\
你是一名评估 RAG 系统输出质量的专业评审。请根据以下信息对系统回答打分。

## 检索到的上下文（系统实际使用的文档片段）：
{context}

## 用户问题：
{question}

## 系统回答：
{answer}

请从两个维度打分，严格只输出 JSON，不要有其他文字：

忠实度（faithfulness，1-5）：回答中每项陈述是否都有上下文支撑？
  5=全部有据 4=绝大部分有据，少量通用知识已标注 3=大部分有据，少量无来源
  2=约一半无依据 1=大量内容凭空生成或与上下文矛盾

相关性（relevance，1-5）：回答是否切题有效解答了问题？
  5=完整直接回答 4=基本回答，细节略不足 3=部分回答有遗漏
  2=回答偏题 1=未回答问题

输出格式（只输出这一行JSON）：
{{"faithfulness": <1-5>, "relevance": <1-5>, "faith_reason": "<10-30字理由>", "rel_reason": "<10-30字理由>"}}"""


def llm_judge(question: str, context: str, answer: str, client) -> dict:
    """调用 qwen-plus 评审回答质量"""
    prompt = JUDGE_PROMPT.format(
        context=context[:3000],
        question=question,
        answer=answer[:1500],
    )
    try:
        resp = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
        # 提取 JSON
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"faithfulness": None, "relevance": None, "faith_reason": raw, "rel_reason": ""}
    except Exception as e:
        return {"faithfulness": None, "relevance": None, "faith_reason": str(e), "rel_reason": ""}


# ========== Citation Accuracy ==========

def extract_citations(answer: str) -> list[tuple[str, int]]:
    """从回答中提取 [来源：TeamName YYYY] 格式的引用"""
    pattern = r'[来\[【]源[：:]?\s*([^\s\]】]+)\s+(\d{4})[^\d]'
    matches = re.findall(pattern, answer)
    result = []
    for team, year in matches:
        try:
            result.append((team.strip(), int(year)))
        except ValueError:
            pass
    return result


def check_citation_accuracy(answer: str, retrieved_chunks: list) -> dict:
    """验证回答中的引用来源是否在检索结果中"""
    cited = extract_citations(answer)
    if not cited:
        return {"cited_count": 0, "valid_count": 0, "accuracy": None, "detail": "无引用"}

    valid_sources = {
        (c['metadata'].get('team_name', ''), c['metadata'].get('year', 0))
        for c in retrieved_chunks
    }

    valid = [(t, y) for t, y in cited if (t, y) in valid_sources]
    return {
        "cited_count": len(cited),
        "valid_count": len(valid),
        "accuracy": round(len(valid) / len(cited), 4) if cited else None,
        "detail": f"引用 {len(cited)} 处，{len(valid)} 处在检索结果中",
    }


# ========== 采样 ==========

def sample_questions(dataset: list) -> list:
    rule = [q for q in dataset if q['type'] == 'rule_based']
    llm = [q for q in dataset if q['type'] == 'llm_generated']
    sampled = random.sample(rule, min(SAMPLE_RULE, len(rule))) + \
              random.sample(llm, min(SAMPLE_LLM, len(llm)))
    random.shuffle(sampled)
    return sampled


# ========== 主流程 ==========

def main():
    print("加载数据集...")
    with open(DATASET_PATH, encoding='utf-8') as f:
        dataset = json.load(f)

    questions = sample_questions(dataset)
    print(f"  采样 {len(questions)} 题（rule={sum(1 for q in questions if q['type']=='rule_based')}，"
          f"llm={sum(1 for q in questions if q['type']=='llm_generated')}）")

    print("初始化 RAG 系统...")
    rag = QwenRAGSystemOptimized()

    results = []
    scores = {"faithfulness": [], "relevance": [], "citation_accuracy": []}
    type_scores = defaultdict(lambda: {"faithfulness": [], "relevance": [], "citation_accuracy": []})

    total = len(questions)
    t0 = time.time()

    for i, q in enumerate(questions):
        print(f"  [{i+1}/{total}] {q['type']} | {q['question'][:60]}...")

        # 1. 检索
        chunks = rag.hybrid_retrieve(q['question'], k=8)
        context = "\n\n".join(
            f"[来源：{c['metadata']['team_name']} {c['metadata']['year']}]\n{c['text']}"
            for c in chunks
        )

        # 2. 生成
        prompt = rag.generate_prompt(q['question'], chunks)
        try:
            answer = rag.call_qwen_api(prompt)
        except Exception as e:
            print(f"    生成失败: {e}")
            answer = ""

        # 3. LLM 评审
        judgment = llm_judge(q['question'], context, answer, rag.client)

        # 4. 引用准确率
        citation = check_citation_accuracy(answer, chunks)

        # 收集分数
        if judgment.get('faithfulness'):
            scores['faithfulness'].append(judgment['faithfulness'])
            type_scores[q['type']]['faithfulness'].append(judgment['faithfulness'])
        if judgment.get('relevance'):
            scores['relevance'].append(judgment['relevance'])
            type_scores[q['type']]['relevance'].append(judgment['relevance'])
        if citation['accuracy'] is not None:
            scores['citation_accuracy'].append(citation['accuracy'])
            type_scores[q['type']]['citation_accuracy'].append(citation['accuracy'])

        results.append({
            "id": q['id'],
            "type": q['type'],
            "question": q['question'],
            "answer": answer,
            "answer_preview": answer[:400],
            "retrieved_sources": [
                {"team": c['metadata'].get('team_name'), "year": c['metadata'].get('year')}
                for c in chunks
            ],
            "judgment": judgment,
            "citation": citation,
        })

        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (total - i - 1)
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"    已用 {elapsed:.0f}s，预计剩余 {eta:.0f}s")

    # ========== 汇总 ==========
    def avg(lst): return round(sum(lst) / len(lst), 3) if lst else None

    summary = {
        "total": total,
        "overall": {
            "faithfulness":     avg(scores['faithfulness']),
            "relevance":        avg(scores['relevance']),
            "citation_accuracy": avg(scores['citation_accuracy']),
        },
        "by_type": {
            t: {
                "faithfulness":     avg(v['faithfulness']),
                "relevance":        avg(v['relevance']),
                "citation_accuracy": avg(v['citation_accuracy']),
                "n": len(v['faithfulness']),
            }
            for t, v in type_scores.items()
        }
    }

    print("\n" + "=" * 50)
    print("生成质量评估结果")
    print("=" * 50)
    print(f"  Faithfulness:      {summary['overall']['faithfulness']} / 5")
    print(f"  Relevance:         {summary['overall']['relevance']} / 5")
    print(f"  Citation Accuracy: {summary['overall']['citation_accuracy']:.1%}" if summary['overall']['citation_accuracy'] else "  Citation Accuracy: N/A")
    print("\n  --- 按题型细分 ---")
    for t, m in summary['by_type'].items():
        print(f"  {t} (n={m['n']}):  "
              f"Faithfulness={m['faithfulness']}  "
              f"Relevance={m['relevance']}  "
              f"Citation={m['citation_accuracy']:.1%}" if m['citation_accuracy'] else
              f"  {t} (n={m['n']}):  "
              f"Faithfulness={m['faithfulness']}  "
              f"Relevance={m['relevance']}")

    # 低分样例
    low_faith = [r for r in results if r['judgment'].get('faithfulness') and r['judgment']['faithfulness'] <= 2]
    low_rel = [r for r in results if r['judgment'].get('relevance') and r['judgment']['relevance'] <= 2]
    if low_faith:
        print(f"\n  低忠实度样例（{len(low_faith)} 题）：")
        for r in low_faith[:3]:
            print(f"    [{r['type']}] {r['question'][:60]}")
            print(f"    → Faithfulness={r['judgment']['faithfulness']}  原因：{r['judgment']['faith_reason']}")
    if low_rel:
        print(f"\n  低相关性样例（{len(low_rel)} 题）：")
        for r in low_rel[:3]:
            print(f"    [{r['type']}] {r['question'][:60]}")
            print(f"    → Relevance={r['judgment']['relevance']}  原因：{r['judgment']['rel_reason']}")

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存至 {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
