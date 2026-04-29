#!/usr/bin/env python3
"""
iGEM RAG 幻觉测试

构造三类"知识库中不存在"的问题，测试系统是否正确拒答而非编造答案。

Type A: 完全捏造的队伍名
Type B: 真实队伍 + 该队未参赛的年份
Type C: 年份越界（2014 / 2025）

评估：
  - correct_refusal：回答中含拒答表达（"无法确定"等）
  - hallucination：回答自信但答案无依据
  - uncertain：难以判断，人工复核

用法：
  cd /mnt/disk4/EscLab/VAD/igem-rag
  CUDA_VISIBLE_DEVICES="" conda run -n igem-rag python eval/hallucination_test.py
"""

import json
import os
import sys
import random
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI

os.chdir(Path(__file__).parent.parent)
sys.path.insert(0, ".")
load_dotenv()

from QwenRAGSystemOptimized import QwenRAGSystemOptimized

SEED = 42
OUTPUT_PATH = "./eval/hallucination_results.json"
random.seed(SEED)

# ========== 拒答关键词 ==========
REFUSAL_PHRASES = [
    "无法确定", "没有相关", "未找到", "不存在", "没有找到",
    "无相关信息", "无法提供", "没有记录", "无从得知", "缺乏相关",
    "根据现有信息", "暂无", "不在", "未参加", "未收录",
    "cannot", "no information", "not found", "unable to",
]

def is_refusal(answer: str) -> bool:
    a = answer.lower()
    return any(p.lower() in a for p in REFUSAL_PHRASES)


# ========== 构造测试用例 ==========

def build_fake_team_questions() -> list[dict]:
    """Type A：捏造队伍名"""
    fake_teams = [
        "ShanghaiTech_Synbio", "Beijing_BioDesign", "MIT_CRISPRLab",
        "Tsinghua_GreenCell", "NJU_SyntheticLife", "PKU_BioFactory",
        "Fudan_MetaCell", "SJTU_NanoCell", "ZJU_BioEngine",
        "HUST_SynBio2023", "CUHK_MetaGenetics", "NUS_SynthLab",
        "Tokyo_BioCircuit", "Seoul_GeneEdit", "Paris_SynCell",
        "Oxford_BioFab2099", "Harvard_SyntheticBio", "Stanford_BioCircuit",
        "ETH_MetaSynth", "Cambridge_SynCell",
    ]
    templates = [
        "{team}团队在{year}年的iGEM项目中选择了什么底盘生物？",
        "{team}团队在{year}年用到了哪些关键生物元件？",
        "{year}年{team}队的建模方法是什么？",
        "{team}团队在{year}年开展了哪些人类实践活动？",
        "{year}年{team}队遇到了哪些技术挑战？",
    ]
    years = [2018, 2019, 2020, 2021, 2022, 2023]
    questions = []
    for i, team in enumerate(fake_teams):
        year = random.choice(years)
        tmpl = templates[i % len(templates)]
        questions.append({
            "id": f"hall_A_{i:03d}",
            "type": "fake_team",
            "question": tmpl.format(team=team, year=year),
            "nonexistent": {"team": team, "year": year},
        })
    return questions


def build_wrong_year_questions(metadata_path: str) -> list[dict]:
    """Type B：真实队伍 + 该队未参赛的年份"""
    with open(metadata_path) as f:
        data = json.load(f)

    team_years = defaultdict(set)
    for d in data:
        team_years[d['metadata']['team_name']].add(d['metadata']['year'])

    all_years = set(range(2015, 2025))

    # 找有"缺席年份"的队伍（参赛至少3年但有断档）
    candidates = []
    for team, years in team_years.items():
        missing = sorted(all_years - years)
        # 只取 2016-2023 内的缺席年（不取边界）
        missing = [y for y in missing if 2016 <= y <= 2023]
        if missing and len(years) >= 2:
            candidates.append((team, sorted(years), missing))

    random.shuffle(candidates)

    templates = [
        "{team}团队在{year}年的iGEM项目研究了什么课题？",
        "{year}年{team}队的湿实验使用了什么实验方法？",
        "{team}团队在{year}年的建模工作是怎样的？",
        "{year}年{team}队做了哪些人类实践？",
        "{team}团队在{year}年取得了什么成果？",
    ]

    questions = []
    for i, (team, real_years, missing_years) in enumerate(candidates[:20]):
        year = random.choice(missing_years)
        tmpl = templates[i % len(templates)]
        questions.append({
            "id": f"hall_B_{i:03d}",
            "type": "wrong_year",
            "question": tmpl.format(team=team, year=year),
            "nonexistent": {
                "team": team,
                "year": year,
                "real_years": real_years,
            },
        })
    return questions


def build_out_of_range_questions() -> list[dict]:
    """Type C：完全超出知识库年份范围（2014 / 2025）"""
    # 用真实队名+越界年份，更有迷惑性
    real_teams = [
        "Waterloo", "Aachen", "Oxford", "EPFL", "Fudan",
        "Leiden", "Vilnius-Lithuania", "DTU-Denmark", "Paris_Bettencourt", "Mingdao",
    ]
    templates_2014 = [
        "{team}团队在2014年的iGEM项目中做了什么研究？",
        "2014年{team}队参加iGEM时用了什么底盘生物？",
        "{team}在2014年的建模方法是什么？",
    ]
    templates_2025 = [
        "{team}团队在2025年的iGEM项目是什么？",
        "2025年{team}队有什么创新技术？",
        "{team}在2025年的人类实践活动有哪些？",
    ]
    questions = []
    for i, team in enumerate(real_teams):
        if i < 5:
            tmpl = templates_2014[i % len(templates_2014)]
            year = 2014
        else:
            tmpl = templates_2025[i % len(templates_2025)]
            year = 2025
        questions.append({
            "id": f"hall_C_{i:03d}",
            "type": "out_of_range",
            "question": tmpl.format(team=team),
            "nonexistent": {"team": team, "year": year},
        })
    return questions


# ========== LLM 二次判断（针对模糊情况） ==========

JUDGE_PROMPT = """\
以下是一个 RAG 系统对"知识库中不存在的内容"的回答。
请判断系统是否产生了幻觉（即自信地给出了无依据的答案）。

问题：{question}
系统回答（前500字）：{answer}

请只输出以下三个词之一：
- hallucination：系统自信地给出了答案，但答案明显没有依据或是编造的
- refusal：系统明确表示无法确定、没有相关信息或委婉拒绝
- uncertain：无法判断"""

def llm_judge(question: str, answer: str, client: OpenAI) -> str:
    try:
        resp = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                question=question, answer=answer[:500]
            )}],
            max_tokens=10,
            temperature=0,
        )
        verdict = resp.choices[0].message.content.strip().lower()
        if "hallucination" in verdict: return "hallucination"
        if "refusal" in verdict: return "refusal"
        return "uncertain"
    except Exception as e:
        return "uncertain"


# ========== 主流程 ==========

def main():
    print("构造测试用例...")
    questions = (
        build_fake_team_questions() +
        build_wrong_year_questions("./out_embedding/chunk_metadata.json") +
        build_out_of_range_questions()
    )
    print(f"  Type A (捏造队名): {sum(1 for q in questions if q['type']=='fake_team')} 题")
    print(f"  Type B (真实队+错误年份): {sum(1 for q in questions if q['type']=='wrong_year')} 题")
    print(f"  Type C (年份越界): {sum(1 for q in questions if q['type']=='out_of_range')} 题")
    print(f"  合计: {len(questions)} 题")

    print("\n初始化 RAG 系统...")
    rag = QwenRAGSystemOptimized()

    results = []
    counts = {"hallucination": 0, "refusal": 0, "uncertain": 0}
    type_counts = defaultdict(lambda: {"hallucination": 0, "refusal": 0, "uncertain": 0})

    for i, q in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {q['type']} | {q['question'][:50]}...")
        result = rag.query(q['question'])
        answer = result['answer']

        # 第一层：关键词检测
        if is_refusal(answer):
            verdict = "refusal"
        else:
            # 第二层：LLM 判断
            verdict = llm_judge(q['question'], answer, rag.client)

        counts[verdict] += 1
        type_counts[q['type']][verdict] += 1

        results.append({
            **q,
            "answer_preview": answer[:300],
            "verdict": verdict,
        })

    # ========== 打印结果 ==========
    total = len(questions)
    print("\n" + "=" * 50)
    print("幻觉测试结果")
    print("=" * 50)
    print(f"  正确拒答 (refusal):    {counts['refusal']:3d} / {total}  ({counts['refusal']/total:.1%})")
    print(f"  幻觉     (hallucination): {counts['hallucination']:3d} / {total}  ({counts['hallucination']/total:.1%})")
    print(f"  待人工复核 (uncertain): {counts['uncertain']:3d} / {total}  ({counts['uncertain']/total:.1%})")

    print("\n  --- 按类型细分 ---")
    for t, c in sorted(type_counts.items()):
        n = sum(c.values())
        print(f"  {t} (n={n}): "
              f"拒答={c['refusal']} ({c['refusal']/n:.0%})  "
              f"幻觉={c['hallucination']} ({c['hallucination']/n:.0%})  "
              f"待复核={c['uncertain']} ({c['uncertain']/n:.0%})")

    print("\n  --- 幻觉样例 ---")
    for r in results:
        if r['verdict'] == 'hallucination':
            print(f"  [{r['type']}] {r['question']}")
            print(f"  回答预览: {r['answer_preview'][:150]}")
            print()

    # 保存
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total": total,
                "counts": counts,
                "by_type": {t: dict(c) for t, c in type_counts.items()},
                "hallucination_rate": round(counts['hallucination'] / total, 4),
                "refusal_rate": round(counts['refusal'] / total, 4),
            },
            "details": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"详细结果已保存至 {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
