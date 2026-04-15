#!/usr/bin/env python3
"""
iGEM RAG 评估数据集生成脚本

生成两类问题：
  - 规则化题 (~200题)：填模板，ground truth = (team, year, canonical_subsection)
  - LLM生成题 (~200题)：让 Qwen 从 chunk 反向生成问题，ground truth = chunk_id

用法：
  cd /mnt/disk4/EscLab/VAD/igem-rag
  python eval/generate_dataset.py
"""

import json
import random
import os
import sys
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# ========== 配置 ==========
SEED = 42
RULE_BASED_TARGET = 200    # 规则化题目标数
LLM_GENERATED_TARGET = 200  # LLM生成题目标数
MIN_CHUNK_TEXT_LEN = 150    # chunk 正文最短字符数（过滤太短的）
METADATA_PATH = "./out_embedding/chunk_metadata.json"
OUTPUT_PATH = "./eval/eval_dataset.json"

random.seed(SEED)


# ========== Section/Subsection 规范化 ==========

def normalize_section(section: str) -> str | None:
    s = section.lower()
    if any(k in s for k in ['wet lab', 'wetlab', '湿实验']):
        return 'Wet Lab'
    if any(k in s for k in ['建模', 'model', 'dry lab', '干实验']):
        return 'Model'
    if any(k in s for k in ['human practices', 'humanpractices', '人类实践', '人文实践',
                             '人力实践', '人本实践', '公众实践']):
        return 'Human Practices'
    return None


def normalize_subsection(subsection: str, section: str) -> str | None:
    sub = subsection.strip()

    if section == 'Wet Lab':
        if '底盘生物' in sub:
            return '底盘生物的选择'
        if '关键' in sub and ('元件' in sub or '组件' in sub):
            return '关键生物元件与表达产物'
        if '创新' in sub:
            return '实验方法创新点'
        if '验证方法' in sub or ('验证' in sub and '结果' not in sub):
            return '实验验证方法'
        if '验证结果' in sub or ('结果' in sub and '验证' in sub):
            return '实验验证结果'
        if '技术挑战' in sub:
            return '技术挑战及解决方案'
        if '实验设计' in sub:
            return '实验设计分析'

    elif section == 'Model':
        if '建模目' in sub or '模型目' in sub:
            return '建模目的'
        if '模型类型' in sub:
            return '模型类型'
        if '输入参数' in sub or ('假设' in sub and '条件' in sub):
            return '输入参数与假设'
        if '吻合' in sub or '一致' in sub or '符合' in sub:
            return '预测与实验数据吻合度'
        if '开源' in sub or '代码' in sub:
            return '开源代码'

    elif section == 'Human Practices':
        if '社会需求' in sub:
            return '社会需求调研方法'
        if '利益相关' in sub:
            return '利益相关方合作'
        if '公众推广' in sub or '公众科普' in sub or '科普活动' in sub or '推广活动' in sub:
            return '公众推广活动'
        if '伦理' in sub:
            return '伦理审查'
        if '具体案例' in sub or '影响力' in sub or '影响数据' in sub:
            return '具体案例与影响'

    return None


# ========== 规则化问题模板 ==========

TEMPLATES = {
    'Wet Lab': {
        '底盘生物的选择': [
            "{team}团队在{year}年的iGEM项目中选择了什么底盘生物？选择理由是什么？",
            "{year}年{team}队的湿实验中，底盘生物的选择策略是怎样的？",
        ],
        '关键生物元件与表达产物': [
            "{team}团队在{year}年的项目中用到了哪些关键生物元件？",
            "{year}年{team}队的湿实验涉及哪些核心遗传元件和表达产物？",
        ],
        '实验方法创新点': [
            "{team}团队在{year}年的湿实验中有哪些实验方法上的创新？",
            "{year}年{team}队在实验技术层面做出了什么创新？",
        ],
        '实验验证方法': [
            "{team}团队在{year}年采用了哪些方法来验证其实验设计？",
            "{year}年{team}队的湿实验验证方案是怎样的？",
        ],
        '实验验证结果': [
            "{team}团队在{year}年的实验得到了哪些关键结果？",
            "{year}年{team}队的湿实验验证结果如何，是否达到预期目标？",
        ],
        '技术挑战及解决方案': [
            "{team}团队在{year}年的实验中遇到了哪些技术挑战？是如何解决的？",
            "{year}年{team}队在湿实验过程中面对了什么困难，采取了什么应对措施？",
        ],
        '实验设计分析': [
            "{team}团队在{year}年是如何设计其湿实验方案的？",
            "{year}年{team}队的整体湿实验设计思路是什么？",
        ],
    },
    'Model': {
        '建模目的': [
            "{team}团队在{year}年为什么要建立数学模型？核心目标是什么？",
            "{year}年{team}队的计算建模是为了解决什么科学问题？",
        ],
        '模型类型': [
            "{team}团队在{year}年使用了哪种类型的数学模型？",
            "{year}年{team}队建立了什么类型的计算模型来支持实验设计？",
        ],
        '输入参数与假设': [
            "{team}团队在{year}年建模时设定了哪些关键输入参数和假设条件？",
            "{year}年{team}队的模型中包含哪些核心参数和前提假设？",
        ],
        '预测与实验数据吻合度': [
            "{team}团队在{year}年的模型预测结果与实验数据吻合程度如何？",
            "{year}年{team}队如何验证其模型预测和真实实验数据的一致性？",
        ],
        '开源代码': [
            "{team}团队在{year}年是否开源了建模代码？使用了什么语言或工具？",
            "{year}年{team}队的计算建模代码是否对外公开？",
        ],
    },
    'Human Practices': {
        '社会需求调研方法': [
            "{team}团队在{year}年如何开展社会需求调研？采用了哪些方法？",
            "{year}年{team}队通过什么方式了解公众对其项目的需求？",
        ],
        '利益相关方合作': [
            "{team}团队在{year}年与哪些利益相关方进行了合作或访谈？",
            "{year}年{team}队在人类实践中接触了哪些关键合作方？",
        ],
        '公众推广活动': [
            "{team}团队在{year}年开展了哪些公众科普或推广活动？",
            "{year}年{team}队如何向社会公众传播合成生物学知识？",
        ],
        '伦理审查': [
            "{team}团队在{year}年如何处理项目的伦理问题？经过了哪些审查流程？",
            "{year}年{team}队对项目进行了怎样的伦理评估与生物安全审查？",
        ],
        '具体案例与影响': [
            "{team}团队在{year}年的人类实践产生了哪些具体的社会影响？",
            "{year}年{team}队的人类实践工作有哪些可量化的成效？",
        ],
    },
}

# LLM 生成题提示词
LLM_QUESTION_PROMPT = """\
你是一位 iGEM 合成生物学领域的研究者。请根据以下文段内容，提出一个自然的问题。

要求：
1. 问题要像真实研究者会问的，语气自然，不要生硬
2. 问题应聚焦这段文字的核心内容，而不是边缘细节
3. 不要在问题中直接复制文段里的原话或专业术语
4. 不要在问题中直接提及团队名称和年份（让检索系统自己去匹配）
5. 问题必须能从这段文字中找到答案

文段内容：
{chunk_text}

只输出问题本身，不要加任何前缀或解释。"""


# ========== 主逻辑 ==========

def load_and_normalize_chunks(metadata_path: str):
    """加载并规范化所有 chunk"""
    with open(metadata_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    normalized = []
    for item in raw:
        sec_raw = item["metadata"].get("Section", "")
        sub_raw = item["metadata"].get("Subsection", "")
        text = item.get("text", "")

        # 提取正文（去掉【模块】行）
        body = text
        if "【正文】:" in text:
            body = text.split("【正文】:")[1].strip()
        elif "【正文】：" in text:
            body = text.split("【正文】：")[1].strip()

        if len(body) < MIN_CHUNK_TEXT_LEN:
            continue

        canon_sec = normalize_section(sec_raw)
        if canon_sec is None:
            continue

        canon_sub = normalize_subsection(sub_raw, canon_sec)

        normalized.append({
            "chunk_id": item["id"],
            "text": body,
            "team_name": item["metadata"]["team_name"],
            "year": item["metadata"]["year"],
            "canon_section": canon_sec,
            "canon_subsection": canon_sub,  # None 表示无法规范化
        })

    return normalized


def build_rule_based_questions(chunks, target: int) -> list[dict]:
    """规则化题：按 (canon_section, canon_subsection) 均匀抽样并填模板"""
    # 建索引：(section, subsection) -> [chunk, ...]
    index = defaultdict(list)
    for c in chunks:
        if c["canon_subsection"] is not None:
            key = (c["canon_section"], c["canon_subsection"])
            index[key].append(c)

    valid_keys = [k for k in index if k[1] in TEMPLATES.get(k[0], {})]
    print(f"[规则化] 可用 (section, subsection) 组合数：{len(valid_keys)}")

    # 均匀分配配额
    per_group = max(1, target // len(valid_keys))
    questions = []
    qid = 0

    for key in sorted(valid_keys):
        sec, sub = key
        pool = index[key]
        sample_size = min(per_group, len(pool))
        sampled = random.sample(pool, sample_size)
        templates = TEMPLATES[sec][sub]

        for chunk in sampled:
            tmpl = random.choice(templates)
            question = tmpl.format(team=chunk["team_name"], year=chunk["year"])
            questions.append({
                "id": f"rule_{qid:04d}",
                "question": question,
                "type": "rule_based",
                "difficulty": "easy",
                "ground_truth": {
                    "team_name": chunk["team_name"],
                    "year": chunk["year"],
                    "canon_section": sec,
                    "canon_subsection": sub,
                    # 参考 chunk_id（同队同年同子模块可能有多个 chunk，评估时用元数据匹配）
                    "ref_chunk_id": chunk["chunk_id"],
                },
            })
            qid += 1

    # 若总量超标则随机截断
    random.shuffle(questions)
    return questions[:target]


def build_llm_questions(chunks, target: int, client: OpenAI) -> list[dict]:
    """LLM生成题：按 canon_section 均匀抽样，调 Qwen 生成问题"""
    # 按 section 分组（不限 subsection）
    by_section = defaultdict(list)
    for c in chunks:
        by_section[c["canon_section"]].append(c)

    sections = list(by_section.keys())
    per_section = target // len(sections)
    remainder = target % len(sections)

    questions = []
    qid = 0

    for i, sec in enumerate(sections):
        quota = per_section + (1 if i < remainder else 0)
        pool = by_section[sec]
        sampled = random.sample(pool, min(quota, len(pool)))

        print(f"[LLM生成] {sec}: 生成 {len(sampled)} 道题...")

        for chunk in sampled:
            prompt = LLM_QUESTION_PROMPT.format(chunk_text=chunk["text"][:800])
            try:
                resp = client.chat.completions.create(
                    model="qwen-plus",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.7,
                )
                question = resp.choices[0].message.content.strip()
                # 简单过滤：太短或没有问号的跳过
                if len(question) < 10 or "？" not in question and "?" not in question:
                    print(f"  跳过（问题不合格）: {question[:50]}")
                    continue
            except Exception as e:
                print(f"  API 调用失败: {e}")
                continue

            questions.append({
                "id": f"llm_{qid:04d}",
                "question": question,
                "type": "llm_generated",
                "difficulty": "medium",
                "ground_truth": {
                    "chunk_id": chunk["chunk_id"],
                    "team_name": chunk["team_name"],
                    "year": chunk["year"],
                    "canon_section": sec,
                    "canon_subsection": chunk["canon_subsection"],
                },
            })
            qid += 1
            if qid % 20 == 0:
                print(f"  已生成 {qid} 道 LLM 题...")

    return questions


def main():
    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误：未找到 DASHSCOPE_API_KEY，请检查 .env 文件")
        sys.exit(1)

    client = OpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key,
    )

    print("=== 加载并规范化 chunks ===")
    chunks = load_and_normalize_chunks(METADATA_PATH)
    print(f"有效 chunks 总数：{len(chunks)}")

    # 按 section 统计
    by_sec = defaultdict(int)
    for c in chunks:
        by_sec[c["canon_section"]] += 1
    for sec, cnt in sorted(by_sec.items()):
        print(f"  {sec}: {cnt}")

    print("\n=== 生成规则化题 ===")
    rule_qs = build_rule_based_questions(chunks, RULE_BASED_TARGET)
    print(f"规则化题实际生成：{len(rule_qs)} 道")

    print("\n=== 生成 LLM 题 ===")
    llm_qs = build_llm_questions(chunks, LLM_GENERATED_TARGET, client)
    print(f"LLM生成题实际生成：{len(llm_qs)} 道")

    # 合并并统计
    all_questions = rule_qs + llm_qs
    print(f"\n=== 数据集总计：{len(all_questions)} 道题 ===")

    # 保存
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)
    print(f"已保存至 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
