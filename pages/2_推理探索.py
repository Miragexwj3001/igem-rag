import json
import re
from statistics import mean
from typing import Dict, List

import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from ui_core import (
    PROJ_COL,
    SUMMARY_COL,
    TEAM_COL,
    TRACK_COL,
    WIKI_COL,
    detect_user_lang,
    get_rag_system,
    get_similarity_artifacts,
    inject_styles,
    load_teams_table,
    normalize_list,
    normalize_text,
    render_top_nav,
    sidebar_global_api_key,
)

st.set_page_config(page_title="推理探索", page_icon="🧠", layout="wide")

TEAM_STRENGTH_OPTIONS = [
    "wet_lab_strong",
    "modeling_strong",
    "hardware_strong",
    "software_strong",
    "hp_strong",
    "storytelling_strong",
]

FOCUS_OPTIONS = [
    "experiment",
    "human_practices",
    "implementation",
    "entrepreneurship",
    "software",
    "hardware",
    "balanced",
]


def extract_intent(brief: str, strengths: List[str], goals: List[str]) -> List[str]:
    en_terms = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", brief.lower())
    zh_terms = re.findall(r"[\u4e00-\u9fff]{2,}", brief)
    stop = {"project", "igem", "using", "with", "for", "and", "the", "this", "that", "build"}
    en_terms = [w for w in en_terms if w not in stop]
    terms = []
    for t in en_terms + zh_terms + goals + strengths:
        if t and t not in terms:
            terms.append(t)
    return terms[:14]


def rank_projects(df: pd.DataFrame, query_text: str, keywords: List[str]) -> List[Dict]:
    vec, mat = get_similarity_artifacts(df)
    q = vec.transform([query_text])
    sims = cosine_similarity(q, mat).flatten()

    local = df.copy()
    local["sim"] = sims

    out = []
    for (team, year_num, proj), g in local.groupby([TEAM_COL, "year_num", PROJ_COL], dropna=False):
        sim_vals = sorted(g["sim"].tolist(), reverse=True)
        max_sim = sim_vals[0] if sim_vals else 0.0
        avg_top3 = mean(sim_vals[:3]) if sim_vals else 0.0
        full_text = " ".join(g["search_text"].astype(str).tolist()).lower()
        hit_cnt = sum(1 for k in keywords if str(k).lower() in full_text)
        keyword_bonus = min(hit_cnt / 8.0, 1.0)
        score = 0.6 * max_sim + 0.25 * avg_top3 + 0.15 * keyword_bonus

        out.append(
            {
                "project_key": f"{team}__{int(year_num) if pd.notna(year_num) else ''}__{proj}",
                "team_name": str(team),
                "year": int(year_num) if pd.notna(year_num) else None,
                "project_title": str(proj),
                "track": str(g[TRACK_COL].iloc[0]),
                "wiki_url": str(g[WIKI_COL].iloc[0]) if WIKI_COL in g.columns else "",
                "summary": normalize_text(str(g[SUMMARY_COL].iloc[0])),
                "score": round(float(score), 4),
            }
        )

    return sorted(out, key=lambda x: x["score"], reverse=True)[:6]


def select_evidence(rag, top_projects: List[Dict], query_text: str) -> List[Dict]:
    chunks = rag.hybrid_retrieve(query_text, k=36, recall_candidates=100)
    wanted = {(p["team_name"], p["year"]) for p in top_projects}

    evidence = []
    ev_idx = 1
    seen = set()
    for c in chunks:
        md = c.get("metadata", {})
        key = (md.get("team_name"), md.get("year"))
        text = normalize_text(c.get("text", ""))
        if key not in wanted or not text:
            continue
        uniq = (key, text[:140])
        if uniq in seen:
            continue
        seen.add(uniq)
        evidence.append(
            {
                "evidence_id": f"ev{ev_idx}",
                "team_name": md.get("team_name"),
                "year": md.get("year"),
                "page_title": md.get("page_title") or md.get("filename") or "Unknown",
                "quote": text[:420],
                "source_url": md.get("source_url") or "",
            }
        )
        ev_idx += 1
        if len(evidence) >= 20:
            break

    return evidence


def extract_json_block(text: str) -> Dict | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass

    for patt in [r"```json\s*(\{.*?\})\s*```", r"(\{.*\})"]:
        m = re.search(patt, text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    return None


def call_planner_llm(rag, request: Dict, top_projects: List[Dict], evidence: List[Dict]) -> Dict:
    lang = detect_user_lang(request["project_brief"])
    payload = {
        "request": request,
        "top_projects": top_projects[:4],
        "evidence": evidence[:14],
    }

    if lang == "zh":
        prompt = (
            "你是 iGEM 冲奖项目总监。请直接输出可执行方案，不要写泛化路线说明。"
            "基于输入证据，输出 JSON，字段严格为："
            "summary, strategy, weekly_plan, immediate_actions, risks。"
            "要求："
            "1) summary: 一句话结论(<=60字) + 核心抓手(2-4条)。"
            "2) strategy: 分为 technical_path, validation_path, wiki_story_path，每项必须可执行。"
            "3) weekly_plan: 必须 8 项（week=1..8），每项含 objective, tasks(>=3), deliverables(>=2), acceptance_criteria(>=2)。"
            "4) immediate_actions: 给出接下来7天可做任务(>=6条)，每条必须具体到动作和产出。"
            "5) risks: 给出3-5条关键风险和对应缓解动作。"
            "6) 禁止输出“路线1/路线2/优势/风险模板化段落”；禁止空泛词；禁止逐字断行。"
            "7) 仅使用输入中的队伍和证据，不得虚构。"
            "仅输出 JSON，不要 markdown。\n\n"
            f"INPUT:\n{json.dumps(payload, ensure_ascii=False)}"
        )
    else:
        prompt = (
            "You are an iGEM project director. Output a concrete execution plan only, not generic route templates. "
            "Return strict JSON keys: summary, strategy, weekly_plan, immediate_actions, risks. "
            "weekly_plan must have exactly 8 weeks with objective, tasks(>=3), deliverables(>=2), acceptance_criteria(>=2). "
            "No generic wording, no hallucinated teams/years, no broken-character line breaks. JSON only.\n\n"
            f"INPUT:\n{json.dumps(payload, ensure_ascii=False)}"
        )

    raw = rag.call_qwen_api(prompt)
    parsed = extract_json_block(raw)
    return parsed if isinstance(parsed, dict) else {}


def fallback_plan(request: Dict, top_projects: List[Dict], evidence: List[Dict]) -> Dict:
    brief = normalize_text(request.get("project_brief", ""))
    refs = [f"{p['team_name']}({p['year']})" for p in top_projects[:3]]
    ref_text = "、".join(refs) if refs else "历史相关项目"

    weekly = []
    for w in range(1, 9):
        weekly.append(
            {
                "week": w,
                "objective": f"第{w}周目标：围绕项目核心假设推进可验证里程碑",
                "tasks": [
                    "明确本周单一核心问题，并写出可测量指标",
                    "完成一次实验或建模迭代，并记录参数与结果",
                    "更新Wiki进展页与证据链，确保可追溯",
                ],
                "deliverables": [
                    f"Week{w} 数据记录表（含原始数据与结论）",
                    f"Week{w} 决策纪要（保留取舍依据）",
                ],
                "acceptance_criteria": [
                    "至少1个指标达到预设阈值或给出失败原因",
                    "本周产出可被队内成员复现",
                ],
            }
        )

    return {
        "summary": {
            "one_line": f"方向可行，建议用“实验验证 + 应用场景”双线并进；参考 {ref_text} 的可复用经验。",
            "key_levers": [
                "聚焦一个高价值场景，避免目标发散",
                "每周形成可复现证据，强化答辩可信度",
                "技术路线与Wiki叙事同步推进",
            ],
        },
        "strategy": {
            "technical_path": [
                "先做最小可行原型（MVP），再逐步补齐性能指标",
                "将关键变量参数化，形成可复用实验模板",
            ],
            "validation_path": [
                "设置阳性/阴性对照并固定评估口径",
                "每周至少一次结果复核，保留失败样本",
            ],
            "wiki_story_path": [
                "按“问题-方案-证据-影响”结构持续更新Wiki",
                "把每周关键图表沉淀为最终答辩素材",
            ],
        },
        "weekly_plan": weekly,
        "immediate_actions": [
            "D1: 明确目标用户和应用场景，产出一页问题定义文档",
            "D2: 列出3个可验证核心假设，产出指标表",
            "D3: 设计第一轮实验/建模流程，产出流程图",
            "D4: 跑通一次最小验证，产出结果截图和原始记录",
            "D5: 召开复盘会，产出风险清单与改进项",
            "D6: 更新Wiki草稿页，产出证据链目录",
            "D7: 确认下一周里程碑，产出排期表",
        ],
        "risks": [
            {"risk": "目标过大导致推进缓慢", "mitigation": "收敛到单一核心问题，采用周里程碑管理"},
            {"risk": "实验与叙事脱节", "mitigation": "每次实验后同步更新Wiki证据与结论"},
            {"risk": "数据不稳定", "mitigation": "增加对照组和重复实验，提前设定验收阈值"},
        ],
        "_debug": {"brief": brief, "evidence_count": len(evidence)},
    }


def sanitize_plan(plan: Dict) -> Dict:
    out = dict(plan or {})
    summary = out.get("summary", {})
    if isinstance(summary, str):
        summary = {"one_line": normalize_text(summary), "key_levers": []}
    summary["one_line"] = normalize_text(summary.get("one_line", summary.get("conclusion", "")))
    summary["key_levers"] = normalize_list(summary.get("key_levers", summary.get("highlights", [])))
    out["summary"] = summary

    strategy = out.get("strategy", {}) if isinstance(out.get("strategy", {}), dict) else {}
    strategy["technical_path"] = normalize_list(strategy.get("technical_path", []))
    strategy["validation_path"] = normalize_list(strategy.get("validation_path", []))
    strategy["wiki_story_path"] = normalize_list(strategy.get("wiki_story_path", []))
    out["strategy"] = strategy

    weekly = out.get("weekly_plan", [])
    if isinstance(weekly, dict):
        weekly = list(weekly.values())
    clean_weekly = []
    for i, w in enumerate(weekly, start=1):
        if not isinstance(w, dict):
            continue
        clean_weekly.append(
            {
                "week": w.get("week", i),
                "objective": normalize_text(w.get("objective", "")),
                "tasks": normalize_list(w.get("tasks", [])),
                "deliverables": normalize_list(w.get("deliverables", [])),
                "acceptance_criteria": normalize_list(w.get("acceptance_criteria", [])),
            }
        )
    out["weekly_plan"] = clean_weekly[:8]

    out["immediate_actions"] = normalize_list(out.get("immediate_actions", []))

    risks = out.get("risks", [])
    clean_risks = []
    if isinstance(risks, list):
        for r in risks:
            if isinstance(r, dict):
                clean_risks.append(
                    {
                        "risk": normalize_text(r.get("risk", "")),
                        "mitigation": normalize_text(r.get("mitigation", "")),
                    }
                )
            elif isinstance(r, str):
                clean_risks.append({"risk": normalize_text(r), "mitigation": ""})
    out["risks"] = clean_risks
    return out


def run_pipeline(rag, df: pd.DataFrame, request: Dict) -> Dict:
    intent_keywords = extract_intent(request["project_brief"], request["team_strengths"], request["focus_goal"])
    query_text = " ".join([request["project_brief"]] + intent_keywords + request["team_strengths"] + request["focus_goal"])

    top_projects = rank_projects(df, query_text, intent_keywords)
    evidence = select_evidence(rag, top_projects, query_text)

    plan = call_planner_llm(rag, request, top_projects, evidence)
    if not plan:
        plan = fallback_plan(request, top_projects, evidence)

    plan = sanitize_plan(plan)
    if len(plan.get("weekly_plan", [])) < 8:
        plan = fallback_plan(request, top_projects, evidence)
        plan = sanitize_plan(plan)

    result = {
        "request": request,
        "top_projects": top_projects,
        "evidence_map": evidence,
        "plan": plan,
        "confidence": "high" if len(evidence) >= 10 else ("medium" if len(evidence) >= 5 else "low"),
    }
    return result


def render_plan(result: Dict):
    plan = result.get("plan", {})
    summary = plan.get("summary", {})

    st.markdown("### 结论与执行方案")
    st.info(summary.get("one_line", "已生成可执行方案。"))
    if summary.get("key_levers"):
        st.markdown("**核心抓手**")
        for x in summary["key_levers"]:
            st.write(f"- {x}")

    st.markdown("### 三条执行主线")
    s = plan.get("strategy", {})
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='plan-card'><b>技术主线</b></div>", unsafe_allow_html=True)
        for x in s.get("technical_path", []):
            st.write(f"- {x}")
    with c2:
        st.markdown("<div class='plan-card'><b>验证主线</b></div>", unsafe_allow_html=True)
        for x in s.get("validation_path", []):
            st.write(f"- {x}")
    with c3:
        st.markdown("<div class='plan-card'><b>Wiki/答辩主线</b></div>", unsafe_allow_html=True)
        for x in s.get("wiki_story_path", []):
            st.write(f"- {x}")

    st.markdown("### 8周执行计划")
    for wk in plan.get("weekly_plan", []):
        wname = f"Week {wk.get('week', '')}"
        with st.expander(f"{wname} | {wk.get('objective', '')}", expanded=(str(wk.get("week", "")) in {"1", "week1", "Week 1"})):
            st.markdown("**任务**")
            for t in wk.get("tasks", []):
                st.write(f"- {t}")
            st.markdown("**交付物**")
            for d in wk.get("deliverables", []):
                st.write(f"- {d}")
            st.markdown("**验收标准**")
            for a in wk.get("acceptance_criteria", []):
                st.write(f"- {a}")

    st.markdown("### 接下来7天立即行动")
    for x in plan.get("immediate_actions", []):
        st.write(f"- {x}")

    if plan.get("risks"):
        with st.expander("风险与缓解（可选查看）", expanded=False):
            for r in plan["risks"]:
                st.write(f"- 风险：{r.get('risk', '')} | 缓解：{r.get('mitigation', '')}")

    with st.expander("证据来源（可选查看）", expanded=False):
        for ev in result.get("evidence_map", []):
            title = f"{ev.get('evidence_id', '')} | {ev.get('team_name', '')} ({ev.get('year', '')})"
            with st.expander(title, expanded=False):
                st.write(f"页面：{ev.get('page_title', 'Unknown')}")
                st.write(normalize_text(ev.get("quote", "")))
                if ev.get("source_url"):
                    st.write(f"来源：{ev.get('source_url')}")

    with st.expander("原始结构化结果（调试）", expanded=False):
        st.json(result)


def main():
    inject_styles()
    render_top_nav("推理探索")
    api_key = sidebar_global_api_key()

    try:
        rag = get_rag_system(api_key)
    except Exception as e:
        st.error(str(e))
        st.stop()

    df = load_teams_table()
    if "explore_result" not in st.session_state:
        st.session_state.explore_result = None

    st.markdown("### 推理探索（可执行方案版）")
    st.caption("输入你的项目方向后，系统会先检索相似历史项目，再用大模型归纳出可执行的8周计划。")

    with st.form("explore_form"):
        brief = st.text_area("项目简介（必填）", height=130, placeholder="例如：我们想做可现场检测的食品污染生物传感系统...")
        strengths = st.multiselect("队伍优势（可选）", TEAM_STRENGTH_OPTIONS, default=[])
        budget = st.selectbox("预算等级（可选）", ["不限", "low", "medium", "high"], index=0)
        goals = st.multiselect("强调方向（可选）", FOCUS_OPTIONS, default=[])
        ok = st.form_submit_button("生成可执行方案", use_container_width=True)

    if ok:
        if not brief.strip():
            st.warning("请先填写项目简介。")
        else:
            req = {
                "project_brief": normalize_text(brief.strip()),
                "team_strengths": strengths,
                "budget_level": budget,
                "focus_goal": goals,
            }
            with st.spinner("正在检索证据并生成方案..."):
                st.session_state.explore_result = run_pipeline(rag, df, req)

    if st.session_state.explore_result:
        render_plan(st.session_state.explore_result)


if __name__ == "__main__":
    main()
