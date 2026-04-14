import html
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from QwenRAGSystemOptimized import QwenRAGSystemOptimized

TEAM_COL = "团队名称"
YEAR_COL = "年份"
TRACK_COL = "赛道"
WIKI_COL = "Wiki链接"
PROJ_COL = "项目名称"
SUMMARY_COL = "项目概述"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
          .stApp {
            background:
              radial-gradient(900px 260px at 10% -5%, #def3ff 0%, transparent 60%),
              radial-gradient(700px 220px at 95% 0%, #e7f8f0 0%, transparent 55%),
              #f6f9fc;
          }
          [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f9fcff 0%, #eef5fb 100%);
            border-right: 1px solid #d8e2ec;
          }
          .hero {
            background: linear-gradient(135deg,#0b5f96 0%,#0f766e 100%);
            color:#fff;
            border-radius:18px;
            padding:24px;
            margin:6px 0 16px;
            box-shadow: 0 12px 24px rgba(11,95,150,.2);
          }
          .metric-card,.insight-card,.context-box,.result-box,.plan-card {
            background:#fff;
            border:1px solid #d7e1ea;
            border-radius:12px;
            padding:10px 12px;
          }
          .page-nav {
            display:flex;
            gap:8px;
            margin:6px 0 10px;
            flex-wrap: wrap;
          }
          .page-chip {
            display:inline-block;
            padding:6px 10px;
            border-radius:999px;
            border:1px solid #d7e1ea;
            background:#ffffff;
            color:#1d2b38;
            text-decoration:none;
            font-size:13px;
          }
          .page-chip.active {
            background:#0b5f96;
            color:#fff;
            border-color:#0b5f96;
          }
          .user-bubble {
            background:#0b5f96;
            color:#fff;
            border-radius:14px 14px 4px 14px;
            padding:10px 12px;
            max-width:90%;
            margin-left:auto;
          }
          .assistant-bubble {
            background:#fff;
            color:#1d2b38;
            border:1px solid #d7e1ea;
            border-radius:14px 14px 14px 4px;
            padding:10px 12px;
            max-width:92%;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_top_nav(active: str) -> None:
    pages = [
        ("首页", "/"),
        ("知识检索", "/知识检索台"),
        ("推理探索", "/推理探索"),
    ]
    chips = []
    for name, url in pages:
        cls = "page-chip active" if name == active else "page-chip"
        chips.append(f'<a class="{cls}" href="{url}" target="_self">{name}</a>')
    st.markdown(f"<div class='page-nav'>{''.join(chips)}</div>", unsafe_allow_html=True)


def detect_user_lang(text: str) -> str:
    return "zh" if re.search(r"[\u4e00-\u9fff]", text or "") else "en"


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    s = str(text).replace("\r\n", "\n")
    # Merge broken single-character lines like "传\n感\n器"
    pattern = r"((?:[A-Za-z0-9\u4e00-\u9fff]\s*\n\s*){2,}[A-Za-z0-9\u4e00-\u9fff])"
    s = re.sub(pattern, lambda m: re.sub(r"\s*\n\s*", "", m.group(1)), s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def normalize_list(items) -> List[str]:
    if items is None:
        return []
    if isinstance(items, str):
        items = [items]
    out = []
    for x in items:
        t = normalize_text(str(x))
        if t:
            out.append(t)
    return out


def parse_multi_with_unlimited(selected: List, all_options: List) -> List:
    if (not selected) or ("不限" in selected):
        return []
    return [x for x in selected if x in all_options]


@st.cache_resource(show_spinner=False)
def build_system_cached(api_key: str):
    return QwenRAGSystemOptimized(api_key=api_key or None)


def get_rag_system(api_key: str):
    return build_system_cached(api_key or "")


def sidebar_global_api_key() -> str:
    st.sidebar.markdown("## iGEM 助手")
    st.sidebar.caption("竞赛检索与洞察")

    if "global_api_key_saved" not in st.session_state:
        st.session_state.global_api_key_saved = ""
    if "global_api_key_input" not in st.session_state:
        st.session_state.global_api_key_input = ""

    val = st.sidebar.text_input(
        "DASHSCOPE_API_KEY（全局，只需填一次）",
        type="password",
        key="global_api_key_input",
        placeholder="sk-xxxxxxxx",
    )
    if val.strip():
        st.session_state.global_api_key_saved = val.strip()

    source = "会话缓存" if st.session_state.global_api_key_saved else "环境变量"
    st.sidebar.caption(f"当前使用: {source}")
    if st.session_state.global_api_key_saved and st.sidebar.button("清除已保存 Key", use_container_width=True):
        st.session_state.global_api_key_saved = ""
        st.session_state.global_api_key_input = ""
        st.rerun()

    return st.session_state.global_api_key_saved


@st.cache_data(show_spinner=False)
def load_teams_table() -> pd.DataFrame:
    try:
        df = pd.read_csv("out_embedding/igem_teams.csv", encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv("out_embedding/igem_teams.csv", encoding="utf-8")

    required_cols = [
        "Team ID",
        TEAM_COL,
        YEAR_COL,
        TRACK_COL,
        WIKI_COL,
        PROJ_COL,
        SUMMARY_COL,
        "medals",
        "awards_special_prizes",
        "nominations",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    df = df.fillna("").reset_index(drop=True)
    df["row_id"] = df.index
    df["year_num"] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df["awards_text"] = (
        df["medals"].astype(str)
        + " | "
        + df["awards_special_prizes"].astype(str)
        + " | "
        + df["nominations"].astype(str)
    )
    df["search_text"] = (
        df[TEAM_COL].astype(str)
        + " "
        + df[TRACK_COL].astype(str)
        + " "
        + df[PROJ_COL].astype(str)
        + " "
        + df[SUMMARY_COL].astype(str)
        + " "
        + df["awards_text"].astype(str)
    )
    return df


@st.cache_resource(show_spinner=False)
def build_similarity_index(search_texts: Tuple[str, ...]):
    vec = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=2)
    return vec, vec.fit_transform(search_texts)


def get_similarity_artifacts(df: pd.DataFrame):
    vec, mat = build_similarity_index(tuple(df["search_text"].tolist()))
    return vec, mat


def award_keyword_options(df: pd.DataFrame, topn: int = 80) -> List[str]:
    c = Counter()
    for txt in df["awards_text"].tolist():
        for tok in re.split(r"[;,/|、，\n]+", str(txt)):
            tok = tok.strip()
            if 1 < len(tok) < 48:
                c[tok] += 1
    return [k for k, _ in c.most_common(topn)]


def filter_projects(df: pd.DataFrame, keyword: str, years: List[int], tracks: List[str], awards: List[str]) -> pd.DataFrame:
    out = df.copy()
    if keyword.strip():
        patt = re.escape(keyword.strip())
        m = (
            out[TEAM_COL].str.contains(patt, case=False, regex=True)
            | out[PROJ_COL].str.contains(patt, case=False, regex=True)
            | out[SUMMARY_COL].str.contains(patt, case=False, regex=True)
            | out[TRACK_COL].str.contains(patt, case=False, regex=True)
        )
        out = out[m]
    if years:
        out = out[out["year_num"].isin(years)]
    if tracks:
        out = out[out[TRACK_COL].isin(tracks)]
    if awards:
        patt = "|".join(re.escape(k) for k in awards)
        out = out[out["awards_text"].str.contains(patt, case=False, regex=True)]
    return out.sort_values(by=["year_num", TEAM_COL], ascending=[False, True]).reset_index(drop=True)


def recommend_similar(df: pd.DataFrame, matrix, row_idx: int, topn: int = 5) -> pd.DataFrame:
    sims = cosine_similarity(matrix[row_idx], matrix).flatten()
    ranked = sims.argsort()[::-1]
    rows = []
    for idx in ranked:
        if idx == row_idx:
            continue
        rows.append(
            {
                "相似度": round(float(sims[idx]), 3),
                "年份": int(df.iloc[idx]["year_num"]) if pd.notna(df.iloc[idx]["year_num"]) else "",
                "团队": df.iloc[idx][TEAM_COL],
                "赛道": df.iloc[idx][TRACK_COL],
                "项目": df.iloc[idx][PROJ_COL],
            }
        )
        if len(rows) >= topn:
            break
    return pd.DataFrame(rows)


def run_query_result(rag, question: str) -> Dict:
    with st.spinner("正在检索并生成分析..."):
        return rag.query(question)


def render_sources(sources: List[Dict], title: str = "证据来源"):
    with st.expander(title, expanded=False):
        for s in sources:
            st.write(f"- {s.get('team_name', '未知队伍')}（{s.get('year', '未知年份')}）")


def render_result_block(result: Dict, title: Optional[str] = None):
    if title:
        st.markdown(f"**{title}**")
    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
    st.markdown(normalize_text(result.get("answer", "")))
    st.markdown("</div>", unsafe_allow_html=True)
    render_sources(result.get("sources", []))


def render_chat_bubble(role: str, text: str):
    safe = html.escape(normalize_text(str(text))).replace("\n", "<br>")
    c1, c2 = st.columns([1, 1])
    if role == "user":
        with c2:
            st.markdown(f"<div class='user-bubble'>{safe}</div>", unsafe_allow_html=True)
    else:
        with c1:
            st.markdown(f"<div class='assistant-bubble'>{safe}</div>", unsafe_allow_html=True)
