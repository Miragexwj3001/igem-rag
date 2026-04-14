import streamlit as st

from ui_core import (
    PROJ_COL,
    SUMMARY_COL,
    TEAM_COL,
    TRACK_COL,
    WIKI_COL,
    YEAR_COL,
    award_keyword_options,
    filter_projects,
    get_similarity_artifacts,
    get_rag_system,
    inject_styles,
    load_teams_table,
    normalize_text,
    parse_multi_with_unlimited,
    recommend_similar,
    render_chat_bubble,
    render_sources,
    render_top_nav,
    sidebar_global_api_key,
)

st.set_page_config(page_title="往届项目知识检索台", page_icon="📚", layout="wide")


def build_chat_query(user_question: str, filters: dict, context_project: dict | None) -> str:
    has_zh = any("\u4e00" <= ch <= "\u9fff" for ch in user_question)
    if context_project:
        team = str(context_project.get(TEAM_COL, ""))
        year = str(context_project.get(YEAR_COL, ""))
        proj = str(context_project.get(PROJ_COL, ""))
        summary = str(context_project.get(SUMMARY_COL, ""))[:220]
        if has_zh:
            return (
                f"请优先在 team={team}, year={year}, project={proj} 的上下文下回答。"
                f"项目摘要关键词: {summary}。用户问题: {user_question}。"
                "请用中文回答，并给出证据来源。"
            )
        return (
            f"Answer under context team={team}, year={year}, project={proj}, summary={summary}. "
            f"User question: {user_question}. Please answer in English with evidence citations."
        )

    if has_zh:
        return (
            f"请在以下过滤条件下回答用户问题。"
            f"关键词={filters.get('keyword') or '不限'}，"
            f"年份={filters.get('years') or '不限'}，"
            f"赛道={filters.get('tracks') or '不限'}，"
            f"奖项关键词={filters.get('awards') or '不限'}。"
            f"用户问题：{user_question}。请使用中文回答并给出来源。"
        )
    return (
        f"Answer under filters keyword={filters.get('keyword') or 'none'}, "
        f"years={filters.get('years') or 'none'}, tracks={filters.get('tracks') or 'none'}, "
        f"awards={filters.get('awards') or 'none'}. User question: {user_question}. "
        "Please answer in English with citations."
    )


def main():
    inject_styles()
    render_top_nav("知识检索")
    api_key = sidebar_global_api_key()

    try:
        rag = get_rag_system(api_key)
    except Exception as e:
        st.error(str(e))
        st.stop()

    df = load_teams_table()

    if "search_chat_messages" not in st.session_state:
        st.session_state.search_chat_messages = [{"role": "assistant", "content": "欢迎使用知识检索台。请先设置过滤条件，再在右侧对话提问。"}]
    if "search_filters" not in st.session_state:
        st.session_state.search_filters = {"keyword": "", "years": [], "tracks": [], "awards": []}
    if "search_selected_project" not in st.session_state:
        st.session_state.search_selected_project = None
    if "search_selected_row_id" not in st.session_state:
        st.session_state.search_selected_row_id = None
    if "search_locked_project" not in st.session_state:
        st.session_state.search_locked_project = None

    years_all = sorted([int(y) for y in df["year_num"].dropna().unique().tolist()])
    tracks_all = sorted([t for t in df[TRACK_COL].astype(str).unique().tolist() if t.strip()])
    award_opts = award_keyword_options(df)

    left, right = st.columns([1.1, 1.35], gap="large")

    with left:
        with st.form("search_filter_form"):
            keyword = st.text_input("关键词（可不填）", value=st.session_state.search_filters.get("keyword", ""))
            y_default = ["不限"] if not st.session_state.search_filters["years"] else st.session_state.search_filters["years"]
            t_default = ["不限"] if not st.session_state.search_filters["tracks"] else st.session_state.search_filters["tracks"]
            a_default = ["不限"] if not st.session_state.search_filters["awards"] else st.session_state.search_filters["awards"]

            year_selected = st.multiselect("年份", options=["不限"] + years_all, default=y_default)
            track_selected = st.multiselect("赛道", options=["不限"] + tracks_all, default=t_default)
            award_selected = st.multiselect("奖项关键词", options=["不限"] + award_opts, default=a_default)

            apply_filter = st.form_submit_button("确定检索", use_container_width=True)

        if apply_filter:
            st.session_state.search_filters = {
                "keyword": keyword.strip(),
                "years": parse_multi_with_unlimited(year_selected, years_all),
                "tracks": parse_multi_with_unlimited(track_selected, tracks_all),
                "awards": parse_multi_with_unlimited(award_selected, award_opts),
            }

        f = st.session_state.search_filters
        filtered = filter_projects(df, f["keyword"], f["years"], f["tracks"], f["awards"])
        st.caption(f"当前命中 {len(filtered)} 条项目记录")
        st.dataframe(filtered[[YEAR_COL, TEAM_COL, TRACK_COL, PROJ_COL]].head(160), use_container_width=True, height=320)

        if len(filtered) > 0:
            options = [
                (
                    f"{int(r['year_num']) if r['year_num'] == r['year_num'] else ''} | {r[TEAM_COL]} | {r[PROJ_COL]}",
                    int(r["row_id"]),
                )
                for _, r in filtered.head(400).iterrows()
            ]
            labels = [x[0] for x in options]
            row_ids = [x[1] for x in options]
            default_idx = row_ids.index(st.session_state.search_selected_row_id) if st.session_state.search_selected_row_id in row_ids else 0
            selected_label = st.selectbox("选择项目", options=labels, index=default_idx)
            selected_row_id = row_ids[labels.index(selected_label)]

            st.session_state.search_selected_row_id = selected_row_id
            selected_row = filtered[filtered["row_id"] == selected_row_id].iloc[0]
            st.session_state.search_selected_project = selected_row.to_dict()

            b1, b2 = st.columns(2)
            with b1:
                if st.button("提问前锁定项目", use_container_width=True):
                    st.session_state.search_locked_project = selected_row.to_dict()
            with b2:
                if st.button("解除锁定", use_container_width=True):
                    st.session_state.search_locked_project = None

            st.write(f"团队：{selected_row.get(TEAM_COL, '')}")
            st.write(f"年份：{selected_row.get(YEAR_COL, '')}")
            st.write(f"赛道：{selected_row.get(TRACK_COL, '')}")
            st.write(f"项目：{selected_row.get(PROJ_COL, '')}")
            if str(selected_row.get(WIKI_COL, "")).strip():
                st.markdown(f"Wiki：[链接]({selected_row[WIKI_COL]})")
            st.info(normalize_text(str(selected_row.get(SUMMARY_COL, "暂无项目概述"))))

            _, matrix = get_similarity_artifacts(df)
            sims = recommend_similar(df, matrix, int(selected_row["row_id"]), topn=5)
            st.dataframe(sims, use_container_width=True, height=220)

    with right:
        f = st.session_state.search_filters
        context_project = st.session_state.search_locked_project or st.session_state.search_selected_project
        st.markdown(
            f"<div class='context-box'><b>当前过滤范围</b><br>关键词：{f['keyword'] or '不限'} | 年份：{f['years'] or '不限'} | 赛道：{f['tracks'] or '不限'}</div>",
            unsafe_allow_html=True,
        )
        if context_project:
            prefix = "已锁定" if st.session_state.search_locked_project else "当前选中"
            txt = f"{prefix}：{context_project.get(YEAR_COL, '')} | {context_project.get(TEAM_COL, '')} | {context_project.get(PROJ_COL, '')}"
            st.markdown(f"<div class='context-box'><b>提问上下文</b><br>{txt}</div>", unsafe_allow_html=True)

        if st.button("清空对话", use_container_width=True):
            st.session_state.search_chat_messages = [{"role": "assistant", "content": "对话已清空。你可以继续提问。"}]

        for i, msg in enumerate(st.session_state.search_chat_messages):
            render_chat_bubble(msg["role"], msg["content"])
            if msg.get("sources") and msg["role"] == "assistant":
                render_sources(msg["sources"], title=f"第 {i} 轮证据来源")

        user_prompt = st.chat_input("继续提问：中文提问中文答，英文提问英文答")
        if user_prompt:
            # 先把用户问题写入会话并立即显示
            st.session_state.search_chat_messages.append({"role": "user", "content": user_prompt})
            render_chat_bubble("user", user_prompt)

            q = build_chat_query(user_prompt, st.session_state.search_filters, context_project)
            with st.spinner("正在检索并生成回答..."):
                result = rag.query(q)

            ans = normalize_text(result.get("answer", ""))
            render_chat_bubble("assistant", ans)
            render_sources(result.get("sources", []), title="本轮证据来源")
            st.session_state.search_chat_messages.append(
                {"role": "assistant", "content": ans, "sources": result.get("sources", [])}
            )


if __name__ == "__main__":
    main()
