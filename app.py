import streamlit as st

from ui_core import (
    TRACK_COL,
    get_rag_system,
    inject_styles,
    load_teams_table,
    render_result_block,
    render_top_nav,
    run_query_result,
    sidebar_global_api_key,
)

st.set_page_config(page_title="iGEM 首页洞察", page_icon="🧬", layout="wide")

INSIGHT_CARDS = [
    {
        "title": "2024 Diagnostics 队伍示例",
        "desc": "快速查看 2024 年 Diagnostics 赛道项目。",
        "prompt": "请列出 2024 年 Diagnostics 赛道的 8 个队伍和项目名称，并给出来源。",
        "color": "#0ea5e9",
    },
    {
        "title": "2024 Cancer 相关项目",
        "desc": "检索癌症相关项目示例。",
        "prompt": "请列出 2024 年与 cancer 或 tumor 相关的 iGEM 项目示例（队伍、项目名、一句话摘要）。",
        "color": "#10b981",
    },
    {
        "title": "2023-2024 Biosensor 项目",
        "desc": "按关键词查看近两年 biosensor 项目。",
        "prompt": "请列出 2023-2024 年与 biosensor 相关的 iGEM 项目示例（至少 10 个），并标注年份。",
        "color": "#f59e0b",
    },
    {
        "title": "2024 建模相关项目",
        "desc": "简单检索 modeling/model 相关项目。",
        "prompt": "请列出 2024 年提到 modeling 或 model 的 iGEM 项目示例（队伍、项目、来源）。",
        "color": "#8b5cf6",
    },
    {
        "title": "2024 应用方向概览",
        "desc": "总结 2024 常见应用方向。",
        "prompt": "请根据 2024 年项目，列出 5 个常见应用方向，并给出每个方向的代表队伍与项目。",
        "color": "#ef4444",
    },
]


def main():
    inject_styles()
    api_key = sidebar_global_api_key()

    try:
        rag = get_rag_system(api_key)
    except Exception as e:
        st.error(str(e))
        st.stop()

    df = load_teams_table()
    if "home_card_results" not in st.session_state:
        st.session_state.home_card_results = {}
    if "home_custom_result" not in st.session_state:
        st.session_state.home_custom_result = None

    render_top_nav("首页")
    st.markdown(
        "<div class='hero'><h1>iGEM 智能洞察中心</h1><p>首页可快速获取热点信息，顶部可切换到知识检索和推理探索页面。</p></div>",
        unsafe_allow_html=True,
    )

    years = df["year_num"].dropna().astype(int)
    tracks = df[TRACK_COL].replace("", None).dropna().nunique()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='metric-card'><div>项目记录数</div><div><b>{len(df):,}</b></div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(
            f"<div class='metric-card'><div>覆盖年份</div><div><b>{years.min()}-{years.max()}</b></div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(f"<div class='metric-card'><div>赛道数量</div><div><b>{tracks}</b></div></div>", unsafe_allow_html=True)

    row_layouts = [3, 2]
    cursor = 0
    for per_row in row_layouts:
        cols = st.columns(per_row)
        for col in cols:
            if cursor >= len(INSIGHT_CARDS):
                break
            card = INSIGHT_CARDS[cursor]
            with col:
                st.markdown(
                    f"<div class='insight-card' style='border-top:4px solid {card['color']};'><div><b>{card['title']}</b></div><div>{card['desc']}</div></div>",
                    unsafe_allow_html=True,
                )
                if st.button("立即查看", key=f"home_card_{cursor}", use_container_width=True):
                    st.session_state.home_card_results[cursor] = run_query_result(rag, card["prompt"])
                if cursor in st.session_state.home_card_results:
                    render_result_block(st.session_state.home_card_results[cursor], title="卡片结果")
            cursor += 1

    st.markdown("---")
    q = st.text_area("自定义问题", height=90)
    if st.button("运行自定义问题"):
        if q.strip():
            st.session_state.home_custom_result = run_query_result(rag, q.strip())
        else:
            st.warning("请输入问题后再运行。")
    if st.session_state.home_custom_result:
        render_result_block(st.session_state.home_custom_result, title="自定义问题结果")


if __name__ == "__main__":
    main()
