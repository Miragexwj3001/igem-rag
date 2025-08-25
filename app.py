import streamlit as st
from QwenRAGSystemOptimized import QwenRAGSystemOptimized

st.set_page_config(page_title="iGEM 合成生物学 RAG 助手", page_icon="🧬", layout="wide")


def build_system(api_key: str):
    """初始化 RAG 系统，可传入用户 API Key"""
    return QwenRAGSystemOptimized(
        api_key=api_key or None
    )


def main():
    st.title("🧬 iGEM 合成生物学 RAG 助手")

    # 侧边栏：API Key + 参数
    st.sidebar.header("设置")
    api_key = st.sidebar.text_input(
        "DASHSCOPE_API_KEY（用户可选填）",
        type="password",
        help="不填写则尝试读取服务器环境变量",
        placeholder="sk-xxxxxxxx",
    )

    if "rag_system" not in st.session_state or st.session_state.get("api_key_cached") != (api_key or "ENV"):
        # 初始化/重建系统实例（当用户切换Key时）
        try:
            st.session_state.rag_system = build_system(api_key or None)
            st.session_state.api_key_cached = api_key or "ENV"
        except Exception as e:
            st.error(str(e))
            st.stop()

    rag = st.session_state.rag_system

    # 聊天输入
    st.write("在下面输入你的问题，例如：**如何在 iGEM 项目中设计基因电路的调控策略？**")
    question = st.text_area("你的问题", height=120)

    col_run, col_params = st.columns([1, 1])
    with col_params:
        topk = st.slider("召回候选数（FAISS）", 10, 100, 50, 5)
        rerank_k = st.slider("重排候选数（越小越快）", 1, 10, 5, 1)
        use_bm25 = st.checkbox("启用BM25融合（更稳但更慢）", value=False)

    if st.button("发送"):
        if not question.strip():
            st.warning("请输入问题～")
            st.stop()

        # 如果类里有对应属性，就动态更新
        if hasattr(rag, "topk_faiss"):
            rag.topk_faiss = topk
        if hasattr(rag, "rerank_k"):
            rag.rerank_k = rerank_k
        if hasattr(rag, "use_bm25"):
            rag.use_bm25 = use_bm25

        with st.spinner("正在检索与生成..."):
            try:
                result = rag.query(question)
            except Exception as e:
                st.error(f"出错了：{e}")
                st.stop()

        st.markdown(result["answer"])
        with st.expander("参考来源（按重排顺序）"):
            for s in result["sources"]:
                st.write(f"- {s.get('team_name', '未知团队')}（{s.get('year', '未知年份')}）")


if __name__ == "__main__":
    main()
