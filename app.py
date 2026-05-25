"""Streamlit entrypoint for the resume/job agent project."""

import streamlit as st

from components.sidebar import render_sidebar
from config import load_config
from src.memory.memory_manager import MemoryManager
from src.rag.vector_store import VectorStoreManager
from tabs.analyze import render_analyze_tab
from tabs.interview import render_interview_tab
from tabs.optimize import render_optimize_tab


st.set_page_config(
    page_title="智能求职 Agent",
    page_icon="📌",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def _load_config() -> dict:
    return load_config()


@st.cache_resource(show_spinner=False)
def _get_vector_store(_config: dict) -> VectorStoreManager:
    return VectorStoreManager(_config)


@st.cache_resource(show_spinner=False)
def _get_memory_manager(_config: dict) -> MemoryManager:
    return MemoryManager(_config)


def main() -> None:
    st.title("智能求职 Agent")
    st.caption("职位分析、简历优化、模拟面试的一体化 Agent 项目")

    try:
        config = _load_config()
    except Exception as exc:
        st.error(f"配置加载失败：{exc}")
        st.info("请检查 `.env` 中的 OPENAI_API_KEY / OPENAI_BASE_URL 等配置。")
        st.stop()

    vector_store = _get_vector_store(config)
    memory_manager = _get_memory_manager(config)

    params = render_sidebar(config)

    tab_analyze, tab_optimize, tab_interview = st.tabs(
        ["职位分析器", "简历优化师", "模拟面试官"]
    )

    with tab_analyze:
        render_analyze_tab(config, params, vector_store, memory_manager)

    with tab_optimize:
        render_optimize_tab(config, params, vector_store)

    with tab_interview:
        render_interview_tab(config, params, vector_store, memory_manager)


if __name__ == "__main__":
    main()

