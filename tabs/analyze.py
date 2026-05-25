"""Tab 1：职位分析器。"""
import streamlit as st
from components.display import show_execution_trace
from src.state import StructuredJD, AgentState
from src.storage import is_cached, get_cached_analysis, save_cached_analysis


def render_analyze_tab(config: dict, params: dict, vector_store=None, memory_manager=None):
    """渲染职位分析标签页。"""
    job_description = params["job_description"]
    original_resume = params["original_resume"]
    execution_mode = params["execution_mode"]
    orchestration_mode = params.get("orchestration_mode", execution_mode)
    rag_enabled = params["rag_enabled"]
    memory_enabled = params["memory_enabled"]

    st.header("📊 职位分析器")
    strategy_label = "自动编排：ReAct 检索增强" if orchestration_mode == "auto" else execution_mode.upper()
    st.markdown(f"当前策略：**{strategy_label}** | RAG：{'开' if rag_enabled else '关'} | 记忆：{'开' if memory_enabled else '关'}")

    c1, c2 = st.columns([1, 1])

    with c1:
        if st.button("🔍 开始分析", type="primary", use_container_width=True, key="btn_analyze"):
            if not job_description.strip():
                st.warning("请先在左侧边栏输入职位描述（JD）")
            else:
                state = _build_initial(job_description, original_resume, execution_mode, rag_enabled, memory_enabled)

                jd_hash = st.session_state.get("_jd_hash", "")
                if jd_hash and is_cached(jd_hash):
                    cached = get_cached_analysis(jd_hash)
                    if cached:
                        st.success("⚡ 命中缓存，无需重新分析！")
                        st.session_state["analysis_result"] = {
                            "structured_jd": StructuredJD(**cached) if isinstance(cached, dict) else cached,
                            "execution_trace": [f"[缓存] 复用 {jd_hash[:8]}... 的分析结果"],
                        }
                        st.session_state["structured_jd"] = st.session_state["analysis_result"]["structured_jd"]
                        st.rerun()

                if rag_enabled:
                    with st.spinner("RAG 检索中..."):
                        state["execution_trace"].append("[ReAct][Think] 分析 JD 和简历关键词，判断是否需要检索行业知识。")
                        state["rag_context"] = _run_rag(config, state, vector_store)
                        if state["rag_context"]:
                            state["execution_trace"].append("[ReAct][Act] 调用 RAG 知识库检索岗位相关术语和模板。")
                            state["execution_trace"].append("[ReAct][Observe] 已获得可注入 JD 分析 Prompt 的参考片段。")
                        else:
                            state["execution_trace"].append("[ReAct][Observe] RAG 未返回可用片段，使用原始 JD 继续分析。")

                if memory_enabled:
                    state["memory_context"] = _run_memory(config, state, memory_manager)

                with st.spinner("正在分析职位描述..."):
                    try:
                        from src.agents.job_analyzer import analyze_job_node, compute_similarity_node
                        result = analyze_job_node(state, config)
                        if not result.get("error_message"):
                            state.update(result)
                            state["execution_trace"] = result.get("execution_trace", [])
                            result.update(compute_similarity_node(state, config))
                    except Exception as e:
                        st.error(f"分析异常：{e}")
                        st.stop()

                if result.get("error_message") and not result.get("structured_jd"):
                    st.error(result["error_message"])
                else:
                    st.success("职位分析完成！")
                    st.session_state["analysis_result"] = result
                    st.session_state["structured_jd"] = result.get("structured_jd")
                    if jd_hash and result.get("structured_jd"):
                        sj = result["structured_jd"]
                        save_cached_analysis(jd_hash, None, sj.model_dump() if hasattr(sj, "model_dump") else sj)

    with c2:
        if st.button("🗑️ 清除结果", use_container_width=True, key="btn_clear_analysis"):
            for k in ["analysis_result", "structured_jd"]:
                st.session_state.pop(k, None)
            st.rerun()

    # 展示结果
    r = st.session_state.get("analysis_result")
    sj = st.session_state.get("structured_jd")
    if r and sj:
        st.divider()
        st.subheader("📋 分析结果")
        jd = sj.model_dump() if hasattr(sj, "model_dump") else sj
        ca, cb = st.columns(2)
        with ca:
            st.markdown("#### 🔧 核心技能")
            for s in jd.get("core_skills", []): st.markdown(f"- `{s}`")
            st.markdown("#### 🎓 硬性要求")
            for s in jd.get("hard_requirements", []): st.markdown(f"- {s}")
        with cb:
            st.markdown("#### 💬 软性素质")
            for s in jd.get("soft_skills", []): st.markdown(f"- {s}")
            st.markdown("#### 🏷️ 业务关键词")
            st.markdown(" ".join(f"`{k}`" for k in jd.get("keywords", [])))
        st.markdown("#### 📌 岗位概要")
        st.info(jd.get("role_summary", ""))
        with st.expander("查看完整 JSON"):
            st.json(jd)
        show_execution_trace(r)


def _build_initial(jd, resume, mode, rag, mem):
    return {
        "job_description": jd, "original_resume": resume,
        "execution_mode": mode, "rag_enabled": rag, "memory_enabled": mem,
        "structured_jd": None, "resume_jd_similarity": None,
        "optimized_resume": None, "interview_questions": None,
        "interview_history": [], "interview_report": None,
        "current_stage": "analyze", "error_message": None,
        "review_passed": False, "review_feedback": None,
        "reflection_score": None, "revision_round": 0,
        "react_observation": None, "react_round": 0,
        "plan_steps": None,
        "rag_context": None, "memory_context": None, "execution_trace": [],
        "user_requirements": "",
    }


def _run_rag(config, state, vector_store=None):
    if vector_store is None:
        from src.rag.vector_store import VectorStoreManager
        vector_store = VectorStoreManager(config)
        vector_store.initialize()
    return vector_store.retrieve(state["job_description"][:300] + " " + state.get("original_resume", "")[:200], k=config["rag_top_k"])


def _run_memory(config, state, memory_manager=None):
    if memory_manager is None:
        from src.memory.memory_manager import MemoryManager
        memory_manager = MemoryManager(config)
    return memory_manager.load_memory(state["job_description"][:300])
