"""Tab 2：简历优化师 —— 对话式修改 + 右侧实时预览。"""
import hashlib

import streamlit as st
from src.orchestrator import run_resume_optimization


def render_optimize_tab(config: dict, params: dict, vector_store=None):
    job_description = st.session_state.get("global_jd", "")
    original_resume = st.session_state.get("global_resume", "")
    orchestration_mode = params.get("orchestration_mode", params.get("execution_mode", "auto"))
    rag_enabled = params["rag_enabled"]

    st.header("📝 简历优化师")
    strategy_label = "自动编排：Plan-and-Execute + Reflection" if orchestration_mode == "auto" else params.get("execution_mode", "").upper()
    st.caption(f"左侧对话提出修改意见，右侧实时预览简历 | 当前策略：{strategy_label}")

    # ── 初始化 session ──
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "resume_versions" not in st.session_state:
        st.session_state["resume_versions"] = []
    if "current_resume" not in st.session_state:
        st.session_state["current_resume"] = original_resume
    # 如果还没有编辑历史，侧边栏简历变化时同步过来。
    if not st.session_state["resume_versions"] and original_resume.strip() and st.session_state["current_resume"] != original_resume:
        st.session_state["current_resume"] = original_resume
    if "chat_summary" not in st.session_state:
        st.session_state["chat_summary"] = ""
    if "optimize_plan_steps" not in st.session_state:
        st.session_state["optimize_plan_steps"] = []
    if "optimize_agent_results" not in st.session_state:
        st.session_state["optimize_agent_results"] = []
    if "optimize_rag_hits" not in st.session_state:
        st.session_state["optimize_rag_hits"] = []
    # ── 如果没有优化过且没在聊天，显示开始条件 ──
    structured_jd = st.session_state.get("structured_jd")

    col_left, col_right = st.columns([65, 35])

    # ═══════════════════ 右侧：简历面板 ═══════════════════
    with col_right:
        versions = st.session_state["resume_versions"]
        ver_label = f"v{len(versions)}" if versions else "原始"
        resume_digest = hashlib.sha1(st.session_state["current_resume"].encode("utf-8")).hexdigest()[:8]

        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(f"### 📄 {ver_label} 版本")
        with c2:
            if versions:
                if st.button("↩️ 回退", key="btn_rollback", use_container_width=True):
                    st.session_state["resume_versions"].pop()
                    prev = st.session_state["resume_versions"][-1] if st.session_state["resume_versions"] else original_resume
                    st.session_state["current_resume"] = prev
                    st.session_state["chat_history"].append({"role": "system", "content": "已回退到上一版本"})
                    st.rerun()

        st.text_area(
            "当前简历",
            value=st.session_state["current_resume"],
            height=500, label_visibility="collapsed", key=f"resume_panel_{len(versions)}_{resume_digest}",
        )

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if st.button("💾 保存", use_container_width=True, key="btn_save_final"):
                st.session_state["final_resume"] = st.session_state["current_resume"]
                st.toast("已保存！", icon="💾")
        with col_s2:
            if st.button("🔄 重置", use_container_width=True, key="btn_reset_all"):
                st.session_state["chat_history"] = []
                st.session_state["resume_versions"] = []
                st.session_state["current_resume"] = original_resume
                st.session_state["chat_summary"] = ""
                st.rerun()

    # ═══════════════════ 左侧：对话区 ═══════════════════
    with col_left:
        with st.container(height=500):
            history = st.session_state["chat_history"]
            if not history:
                st.info("👋 在下方输入你的修改需求，Agent 会同步修改简历。\n\n例如：把工作经历加上量化数据、精简自我评价到 50 字、突出 Python 技能...")
            for msg in history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            plan_steps = st.session_state.get("optimize_plan_steps") or []
            if plan_steps:
                with st.expander("📋 本轮 Plan-and-Execute / Reflection 轨迹"):
                    for step in plan_steps:
                        st.markdown(f"- **{step['name']}**：{step['status']}")

            agent_results = st.session_state.get("optimize_agent_results") or []
            if agent_results:
                with st.expander("Agent 结构化结果"):
                    for item in agent_results:
                        status = "OK" if item.get("ok") else "FAILED"
                        st.markdown(
                            f"- **{item.get('agent')} / {item.get('stage')}**："
                            f"{status} | `{item.get('metadata', {})}`"
                        )
                        if item.get("error"):
                            st.caption(f"error: {item['error']}")

            rag_hits = st.session_state.get("optimize_rag_hits") or []
            if rag_hits:
                with st.expander(f"RAG 命中文档（{len(rag_hits)}）"):
                    for hit in rag_hits:
                        score = hit.get("score")
                        score_text = f"{score:.6f}" if isinstance(score, float) else str(score or "-")
                        st.markdown(
                            f"**#{hit.get('rank')} {hit.get('source', 'knowledge_base')}** "
                            f"| 召回：`{hit.get('retrieval', 'hybrid')}` "
                            f"| score：`{score_text}`"
                        )
                        st.caption(hit.get("source_path", ""))
                        st.write(hit.get("content", ""))

        # ── 输入区 ──
        user_input = st.chat_input("提出修改需求...", key="chat_input")

        if user_input:
            if not structured_jd:
                st.warning("请先在「职位分析」标签页完成 JD 分析")
                return
            if not st.session_state["current_resume"].strip():
                st.warning("请先在左侧边栏上传或粘贴简历内容")
                return

            # 添加到聊天历史
            st.session_state["chat_history"].append({"role": "user", "content": user_input})

            # 是否首次编辑
            is_first = len(st.session_state["resume_versions"]) == 0

            with st.spinner("Agent 正在修改..."):
                result = run_resume_optimization(
                    config=config,
                    vector_store=vector_store,
                    current_resume=st.session_state["current_resume"],
                    user_input=user_input,
                    job_description=job_description,
                    chat_summary=st.session_state["chat_summary"],
                    is_first_edit=is_first,
                    rag_enabled=rag_enabled,
                )
                st.session_state["optimize_plan_steps"] = result.get("plan_steps", [])
                st.session_state["optimize_agent_results"] = result.get("agent_results", [])
                st.session_state["optimize_rag_hits"] = result.get("rag_hits", [])
                if result.get("rag_error"):
                    st.info(f"RAG 检索暂不可用，本次将直接根据当前简历修改：{result['rag_error']}")
                if result.get("web_error"):
                    st.info(f"Web Search 暂不可用，本次将不使用联网参考：{result['web_error']}")

            if result.get("error_message") and not result.get("optimized_resume"):
                st.error(result["error_message"])
            else:
                new_resume = result.get("optimized_resume")
                reply = result.get("reply", "已完成修改。")

                # 保存版本
                st.session_state["resume_versions"].append(st.session_state["current_resume"])
                st.session_state["current_resume"] = new_resume

                # 更新聊天历史
                st.session_state["chat_history"].append({"role": "assistant", "content": reply})

                # 更新对话摘要
                st.session_state["chat_summary"] += f"\n- 用户：{user_input} → Agent：{reply}"

                st.rerun()

    # ── 底部清除 ──
    st.divider()
    if st.button("🗑️ 清除对话历史", key="btn_clear_chat", use_container_width=False):
        st.session_state["chat_history"] = []
        st.session_state["resume_versions"] = []
        st.session_state["current_resume"] = original_resume
        st.session_state["chat_summary"] = ""
        st.session_state["optimize_plan_steps"] = []
        st.session_state["optimize_agent_results"] = []
        st.session_state["optimize_rag_hits"] = []
        st.rerun()
