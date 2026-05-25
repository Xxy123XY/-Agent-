"""Tab 3：模拟面试官。"""
import streamlit as st
from langchain_core.messages import HumanMessage
from components.display import show_interview_report


def render_interview_tab(config: dict, params: dict, vector_store=None, memory_manager=None):
    job_description = params["job_description"]
    original_resume = params["original_resume"]
    execution_mode = params["execution_mode"]
    orchestration_mode = params.get("orchestration_mode", execution_mode)
    rag_enabled = params["rag_enabled"]
    memory_enabled = params["memory_enabled"]

    st.header("💬 模拟面试官")
    strategy_label = "自动编排：ReAct 出题 + Reflection 评估" if orchestration_mode == "auto" else execution_mode.upper()
    st.markdown(f"当前策略：**{strategy_label}** | RAG：{'开' if rag_enabled else '关'} | 记忆：{'开' if memory_enabled else '关'}")

    for key, default in [("interview_started", False), ("interview_messages", []),
                         ("interview_state", None), ("interview_questions_list", None)]:
        if key not in st.session_state: st.session_state[key] = default

    structured_jd = st.session_state.get("structured_jd")
    final_resume = st.session_state.get("final_resume", "") or original_resume

    SOURCE_ICONS = {"网上真实面经": "🌐", "向量题库": "🧠", "本地知识库": "📚", "JD分析": "📋", "简历挖掘": "📄"}

    ci1, ci2, ci3 = st.columns([1, 1, 1])

    with ci1:
        if st.button("🎤 生成面试题", type="primary", use_container_width=True, key="btn_gen_q"):
            if not structured_jd:
                st.warning("请先完成 JD 分析")
            else:
                with st.spinner("生成面试题中..."):
                    from src.agents.interviewer import generate_interview_questions_node
                    state = {
                        "structured_jd": structured_jd,
                        "optimized_resume": final_resume,
                        "original_resume": original_resume,
                        "rag_context": _rag(config, vector_store, job_description, original_resume) if rag_enabled else "",
                        "execution_trace": [
                            "[ReAct][Think] 根据 JD、简历和岗位技能决定出题方向。",
                            "[ReAct][Act] 检索本地知识库和在线面经作为题目参考。" if rag_enabled else "[ReAct][Act] RAG 未启用，直接基于 JD 和简历出题。",
                        ],
                    }
                    r = generate_interview_questions_node(state, config)
                if r.get("error_message"):
                    st.error(r["error_message"])
                else:
                    st.session_state["interview_questions_list"] = r.get("interview_questions", [])
                    st.session_state["interview_question_bank_hits"] = r.get("interview_question_bank_hits", [])
                    st.session_state["interview_question_bank_added"] = r.get("interview_question_bank_added", 0)
                    st.session_state["interview_supervisor_decision"] = r.get("interview_supervisor_decision", {})
                    st.success(f"已生成 {len(st.session_state['interview_questions_list'])} 道题！")

    with ci2:
        if st.button("▶️ 开始面试", type="primary", use_container_width=True, key="btn_start_iv"):
            if not st.session_state.get("interview_questions_list"):
                st.warning("请先生成面试题")
            else:
                st.session_state["interview_started"] = True
                st.session_state["interview_messages"] = []
                from src.agents.interviewer import conduct_interview_node
                istate = {
                    "structured_jd": structured_jd, "optimized_resume": final_resume,
                    "original_resume": original_resume, "interview_history": [],
                    "memory_context": _mem(config, memory_manager, job_description) if memory_enabled else "",
                }
                r = conduct_interview_node(istate, config, user_answer="")
                if r.get("error_message"):
                    st.error(r["error_message"])
                else:
                    msgs = r.get("interview_history", [])
                    st.session_state["interview_messages"] = msgs
                    st.session_state["interview_state"] = {
                        "structured_jd": structured_jd, "optimized_resume": final_resume,
                        "interview_history": msgs,
                    }
                st.rerun()

    with ci3:
        if st.button("🛑 结束面试 & 评估", type="secondary", use_container_width=True, key="btn_end_iv"):
            if not st.session_state.get("interview_messages"):
                st.warning("尚未开始面试")
            else:
                with st.spinner("生成评估报告..."):
                    from src.agents.interviewer import evaluate_interview_node
                    istate = st.session_state.get("interview_state", {})
                    istate["structured_jd"] = structured_jd
                    r = evaluate_interview_node(istate, config)
                if r.get("error_message"):
                    st.error(r["error_message"])
                else:
                    st.session_state["interview_report"] = r.get("interview_report")
                    st.session_state["interview_started"] = False
                    st.session_state["interview_eval_trace"] = [
                        "[Reflection][Review] 汇总候选人回答，检查技术深度、表达逻辑和岗位匹配。",
                        "[Reflection][Critic] 输出结构化评分、亮点和改进建议。",
                    ]
                    if memory_enabled:
                        rep = r.get("interview_report")
                        if rep:
                            scores = {"technical_depth": getattr(rep, "technical_depth", 0),
                                      "expression_logic": getattr(rep, "expression_logic", 0),
                                      "job_match": getattr(rep, "job_match", 0)}
                            summary = (f"面试评估：技术{getattr(rep, 'technical_depth', 0)}/10，"
                                       f"逻辑{getattr(rep, 'expression_logic', 0)}/10，"
                                       f"匹配{getattr(rep, 'job_match', 0)}/10。"
                                       f"{getattr(rep, 'overall_suggestion', '')}")
                            _save_memory(config, memory_manager, summary, scores)
                    st.success("评估报告已生成！")
                st.rerun()

    # 面试题清单
    qs = st.session_state.get("interview_questions_list")
    if qs:
        decision = st.session_state.get("interview_supervisor_decision") or {}
        route = decision.get("route") or {}
        if route:
            with st.expander("🧭 Supervisor 路由决策"):
                st.markdown(
                    f"题库：**{'启用' if route.get('question_bank') else '跳过'}** | "
                    f"RAG：**{'启用' if route.get('rag_context') else '跳过'}** | "
                    f"web_search：**{'启用' if route.get('web_search') else '跳过'}**"
                )
                st.caption(
                    f"题库命中：{decision.get('bank_hit_count', 0)} / "
                    f"阈值：{decision.get('min_bank_hits', 0)}"
                )
                for reason in decision.get("reasons", []):
                    st.markdown(f"- {reason}")

        bank_hits = st.session_state.get("interview_question_bank_hits") or []
        if bank_hits:
            with st.expander(f"🧠 向量题库命中（{len(bank_hits)}）"):
                for hit in bank_hits:
                    st.markdown(
                        f"**{hit.get('rank')}. [{hit.get('category', '未分类')}] "
                        f"{hit.get('source', '题库')}** "
                        f"| 召回：`{hit.get('retrieval', 'hybrid')}` "
                        f"| 分数：`{hit.get('score', '-')}`"
                    )
                    st.caption(hit.get("question", ""))
                    if hit.get("expected_points"):
                        st.caption(f"考察点：{hit.get('expected_points')}")
        added = st.session_state.get("interview_question_bank_added") or 0
        if added:
            st.caption(f"本轮已从 web_search 抽取并写入向量题库：{added} 道候选题。")

        with st.expander("📋 查看面试题清单"):
            for i, q in enumerate(qs, 1):
                cat = q.get("category", "")
                question = q.get("question", "")
                pts = q.get("expected_points", [])
                src = q.get("source", "JD分析")
                icon = SOURCE_ICONS.get(src, "📌")
                st.markdown(f"**{i}. [{cat}] {icon} {src}**")
                st.markdown(question)
                if pts and isinstance(pts, list):
                    st.caption(f"考察要点: {'; '.join(str(p) for p in pts[:3])}")

    # 面试对话
    if st.session_state.get("interview_started"):
        st.divider()
        st.subheader("🎙️ 面试进行中...")
        with st.container(height=400):
            for msg in st.session_state.get("interview_messages", []):
                if hasattr(msg, "type"):
                    with st.chat_message("assistant" if msg.type == "ai" else "user"):
                        st.markdown(msg.content)

        ua = st.chat_input("在此输入您的回答...")
        if ua:
            st.session_state["interview_messages"].append(HumanMessage(content=ua))
            with st.spinner("面试官思考中..."):
                from src.agents.interviewer import conduct_interview_node
                cur = st.session_state.get("interview_state", {})
                cur["interview_history"] = st.session_state["interview_messages"]
                if memory_enabled:
                    cur["memory_context"] = _mem(config, memory_manager, job_description)
                r = conduct_interview_node(cur, config, user_answer=ua)
            if r.get("error_message"):
                st.error(r["error_message"])
            else:
                st.session_state["interview_messages"] = r.get("interview_history", [])
                st.session_state["interview_state"]["interview_history"] = st.session_state["interview_messages"]
            st.rerun()

    show_interview_report()
    eval_trace = st.session_state.get("interview_eval_trace")
    if eval_trace:
        with st.expander("🧠 Reflection 评估轨迹"):
            for item in eval_trace:
                st.markdown(f"- {item}")

    if st.button("🗑️ 清除面试记录", use_container_width=True, key="btn_clear_iv"):
        for k in ["interview_started", "interview_messages", "interview_state",
                   "interview_questions_list", "interview_report", "interview_eval_trace",
                   "interview_question_bank_hits", "interview_question_bank_added",
                   "interview_supervisor_decision"]:
            st.session_state[k] = False if k == "interview_started" else ([] if k == "interview_messages" else None)
        st.rerun()


def _rag(config, vector_store, jd, resume):
    if vector_store is None:
        from src.rag.vector_store import VectorStoreManager
        vector_store = VectorStoreManager(config)
        vector_store.initialize()
    return vector_store.retrieve(f"{jd[:300]} {resume[:200]}", k=config["rag_top_k"])


def _mem(config, memory_manager, jd):
    if memory_manager is None:
        from src.memory.memory_manager import MemoryManager
        memory_manager = MemoryManager(config)
    return memory_manager.load_memory(jd[:300])


def _save_memory(config, memory_manager, summary, scores):
    if memory_manager is None:
        from src.memory.memory_manager import MemoryManager
        memory_manager = MemoryManager(config)
    memory_manager.save_memory(summary, scores)
