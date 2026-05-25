"""共用展示组件 —— 执行轨迹 + 面试评估报告。"""
import streamlit as st


def show_execution_trace(state: dict):
    """展示 Agent 执行轨迹。"""
    trace = state.get("execution_trace") or []
    if not trace:
        return
    with st.expander(f"🔍 查看执行轨迹（{len(trace)} 步）"):
        for i, t in enumerate(trace, 1):
            st.text(f"{i}. {t}")

    plan_steps = state.get("plan_steps") or []
    if plan_steps:
        with st.expander(f"📋 任务拆解清单（{len(plan_steps)} 步）"):
            for step in plan_steps:
                icon = {"pending": "⏳", "running": "🔄", "done": "✅", "failed": "❌"}
                st.markdown(
                    f"{icon.get(step.get('status', 'pending'), '❓')} "
                    f"**{step.get('step_id')}. {step.get('name')}** — {step.get('description')}"
                )
                if step.get("result"):
                    st.caption(f"结果: {step['result'][:200]}")


def show_interview_report():
    """展示面试评估报告。"""
    report = st.session_state.get("interview_report")
    if not report:
        return
    st.divider()
    st.subheader("📊 面试评估报告")

    d = report.model_dump() if hasattr(report, "model_dump") else report

    c1, c2, c3 = st.columns(3)
    c1.metric("技术深度", f"{d.get('technical_depth', 0)}/10")
    c2.metric("表达逻辑", f"{d.get('expression_logic', 0)}/10")
    c3.metric("岗位匹配度", f"{d.get('job_match', 0)}/10")

    l, r = st.columns(2)
    with l:
        st.markdown("#### ✅ 亮点")
        for s in d.get("strengths", []):
            st.markdown(f"- {s}")
    with r:
        st.markdown("#### ⚠️ 待改进")
        for i in d.get("improvements", []):
            st.markdown(f"- {i}")

    st.markdown("#### 💡 综合评价")
    st.info(d.get("overall_suggestion", ""))
    with st.expander("查看完整评估 JSON"):
        st.json(d)
