"""侧边栏组件 —— 文件上传、已存文件快捷加载、Agent 运行配置。"""
import streamlit as st
from components.upload import parse_uploaded_file
from src.storage import delete_file, save_file, list_files, read_file, is_cached


def _clear_analysis_state():
    for key in [
        "analysis_result",
        "structured_jd",
        "interview_questions_list",
        "interview_report",
    ]:
        st.session_state.pop(key, None)


def _clear_resume_work_state():
    for key in [
        "chat_history",
        "resume_versions",
        "current_resume",
        "final_resume",
        "chat_summary",
        "interview_started",
        "interview_messages",
        "interview_state",
        "interview_questions_list",
        "interview_report",
    ]:
        st.session_state.pop(key, None)


def _clear_loaded_jd():
    st.session_state["global_jd"] = ""
    st.session_state.pop("_jd_hash", None)
    _clear_analysis_state()


def _clear_loaded_resume():
    st.session_state["global_resume"] = ""
    _clear_resume_work_state()


def render_sidebar(config: dict) -> dict:
    """渲染侧边栏，返回用户输入和配置。"""
    with st.sidebar:
        st.header("📋 全局输入")

        # ── 已存储文件快捷加载 ──
        stored_jd = list_files("jd")
        stored_resume = list_files("resume")

        if stored_jd:
            jd_names = {f"{'✅' if is_cached(f['hash']) else '🆕'} {f['name']}": f["hash"] for f in stored_jd}
            st.selectbox(f"📂 已存 JD（{len(stored_jd)} 份）", list(jd_names.keys()), key="jd_pick")
            jd_load_col, jd_delete_col = st.columns([2, 1])
            with jd_load_col:
                if st.button("✅ 加载 JD", key="jd_confirm", use_container_width=True):
                    pick = st.session_state.get("jd_pick", "")
                    if pick in jd_names:
                        content = read_file(jd_names[pick], "jd")
                        if content:
                            if st.session_state.get("global_jd") != content:
                                _clear_analysis_state()
                            st.session_state["global_jd"] = content
                            st.session_state["_jd_hash"] = jd_names[pick]
                            st.rerun()
            with jd_delete_col:
                if st.button("🗑️ 删除", key="jd_delete", use_container_width=True):
                    pick = st.session_state.get("jd_pick", "")
                    if pick in jd_names:
                        deleted_hash = jd_names[pick]
                        deleted = delete_file(deleted_hash, "jd")
                        if st.session_state.get("_jd_hash") == deleted_hash:
                            _clear_loaded_jd()
                        st.toast("已删除所选 JD" if deleted else "未找到该 JD 文件", icon="🗑️")
                        st.rerun()

        if stored_resume:
            cv_names = {f["name"]: f["hash"] for f in stored_resume}
            st.selectbox(f"📂 已存简历（{len(stored_resume)} 份）", list(cv_names.keys()), key="cv_pick")
            cv_load_col, cv_delete_col = st.columns([2, 1])
            with cv_load_col:
                if st.button("✅ 加载简历", key="cv_confirm", use_container_width=True):
                    pick = st.session_state.get("cv_pick", "")
                    if pick in cv_names:
                        content = read_file(cv_names[pick], "resume")
                        if content:
                            if st.session_state.get("global_resume") != content:
                                _clear_resume_work_state()
                            st.session_state["global_resume"] = content
                            st.rerun()
            with cv_delete_col:
                if st.button("🗑️ 删除", key="cv_delete", use_container_width=True):
                    pick = st.session_state.get("cv_pick", "")
                    if pick in cv_names:
                        deleted_hash = cv_names[pick]
                        current_content = st.session_state.get("global_resume", "")
                        deleted_content = read_file(deleted_hash, "resume") or ""
                        deleted = delete_file(deleted_hash, "resume")
                        if current_content and current_content == deleted_content:
                            _clear_loaded_resume()
                        st.toast("已删除所选简历" if deleted else "未找到该简历文件", icon="🗑️")
                        st.rerun()

        # ── JD 上传 ──
        def on_jd_upload():
            file = st.session_state.get("jd_uploader")
            if file is not None:
                text = parse_uploaded_file(file)
                if not text.startswith("⚠️"):
                    if st.session_state.get("global_jd") != text:
                        _clear_analysis_state()
                    st.session_state["global_jd"] = text
                    meta = save_file(text, file.name, "jd")
                    st.session_state["_jd_hash"] = meta["hash"]
                    st.toast(f"JD 已解析，共 {len(text)} 字", icon="✅")
                else:
                    st.error(text)

        st.file_uploader(
            "拖拽 JD 文件，或点击浏览（PDF / DOCX / TXT / MD）",
            type=["pdf", "docx", "doc", "txt", "md"],
            key="jd_uploader", on_change=on_jd_upload,
        )
        job_description = st.text_area(
            "职位描述", height=160,
            placeholder="粘贴职位描述，或上传文件自动填充...",
            label_visibility="collapsed", key="global_jd",
        )

        # ── 简历上传 ──
        def on_resume_upload():
            file = st.session_state.get("resume_uploader")
            if file is not None:
                text = parse_uploaded_file(file)
                if not text.startswith("⚠️"):
                    if st.session_state.get("global_resume") != text:
                        _clear_resume_work_state()
                    st.session_state["global_resume"] = text
                    save_file(text, file.name, "resume")
                    st.toast(f"简历已解析，共 {len(text)} 字", icon="✅")
                else:
                    st.error(text)

        st.file_uploader(
            "拖拽简历文件，或点击浏览", type=["pdf", "docx", "doc", "txt", "md"],
            key="resume_uploader", on_change=on_resume_upload,
        )
        original_resume = st.text_area(
            "原始简历", height=200,
            placeholder="粘贴简历内容，或上传文件自动填充...",
            label_visibility="collapsed", key="global_resume",
        )

        st.divider()

        # ── Agent 运行配置 ──
        st.header("⚙️ Agent 运行配置")

        orchestration_mode = st.selectbox(
            "执行策略",
            options=["auto", "react", "plan_exec", "reflection"],
            format_func=lambda x: {
                "auto": "自动编排（推荐）",
                "react": "实验模式：ReAct",
                "plan_exec": "实验模式：Plan-and-Execute",
                "reflection": "实验模式：Reflection",
            }[x],
            key="orchestration_mode",
        )
        execution_mode = orchestration_mode
        rag_enabled = st.checkbox("启用 RAG 知识库", value=True, key="rag_enabled")
        memory_enabled = st.checkbox("启用长期记忆", value=False, key="memory_enabled")

        st.divider()

        with st.expander("📖 范式说明"):
            st.markdown({
                "auto": (
                    "**自动编排**：职位分析使用 ReAct 思路做检索增强，"
                    "简历优化使用 Plan-and-Execute 拆解任务，"
                    "生成后使用 Reflection 审核与修正。"
                ),
                "reflection": "**Reflection**：生成→审查→打分→修正。",
                "react": "**ReAct**：思考→行动→观察循环。",
                "plan_exec": "**Plan-and-Execute**：先规划后执行。",
            }.get(orchestration_mode, ""))

        st.caption(f"模型: gpt-4o-mini / text-embedding-3-small")

        return {
            "job_description": job_description,
            "original_resume": original_resume,
            "execution_mode": execution_mode,
            "orchestration_mode": orchestration_mode,
            "rag_enabled": rag_enabled,
            "memory_enabled": memory_enabled,
        }
