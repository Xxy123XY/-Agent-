"""LangGraph orchestration workflow for resume optimization."""

from typing import TypedDict

from src.agents.writer import chat_edit_node
from src.utils.output_parser import parse_json_object


class ResumeOptimizationState(TypedDict, total=False):
    config: dict
    vector_store: object
    current_resume: str
    user_input: str
    job_description: str
    chat_summary: str
    is_first_edit: bool
    rag_enabled: bool
    rag_context: str
    rag_hits: list[dict]
    rag_error: str | None
    industry_context: str
    industry_error: str | None
    template_context: str
    template_error: str | None
    web_context: str
    web_error: str | None
    score_context: str
    score_error: str | None
    reference_context: str
    plan_steps: list[dict]
    agent_results: list[dict]
    writer_result: dict
    review: dict
    optimized_resume: str | None
    reply: str
    error_message: str | None


def run_resume_optimization(
    *,
    config: dict,
    vector_store,
    current_resume: str,
    user_input: str,
    job_description: str,
    chat_summary: str,
    is_first_edit: bool,
    rag_enabled: bool,
) -> dict:
    """Run the LangGraph resume optimization workflow for one edit turn."""
    initial_state: ResumeOptimizationState = {
        "config": config,
        "vector_store": vector_store,
        "current_resume": current_resume,
        "user_input": user_input,
        "job_description": job_description,
        "chat_summary": chat_summary,
        "is_first_edit": is_first_edit,
        "rag_enabled": rag_enabled,
        "agent_results": [],
        "reply": "",
        "error_message": None,
    }

    try:
        graph = _create_resume_optimization_graph()
        final_state = graph.invoke(initial_state)
        return _public_result(final_state)
    except Exception as e:
        # Keep the app usable even if LangGraph is unavailable or graph execution fails.
        fallback = _run_resume_optimization_sequential(
            config=config,
            vector_store=vector_store,
            current_resume=current_resume,
            user_input=user_input,
            job_description=job_description,
            chat_summary=chat_summary,
            is_first_edit=is_first_edit,
            rag_enabled=rag_enabled,
        )
        fallback["workflow_error"] = str(e)
        return fallback


def _create_resume_optimization_graph():
    from langgraph.graph import END, StateGraph

    workflow = StateGraph(ResumeOptimizationState)

    workflow.add_node("plan_user_request", _plan_user_request_node)
    workflow.add_node("search_industry_terms", _search_industry_terms_node)
    workflow.add_node("get_resume_template", _get_resume_template_node)
    workflow.add_node("retrieve_rag", _retrieve_rag_node)
    workflow.add_node("search_web", _search_web_node)
    workflow.add_node("write_resume", _write_resume_node)
    workflow.add_node("evaluate_resume_score", _evaluate_resume_score_node)
    workflow.add_node("reflect_review", _reflect_review_node)
    workflow.add_node("finalize", _finalize_node)

    workflow.set_entry_point("plan_user_request")
    workflow.add_edge("plan_user_request", "search_industry_terms")
    workflow.add_edge("search_industry_terms", "get_resume_template")
    workflow.add_edge("get_resume_template", "retrieve_rag")
    workflow.add_edge("retrieve_rag", "search_web")
    workflow.add_edge("search_web", "write_resume")
    workflow.add_edge("write_resume", "evaluate_resume_score")
    workflow.add_edge("evaluate_resume_score", "reflect_review")
    workflow.add_edge("reflect_review", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()


def _plan_user_request_node(state: ResumeOptimizationState) -> dict:
    rag_enabled = state.get("rag_enabled", False)
    plan_steps = _build_plan_steps(
        state.get("user_input", ""),
        rag_enabled=rag_enabled,
        is_first=state.get("is_first_edit", True),
    )
    plan_steps[0]["status"] = "done"
    return {
        "plan_steps": plan_steps,
        "agent_results": state.get("agent_results", []),
    }


def _retrieve_rag_node(state: ResumeOptimizationState) -> dict:
    plan_steps = list(state.get("plan_steps", []))
    if not state.get("rag_enabled", False):
        _set_step_status(plan_steps, 3, "skipped")
        return {"rag_context": "", "rag_error": None, "plan_steps": plan_steps}

    try:
        rag_context, rag_hits = _run_rag_with_hits(
            state["config"],
            state.get("vector_store"),
            f"{state.get('user_input', '')} {state.get('job_description', '')[:200]}",
            k=3,
        )
        _set_step_status(plan_steps, 3, "done" if rag_context else "skipped")
        return {"rag_context": rag_context, "rag_hits": rag_hits, "rag_error": None, "plan_steps": plan_steps}
    except Exception as e:
        _set_step_status(plan_steps, 3, "failed")
        return {"rag_context": "", "rag_hits": [], "rag_error": str(e), "plan_steps": plan_steps}


def _search_industry_terms_node(state: ResumeOptimizationState) -> dict:
    plan_steps = list(state.get("plan_steps", []))
    if not state.get("rag_enabled", False):
        _set_step_status(plan_steps, 1, "skipped")
        return {"industry_context": "", "industry_error": None, "plan_steps": plan_steps}

    try:
        context = _run_tool(
            "search_industry_terms",
            {"query": _build_terms_query(state.get("user_input", ""), state.get("job_description", ""))},
        )
        _set_step_status(plan_steps, 1, "done" if context else "skipped")
        return {"industry_context": context, "industry_error": None, "plan_steps": plan_steps}
    except Exception as e:
        _set_step_status(plan_steps, 1, "failed")
        return {"industry_context": "", "industry_error": str(e), "plan_steps": plan_steps}


def _get_resume_template_node(state: ResumeOptimizationState) -> dict:
    plan_steps = list(state.get("plan_steps", []))
    if not state.get("rag_enabled", False):
        _set_step_status(plan_steps, 2, "skipped")
        return {"template_context": "", "template_error": None, "plan_steps": plan_steps}

    try:
        context = _run_tool(
            "get_resume_template",
            {"job_type": _infer_job_type(state.get("job_description", ""), state.get("user_input", ""))},
        )
        _set_step_status(plan_steps, 2, "done" if context else "skipped")
        return {"template_context": context, "template_error": None, "plan_steps": plan_steps}
    except Exception as e:
        _set_step_status(plan_steps, 2, "failed")
        return {"template_context": "", "template_error": str(e), "plan_steps": plan_steps}


def _search_web_node(state: ResumeOptimizationState) -> dict:
    plan_steps = list(state.get("plan_steps", []))
    if not state.get("rag_enabled", False):
        _set_step_status(plan_steps, 4, "skipped")
        return {"web_context": "", "web_error": None, "plan_steps": plan_steps}

    try:
        web_context = _run_web_search(
            state.get("user_input", ""),
            state.get("job_description", ""),
        )
        _set_step_status(plan_steps, 4, "done" if web_context else "skipped")
        return {"web_context": web_context, "web_error": None, "plan_steps": plan_steps}
    except Exception as e:
        _set_step_status(plan_steps, 4, "failed")
        return {"web_context": "", "web_error": str(e), "plan_steps": plan_steps}


def _write_resume_node(state: ResumeOptimizationState) -> dict:
    plan_steps = list(state.get("plan_steps", []))
    _set_step_status(plan_steps, 5, "running")

    reference_context = _merge_reference_context(
        state.get("industry_context", ""),
        state.get("template_context", ""),
        state.get("rag_context", ""),
        state.get("web_context", ""),
    )
    writer_state = {
        "original_resume": state.get("current_resume", ""),
        "user_requirements": state.get("user_input", ""),
        "rag_context": reference_context,
        "_chat_summary": state.get("chat_summary", ""),
        "_is_first_edit": state.get("is_first_edit", True),
        "execution_trace": [],
    }
    result = chat_edit_node(writer_state, state["config"])
    agent_results = list(state.get("agent_results", []))
    if result.get("agent_result"):
        agent_results.append(result["agent_result"])

    _set_step_status(plan_steps, 5, "done" if result.get("optimized_resume") else "failed")
    return {
        "reference_context": reference_context,
        "writer_result": result,
        "optimized_resume": result.get("optimized_resume"),
        "reply": result.get("reply", ""),
        "error_message": result.get("error_message") if not result.get("optimized_resume") else None,
        "agent_results": agent_results,
        "plan_steps": plan_steps,
    }


def _evaluate_resume_score_node(state: ResumeOptimizationState) -> dict:
    plan_steps = list(state.get("plan_steps", []))
    if state.get("error_message") and not state.get("optimized_resume"):
        _set_step_status(plan_steps, 6, "skipped")
        return {"score_context": "", "score_error": None, "plan_steps": plan_steps}
    if not state.get("rag_enabled", False):
        _set_step_status(plan_steps, 6, "skipped")
        return {"score_context": "", "score_error": None, "plan_steps": plan_steps}

    _set_step_status(plan_steps, 6, "running")
    try:
        score_context = _run_tool(
            "evaluate_resume_score",
            {
                "resume_text": state.get("optimized_resume", "") or state.get("current_resume", ""),
                "job_requirements": _build_requirements_text(state.get("user_input", ""), state.get("job_description", "")),
            },
        )
        _set_step_status(plan_steps, 6, "done" if score_context else "skipped")
        return {"score_context": score_context, "score_error": None, "plan_steps": plan_steps}
    except Exception as e:
        _set_step_status(plan_steps, 6, "failed")
        return {"score_context": "", "score_error": str(e), "plan_steps": plan_steps}


def _reflect_review_node(state: ResumeOptimizationState) -> dict:
    plan_steps = list(state.get("plan_steps", []))
    if state.get("error_message") and not state.get("optimized_resume"):
        _set_step_status(plan_steps, 7, "skipped")
        return {"plan_steps": plan_steps, "review": {}}

    _set_step_status(plan_steps, 7, "running")
    review = _reflect_resume_change(
        state["config"],
        state.get("user_input", ""),
        state.get("current_resume", ""),
        state.get("optimized_resume", "") or state.get("current_resume", ""),
        score_context=state.get("score_context", ""),
    )
    agent_results = list(state.get("agent_results", []))
    if review.get("agent_result"):
        agent_results.append(review["agent_result"])

    _set_step_status(plan_steps, 7, "done" if review.get("passed") else "needs_attention")
    return {
        "review": review,
        "agent_results": agent_results,
        "plan_steps": plan_steps,
    }


def _finalize_node(state: ResumeOptimizationState) -> dict:
    if state.get("error_message") and not state.get("optimized_resume"):
        return state

    reply = state.get("reply") or "已完成修改。"
    review = state.get("review") or {}
    if review.get("feedback"):
        reply = f"{reply}\n\n**Reflection 审核**：{review['feedback']}"

    return {
        "optimized_resume": state.get("optimized_resume") or state.get("current_resume", ""),
        "reply": reply,
        "error_message": None,
    }


def _public_result(state: ResumeOptimizationState) -> dict:
    return {
        "optimized_resume": state.get("optimized_resume"),
        "reply": state.get("reply", ""),
        "plan_steps": state.get("plan_steps", []),
        "agent_results": state.get("agent_results", []),
        "review": state.get("review", {}),
        "rag_hits": state.get("rag_hits", []),
        "industry_error": state.get("industry_error"),
        "template_error": state.get("template_error"),
        "rag_error": state.get("rag_error"),
        "web_error": state.get("web_error"),
        "score_error": state.get("score_error"),
        "error_message": state.get("error_message"),
    }


def _set_step_status(plan_steps: list[dict], index: int, status: str) -> None:
    if 0 <= index < len(plan_steps):
        plan_steps[index]["status"] = status


def _run_resume_optimization_sequential(
    *,
    config: dict,
    vector_store,
    current_resume: str,
    user_input: str,
    job_description: str,
    chat_summary: str,
    is_first_edit: bool,
    rag_enabled: bool,
) -> dict:
    """Fallback implementation matching the graph semantics."""
    industry_context = ""
    industry_error = None
    template_context = ""
    template_error = None
    rag_context = ""
    rag_hits = []
    rag_error = None
    if rag_enabled:
        try:
            industry_context = _run_tool(
                "search_industry_terms",
                {"query": _build_terms_query(user_input, job_description)},
            )
        except Exception as e:
            industry_error = str(e)

        try:
            template_context = _run_tool(
                "get_resume_template",
                {"job_type": _infer_job_type(job_description, user_input)},
            )
        except Exception as e:
            template_error = str(e)

        try:
            rag_context, rag_hits = _run_rag_with_hits(config, vector_store, f"{user_input} {job_description[:200]}", k=3)
        except Exception as e:
            rag_error = str(e)

    web_context = ""
    web_error = None
    if rag_enabled:
        try:
            web_context = _run_web_search(user_input, job_description)
        except Exception as e:
            web_error = str(e)

    reference_context = _merge_reference_context(industry_context, template_context, rag_context, web_context)

    plan_steps = _build_plan_steps(
        user_input,
        rag_enabled=rag_enabled,
        is_first=is_first_edit,
    )
    plan_steps[0]["status"] = "done"
    plan_steps[1]["status"] = "done" if industry_context else ("failed" if industry_error else "skipped")
    plan_steps[2]["status"] = "done" if template_context else ("failed" if template_error else "skipped")
    plan_steps[3]["status"] = "done" if rag_context else ("failed" if rag_error else "skipped")
    plan_steps[4]["status"] = "done" if web_context else ("failed" if web_error else "skipped")
    plan_steps[5]["status"] = "running"
    agent_results = []

    state = {
        "original_resume": current_resume,
        "user_requirements": user_input,
        "rag_context": reference_context,
        "_chat_summary": chat_summary,
        "_is_first_edit": is_first_edit,
        "execution_trace": [],
    }
    result = chat_edit_node(state, config)
    if result.get("agent_result"):
        agent_results.append(result["agent_result"])
    plan_steps[5]["status"] = "done" if result.get("optimized_resume") else "failed"

    if result.get("error_message") and not result.get("optimized_resume"):
        return {
            "error_message": result["error_message"],
            "optimized_resume": None,
            "reply": "",
            "plan_steps": plan_steps,
            "agent_results": agent_results,
            "rag_hits": rag_hits,
            "industry_error": industry_error,
            "template_error": template_error,
            "rag_error": rag_error,
            "web_error": web_error,
            "score_error": None,
        }

    new_resume = result.get("optimized_resume") or current_resume
    reply = result.get("reply", "已完成修改。")

    score_context = ""
    score_error = None
    if rag_enabled:
        plan_steps[6]["status"] = "running"
        try:
            score_context = _run_tool(
                "evaluate_resume_score",
                {
                    "resume_text": new_resume,
                    "job_requirements": _build_requirements_text(user_input, job_description),
                },
            )
            plan_steps[6]["status"] = "done" if score_context else "skipped"
        except Exception as e:
            score_error = str(e)
            plan_steps[6]["status"] = "failed"
    else:
        plan_steps[6]["status"] = "skipped"

    plan_steps[7]["status"] = "running"
    review = _reflect_resume_change(config, user_input, current_resume, new_resume, score_context=score_context)
    if review.get("agent_result"):
        agent_results.append(review["agent_result"])
    plan_steps[7]["status"] = "done" if review.get("passed") else "needs_attention"
    if review.get("feedback"):
        reply = f"{reply}\n\n**Reflection 审核**：{review['feedback']}"

    return {
        "optimized_resume": new_resume,
        "reply": reply,
        "plan_steps": plan_steps,
        "agent_results": agent_results,
        "review": review,
        "rag_hits": rag_hits,
        "industry_error": industry_error,
        "template_error": template_error,
        "rag_error": rag_error,
        "web_error": web_error,
        "score_error": score_error,
        "error_message": None,
    }


def _run_rag(config, vector_store, query: str, k: int = 3) -> str:
    text, _hits = _run_rag_with_hits(config, vector_store, query, k)
    return text


def _run_rag_with_hits(config, vector_store, query: str, k: int = 3) -> tuple[str, list[dict]]:
    if vector_store is None:
        from src.rag.vector_store import VectorStoreManager

        vector_store = VectorStoreManager(config)
        vector_store.initialize()
    if hasattr(vector_store, "retrieve_with_hits"):
        return vector_store.retrieve_with_hits(query, k=k)
    text = vector_store.retrieve(query, k=k)
    return text, []


def _run_web_search(user_input: str, job_description: str) -> str:
    query = _build_web_query(user_input, job_description)
    if not query:
        return ""

    return _run_tool("web_search", {"query": query})


def _run_tool(tool_name: str, tool_input) -> str:
    tool = _get_tool(tool_name)
    if tool is None:
        return ""

    try:
        result = tool.invoke(tool_input)
    except TypeError:
        result = tool.invoke(next(iter(tool_input.values())) if isinstance(tool_input, dict) else tool_input)
    return str(result or "")


def _get_tool(tool_name: str):
    try:
        from src.runtime import create_default_registry

        tool = create_default_registry().tool(tool_name).entrypoint
        if tool is not None:
            return tool
    except Exception:
        pass

    try:
        from src.mcp.client import get_mcp_tools

        for tool in get_mcp_tools():
            if getattr(tool, "name", "") == tool_name:
                return tool
    except Exception:
        pass

    try:
        from src.tools.job_tools import ALL_TOOLS

        for tool in ALL_TOOLS:
            if getattr(tool, "name", "") == tool_name:
                return tool
    except Exception:
        pass

    return None


def _build_web_query(user_input: str, job_description: str) -> str:
    text = f"{user_input} {job_description[:300]}".strip()
    if not text:
        return ""
    return f"{text[:180]} 简历 项目经验 写法 关键词"


def _build_terms_query(user_input: str, job_description: str) -> str:
    text = f"{user_input} {job_description[:300]}".strip()
    return text[:220] or "岗位关键词 行业术语"


def _infer_job_type(job_description: str, user_input: str) -> str:
    text = f"{job_description} {user_input}"
    candidates = [
        "大模型应用工程师",
        "AI工程师",
        "后端开发",
        "数据分析",
        "产品经理",
        "算法工程师",
        "测试工程师",
        "前端开发",
        "Java开发",
        "Python开发",
    ]
    for candidate in candidates:
        if candidate in text:
            return candidate
    return "目标岗位"


def _build_requirements_text(user_input: str, job_description: str) -> str:
    return f"用户修改要求：{user_input}\n岗位要求摘要：{job_description[:1000]}"


def _merge_reference_context(
    industry_context: str,
    template_context: str,
    rag_context: str,
    web_context: str,
) -> str:
    parts = []
    if industry_context:
        parts.append(f"## MCP 行业术语参考\n{industry_context[:1500]}")
    if template_context:
        parts.append(f"## MCP 简历模板参考\n{template_context[:1500]}")
    if rag_context:
        parts.append(f"## 本地 RAG 参考\n{rag_context}")
    if web_context:
        parts.append(
            "## Web Search 参考\n"
            "以下内容只可用于学习岗位表达、关键词和经验描述角度，"
            "不得编造用户简历中不存在的经历、公司、项目或数据。\n"
            f"{web_context[:2500]}"
        )
    return "\n\n".join(parts)


def _build_plan_steps(user_input: str, rag_enabled: bool, is_first: bool) -> list[dict]:
    scope = "完整优化" if is_first else "局部精准修改"
    tool_status = "pending" if rag_enabled else "skipped"
    return [
        {"name": "Plan: 分析用户修改目标", "status": f"pending（{scope}：{user_input[:40]}）"},
        {"name": "Tool: MCP search_industry_terms 检索行业术语", "status": tool_status},
        {"name": "Tool: MCP get_resume_template 获取简历模板", "status": tool_status},
        {"name": "Execute: 混合检索 RAG 参考资料", "status": tool_status},
        {"name": "Tool: MCP web_search 搜索相关简历表达", "status": tool_status},
        {"name": "Execute: 生成简历修改版本", "status": "pending"},
        {"name": "Tool: MCP evaluate_resume_score 评估修改质量", "status": "pending"},
        {"name": "Reflection: 审核修改质量", "status": "pending"},
    ]


def _reflect_resume_change(
    config: dict,
    user_input: str,
    old_resume: str,
    new_resume: str,
    score_context: str = "",
) -> dict:
    if not new_resume or new_resume.strip() == old_resume.strip():
        feedback = "新版本与原文变化不明显，请检查修改需求是否被充分执行。"
        return {
            "passed": False,
            "feedback": feedback,
            "agent_result": {
                "ok": False,
                "agent": "reflection",
                "stage": "resume_change_review",
                "data": {"passed": False, "feedback": feedback},
                "trace": [feedback],
                "error": feedback,
                "metadata": {"rule": "unchanged_resume"},
            },
        }

    prompt = f"""你是简历质量审核 Agent。请判断本次修改是否满足用户要求。

用户要求：
{user_input}

修改前简历摘要：
{old_resume[:1200]}

修改后简历摘要：
{new_resume[:1200]}

MCP 评分工具参考：
{score_context[:1200] if score_context else "无"}

请只输出 JSON：{{"passed": true/false, "feedback": "一句话审核结论"}}"""

    try:
        response = config["fast_llm"].invoke(prompt)
        data = parse_json_object(
            response.content,
            defaults={"passed": True, "feedback": "修改已完成审核。"},
            required_keys=["passed", "feedback"],
        )
        result = {
            "passed": bool(data.get("passed", True)),
            "feedback": str(data.get("feedback", "修改已完成审核。")),
        }
        result["agent_result"] = {
            "ok": True,
            "agent": "reflection",
            "stage": "resume_change_review",
            "data": result.copy(),
            "trace": [result["feedback"]],
            "error": None,
            "metadata": {"parser": "json_object", "model": config.get("fast_model_name", "unknown")},
        }
        return result
    except Exception as e:
        feedback = f"审核模型暂不可用，已保留生成结果（{e}）。"
        return {
            "passed": True,
            "feedback": feedback,
            "agent_result": {
                "ok": False,
                "agent": "reflection",
                "stage": "resume_change_review",
                "data": {"passed": True, "feedback": feedback},
                "trace": [feedback],
                "error": str(e),
                "metadata": {"fallback": "keep_generated_resume"},
            },
        }
