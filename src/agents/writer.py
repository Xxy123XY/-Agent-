"""
简历优化 Agent (Writer) —— 根据结构化 JD 和相似度得分，
重写简历的自我评价和工作经历，突出量化成果和关键词匹配。
"""

from src.agents.base import AgentTimer, failure_result, model_metadata, success_result
from src.state import AgentState
from src.utils.output_parser import OutputParseError, parse_json_object


def _build_optimizer_prompt(
    resume: str,
    structured_jd,
    similarity: float,
    feedback: str = "",
    rag_context: str = "",
    user_requirements: str = "",
) -> str:
    """构建优化 Agent 的提示词。"""
    jd_data = structured_jd.model_dump() if hasattr(structured_jd, "model_dump") else structured_jd

    base = f"""你是一位顶尖的简历优化专家。请根据以下要求优化简历。

## 🎯 用户的核心要求（优先级最高，必须严格遵循）
{f'【{user_requirements}】' if user_requirements else '（无特殊要求，按通用原则优化）'}

## 目标职位信息
- 核心技能要求：{', '.join(jd_data.get('core_skills', []))}
- 硬性要求：{', '.join(jd_data.get('hard_requirements', []))}
- 软性素质：{', '.join(jd_data.get('soft_skills', []))}
- 业务关键词：{', '.join(jd_data.get('keywords', []))}
- 岗位概要：{jd_data.get('role_summary', '')}
- 当前简历与 JD 的向量相似度：{similarity:.2%}
"""

    if rag_context:
        base += f"\n## 参考简历模板与写作范例\n{rag_context}\n"

    base += """## 通用优化原则（在满足用户要求的前提下应用）
1. 重写"自我评价"段落：精准对标 JD 中的关键词和软性素质要求
2. 重构"工作经历"：每段经历突出量化成果（用数字说话）
3. 关键词匹配：确保 JD 中的 keywords 在简历中自然出现
4. 保持简历的真实性和可读性，不要编造不存在的经历
"""

    if feedback:
        base += f"\n## HR 审核反馈\n{feedback}\n"

    base += f"\n## 原始简历\n{resume}\n\n请输出优化后的完整简历："

    return base


def optimize_resume_node(state: AgentState, config: dict) -> dict:
    """简历优化节点（Writer）：生成优化后的简历。

    注意：纯生成逻辑，不含审核。审核由 reviewer.py 负责。
    多轮循环由各范式 Graph 的条件边控制。

    Args:
        state: 全局 AgentState。
        config: 配置字典。

    Returns:
        更新 state 的字典。
    """
    llm = config["llm"]
    resume = state.get("original_resume", "")
    structured_jd = state.get("structured_jd")
    similarity = state.get("resume_jd_similarity", 0.0)
    revision_round = state.get("revision_round", 0)
    previous_feedback = state.get("review_feedback", "")
    rag_context = state.get("rag_context", "")
    user_requirements = state.get("user_requirements", "")

    if not resume or not structured_jd:
        return {"error_message": "简历或结构化 JD 缺失，无法进行优化。"}

    prompt = _build_optimizer_prompt(
        resume, structured_jd, similarity, previous_feedback, rag_context, user_requirements
    )

    try:
        response = llm.invoke(prompt)
        optimized_resume = response.content

        trace = state.get("execution_trace") or []
        trace.append(f"[Writer] 第 {revision_round + 1} 轮优化完成，字数：{len(optimized_resume)}")
        return {
            "optimized_resume": optimized_resume,
            "execution_trace": trace,
            "error_message": None,
        }
    except Exception as e:
        trace = state.get("execution_trace") or []
        trace.append(f"[Writer] 生成失败：{str(e)}")
        return {
            "execution_trace": trace,
            "error_message": f"简历优化生成失败：{str(e)}",
        }


def chat_edit_node(state: AgentState, config: dict) -> dict:
    """聊天式编辑节点：一次 LLM 调用同时输出修改后的简历 + 回复文本。

    用于优化 Tab 的对话式修改流程。不依赖通用优化原则，完全按用户需求。
    """
    timer = AgentTimer()
    llm = config["llm"]
    resume = state.get("original_resume", "")
    user_requirements = state.get("user_requirements", "")
    rag_context = state.get("rag_context", "")
    chat_summary = state.get("_chat_summary", "")
    is_first = state.get("_is_first_edit", True)

    if not resume:
        return failure_result(
            agent="writer",
            stage="resume_chat_edit",
            error="缺少简历内容",
            metadata=model_metadata(config, latency_ms=timer.elapsed_ms()),
            legacy={"error_message": "缺少简历内容"},
        )

    # 首轮：结合 JD 做完整优化；后续：精准修改
    if is_first and user_requirements:
        instruction = (
            f"用户首次提出需求：{user_requirements}\n"
            "这是首次优化，请根据需求完整优化简历。"
        )
    elif user_requirements:
        instruction = f"用户最新需求：{user_requirements}\n只需修改这一部分，其他保持原样。"
    else:
        instruction = "请给出当前简历的简要评价（1-2句话），指出可以优化的方向。简历内容不变。"

    # 对话历史摘要
    summary_section = ""
    if chat_summary:
        summary_section = f"\n## 历史对话摘要\n{chat_summary}\n"

    # RAG 参考
    rag_section = ""
    if rag_context:
        rag_section = f"""
## 参考资料
{rag_context}

重要约束：参考资料只能用于学习岗位关键词、表达方式和项目描述角度。
不得把参考资料中的公司、项目、数据、成果直接写成用户经历；不得编造用户简历中不存在的事实。
"""

    prompt = f"""你是专业的简历修改助手。请根据用户需求修改简历。

## 用户的需求
{instruction}
{summary_section}
{rag_section}
## 当前简历
{resume}

## 输出格式
请输出一个 JSON，格式如下（直接输出 JSON，不要包裹在 markdown 中）：
{{"modified_resume": "修改后的完整简历全文", "reply": "简短说明改了什么，1-3句话"}}

如果用户没有提修改需求，modified_resume 和当前简历完全一致，reply 是简要评价。"""

    try:
        response = llm.invoke(prompt)
        result = parse_json_object(
            response.content,
            defaults={"modified_resume": resume, "reply": "已完成修改。"},
            required_keys=["modified_resume", "reply"],
        )

        trace = state.get("execution_trace") or []
        trace.append(f"[对话编辑] {result.get('reply', '')[:80]}...")

        optimized_resume = result.get("modified_resume", resume)
        reply = result.get("reply", "已完成修改。")
        data = {
            "optimized_resume": optimized_resume,
            "reply": reply,
        }
        return success_result(
            agent="writer",
            stage="resume_chat_edit",
            data=data,
            trace=trace,
            metadata=model_metadata(config, parser="json_object", latency_ms=timer.elapsed_ms()),
            legacy={
                "optimized_resume": optimized_resume,
                "reply": reply,
                "execution_trace": trace,
                "error_message": None,
            },
        )
    except OutputParseError as e:
        trace = state.get("execution_trace") or []
        trace.append(f"[对话编辑] JSON 解析失败：{e}")
        reply = "这次模型输出格式异常，我保留了当前简历，避免覆盖原文。"
        return failure_result(
            agent="writer",
            stage="resume_chat_edit",
            error=f"JSON解析失败：{e}",
            data={"optimized_resume": resume, "reply": reply},
            trace=trace,
            metadata=model_metadata(config, parser="json_object", latency_ms=timer.elapsed_ms()),
            legacy={
                "optimized_resume": resume,
                "reply": reply,
                "execution_trace": trace,
                "error_message": f"JSON解析失败：{e}",
            },
        )
    except Exception as e:
        trace = state.get("execution_trace") or []
        trace.append(f"[对话编辑] 修改失败：{e}")
        reply = "这次修改没有成功，我保留了当前简历。"
        return failure_result(
            agent="writer",
            stage="resume_chat_edit",
            error=f"简历修改失败：{e}",
            data={"optimized_resume": resume, "reply": reply},
            trace=trace,
            metadata=model_metadata(config, latency_ms=timer.elapsed_ms()),
            legacy={
                "optimized_resume": resume,
                "reply": reply,
                "execution_trace": trace,
                "error_message": f"简历修改失败：{e}",
            },
        )


def targeted_edit_node(state: AgentState, config: dict) -> dict:
    """精准修改节点：仅按用户指示做局部修改，不重写整份简历。

    与 optimize_resume_node 的区别：
    - 不应用通用优化原则
    - 不重写自我评价/工作经历
    - 只做用户要求的改动，其他部分原样保留
    """
    llm = config["llm"]
    resume = state.get("original_resume", "")
    user_requirements = state.get("user_requirements", "")
    ref_context = state.get("rag_context", "")

    if not resume or not user_requirements:
        return {"error_message": "缺少简历内容或修改要求"}

    prompt = f"""你是一位专业的简历修改助手。请对以下简历做精准修改。

## 🎯 用户的修改要求（唯一修改目标）
{user_requirements}

{ref_context if ref_context else ""}

## ⚠️ 重要规则
- 只修改用户要求的部分，其他内容原样保留
- 不要重写整份简历
- 不要添加用户没有要求的内容
- 保持原文的语气和风格

## 原始简历
{resume}

请输出修改后的完整简历（未修改部分与原文一致）："""

    try:
        response = llm.invoke(prompt)
        trace = state.get("execution_trace") or []
        trace.append(f"[精准修改] 按用户要求完成修改，字数：{len(response.content)}")
        return {
            "optimized_resume": response.content,
            "execution_trace": trace,
            "error_message": None,
        }
    except Exception as e:
        return {"error_message": f"精准修改失败：{str(e)}"}
