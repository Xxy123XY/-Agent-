"""
ReAct 范式图 —— 实现「思考(Thought) → 行动(Action) → 观察(Observation)」循环。

工作流架构：

  [入口: analyze_job]
       │
       ▼
  [compute_similarity]
       │
       ▼
  [react_loop 入口]
       │
       ▼
  [think] ── 分析当前状态，决定下一步行动 ──┐
       │                                    │
       ▼                                    │
  [act] ── 调用 Tool / 执行操作 ────────────┘
       │
       ▼
  [observe] ── 记录结果，判断是否退出 ──┬── 退出 → [generate_questions]
       │                              │
       └── 继续循环 ──────────────────┘

核心特点：
- think 节点由 gpt-4o 驱动，分析当前状态并给出 "下一步行动" 或 "FINAL_ANSWER"
- act 节点根据 think 的决策，调用绑定的 Tool（RAG 检索、模板查询等）
- observe 节点检查 action 结果，决定继续循环还是退出
"""

import functools
from typing import Literal

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.state import AgentState
from src.utils.output_parser import OutputParseError, parse_json_object


REACT_SYSTEM_PROMPT = """你是一个遵循 ReAct 范式的智能求职助手。请按以下模式工作：

## 输出格式
当需要调用工具时：
THOUGHT: <你的分析思考>
ACTION: <工具名称>
ACTION_INPUT: <工具参数>

当你已经收集到足够信息，可以给出最终答案时：
THOUGHT: <你的总结思考>
FINAL_ANSWER: <最终输出>

## 可用工具
{tool_descriptions}

## 当前任务
{task_description}

## 注意事项
1. 每次只输出一个 THOUGHT + ACTION 组合，或一个 THOUGHT + FINAL_ANSWER 组合。
2. 在调用工具前，先思考这个工具能否帮助你完成任务。
3. 不要虚构工具返回结果，等待 observe 给出真实结果后再继续。
4. 简历优化的最终答案应该是完整的优化后简历文本。"""


def create_react_graph(
    config: dict,
    vector_store=None,
    memory_manager=None,
) -> StateGraph:
    """创建 ReAct 范式 LangGraph 编译图。

    Args:
        config: 全局配置字典。
        vector_store: VectorStoreManager 实例（可选）。
        memory_manager: MemoryManager 实例（可选）。

    Returns:
        编译后的 StateGraph。
    """
    from src.agents.job_analyzer import analyze_job_node, compute_similarity_node
    from src.mcp.client import get_mcp_tools
    from src.tools.job_tools import init_tools

    # 初始化 MCP 工具并绑定到 LLM
    if vector_store:
        init_tools(vector_store, config["tool_llm"])

    tool_llm = config["tool_llm"]
    mcp_tools = get_mcp_tools() if vector_store else []
    if mcp_tools:
        tool_llm = tool_llm.bind_tools(mcp_tools)

    # 构建工具描述供 Think 节点 Prompt 使用
    tool_descriptions = "\n".join([
        f"- {t.name}: {t.description}"
        for t in mcp_tools
    ]) if mcp_tools else "无可用工具（RAG 未启用）"

    analyze_fn = functools.partial(analyze_job_node, config=config)
    similarity_fn = functools.partial(compute_similarity_node, config=config)

    # ── 构建图 ──
    workflow = StateGraph(AgentState)

    # 绑定 functools.partial 后的 think / act / observe 节点
    think_fn = functools.partial(
        _react_think_node, config=config, tool_descriptions=tool_descriptions
    )
    act_fn = functools.partial(
        _react_act_node, config=config, vector_store=vector_store
    )
    observe_fn = functools.partial(
        _react_observe_node, config=config
    )

    workflow.add_node("analyze_job", analyze_fn)
    workflow.add_node("compute_similarity", similarity_fn)
    workflow.add_node("think", think_fn)
    workflow.add_node("act", act_fn)
    workflow.add_node("observe", observe_fn)

    workflow.set_entry_point("analyze_job")

    # JD分析 → 相似度 → ReAct循环
    workflow.add_edge("analyze_job", "compute_similarity")
    workflow.add_edge("compute_similarity", "think")

    # think → act (始终执行)
    workflow.add_edge("think", "act")

    # act → observe (始终执行)
    workflow.add_edge("act", "observe")

    # observe → 循环判断
    workflow.add_conditional_edges(
        "observe",
        _route_after_observe,
        {
            "think": "think",              # 继续循环
            "questions": END,  # 完成
            "error": END,
        },
    )
    return workflow.compile()


# ── ReAct 循环节点 ──

def _react_think_node(state: AgentState, config: dict, tool_descriptions: str = "") -> dict:
    """Think 节点：分析当前状态，输出 THOUGHT + ACTION 或 FINAL_ANSWER。

    Args:
        state: 全局 AgentState。
        config: 配置字典。
        tool_descriptions: 工具描述文本。

    Returns:
        更新 state 的字典。
    """
    llm = config["llm"]
    jd_text = state.get("job_description", "")
    resume = state.get("original_resume", "")
    similarity = state.get("resume_jd_similarity", 0)
    trace = state.get("execution_trace") or []
    react_round = state.get("react_round", 0)

    task = f"""请根据以下职位描述和简历，协助完成简历优化任务。

职位描述：{jd_text[:1000]}

原始简历：{resume[:1000]}

当前简历与JD相似度：{similarity:.2%}
当前优化轮数：{react_round}"""

    prompt = REACT_SYSTEM_PROMPT.format(
        tool_descriptions=tool_descriptions,
        task_description=task,
    )

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content="请开始分析并执行优化任务。"),
    ]

    # 如果有上一轮的观察结果，追加到消息中
    last_observation = state.get("react_observation", "")
    if last_observation and last_observation.startswith("[OBSERVE]"):
        messages.append(AIMessage(content=f"上一轮观察结果：{last_observation}"))
        messages.append(HumanMessage(content="请根据观察结果继续思考下一步行动。"))

    try:
        response = llm.invoke(messages)
        thought = response.content

        trace.append(f"[Think] {thought[:200]}...")
        return {
            "execution_trace": trace,
            "error_message": None,
        }
    except Exception as e:
        trace.append(f"[Think] 失败：{str(e)}")
        return {
            "execution_trace": trace,
            "error_message": f"Think 节点失败：{str(e)}",
        }


def _react_act_node(state: AgentState, config: dict, vector_store=None) -> dict:
    """Act 节点：解析 Think 的输出，执行具体的 Tool 调用或生成操作。

    如果 Think 输出 FINAL_ANSWER，则直接生成优化后的简历。
    如果 Think 输出 ACTION，则调用对应工具。

    Args:
        state: 全局 AgentState。
        config: 配置字典。
        vector_store: VectorStoreManager 实例。

    Returns:
        更新 state 的字典。
    """
    trace = state.get("execution_trace") or []
    last_trace = trace[-1] if trace else ""
    react_round = state.get("react_round", 0)

    # 如果上一轮 think 包含 FINAL_ANSWER，直接生成最终简历
    if "FINAL_ANSWER" in last_trace.upper():
        from src.agents.writer import optimize_resume_node
        result = optimize_resume_node(state, config)
        result["current_stage"] = "react_done"
        result["review_passed"] = True
        trace.append("[Act] Think 输出 FINAL_ANSWER，执行最终生成")
        result["execution_trace"] = trace
        return result

    # 否则，尝试解析 ACTION 并调用工具（通过 MCP）
    if "ACTION:" in last_trace.upper() and vector_store:
        import re
        action_match = re.search(r'ACTION:\s*(\w+)', last_trace, re.IGNORECASE)
        # 多行 ACTION_INPUT：从 ACTION_INPUT: 开始到行尾或遇到 ACTION/FINAL 为止
        input_match = re.search(r'ACTION_INPUT:\s*(.+?)(?=\n\s*(?:ACTION|FINAL|THOUGHT|$)|\Z)', last_trace, re.IGNORECASE | re.DOTALL)

        if action_match:
            tool_name = action_match.group(1).strip()
            raw_input = input_match.group(1).strip() if input_match else ""

            from src.mcp.client import get_mcp_tools
            tool_map = {t.name: t for t in get_mcp_tools()}

            if tool_name in tool_map:
                tool = tool_map[tool_name]

                # 智能解析输入：尝试 JSON → 回退为字符串
                tool_input = _parse_tool_input(raw_input, tool)
                try:
                    # MCP StructuredTool 只支持 async invoke
                    import asyncio
                    try:
                        result = tool.invoke(tool_input)
                    except NotImplementedError:
                        result = asyncio.run(tool.ainvoke(tool_input))
                    observation = f"[OBSERVE] MCP工具 {tool_name} 返回：{str(result)[:500]}"
                    trace.append(f"[Act] MCP调用 {tool_name} → {str(result)[:100]}...")
                    return {
                        "react_observation": observation,
                        "execution_trace": trace,
                        "react_round": react_round + 1,
                        "error_message": None,
                    }
                except Exception as e:
                    trace.append(f"[Act] MCP工具调用失败：{str(e)}")
                    return {
                        "react_observation": f"[OBSERVE] MCP工具调用失败：{str(e)}",
                        "execution_trace": trace,
                        "react_round": react_round + 1,
                        "error_message": None,
                    }

    # 无法解析 → 强制生成最终结果
    trace.append("[Act] 无法解析 ACTION，直接进入最终生成")
    from src.agents.writer import optimize_resume_node
    result = optimize_resume_node(state, config)
    result["review_passed"] = True
    result["current_stage"] = "react_done"
    result["execution_trace"] = trace
    return result


def _react_observe_node(state: AgentState, config: dict) -> dict:
    """Observe 节点：评估 Act 的输出，决定是否退出 ReAct 循环。

    Args:
        state: 全局 AgentState。
        config: 配置字典。

    Returns:
        更新 state 的字典。
    """
    react_round = state.get("react_round", 0)
    max_rounds = config.get("max_react_rounds", 3)
    review_passed = state.get("review_passed", False)
    current_stage = state.get("current_stage", "")

    trace = state.get("execution_trace") or []
    trace.append(f"[Observe] 第 {react_round} 轮完成，状态={current_stage}")

    return {
        "execution_trace": trace,
        "error_message": None,
    }


def _parse_tool_input(raw: str, tool) -> str | dict:
    """智能解析工具输入：JSON 字符串 → dict，否则尝试匹配工具参数名。

    解决 MCP 工具要求 dict 输入但 LLM 可能输出纯字符串的问题。
    """
    # 尝试 1：直接 JSON 解析
    try:
        return parse_json_object(raw)
    except OutputParseError:
        pass

    # 尝试 2：LLM 可能把 JSON 写在引号里，去引号再试
    stripped = raw.strip().strip('"').strip("'")
    try:
        return parse_json_object(stripped)
    except OutputParseError:
        pass

    # 尝试 3：如果工具有 args_schema，取第一个参数名包裹字符串
    if hasattr(tool, "args_schema") and tool.args_schema:
        try:
            schema = tool.args_schema
            if hasattr(schema, "model_json_schema"):
                schema = schema.model_json_schema()
            props = schema.get("properties", {})
            if props:
                first_param = list(props.keys())[0]
                return {first_param: raw}
        except Exception:
            pass

    # 兜底：原始字符串
    return raw


def _route_after_observe(state: AgentState) -> Literal["think", "questions", "error"]:
    """Observe 后的条件路由。

    - 优化完成 → 进入面试题生成
    - 错误 → 终止
    - 否则 → 继续 ReAct 循环
    """
    if state.get("error_message"):
        return "error"
    if state.get("review_passed") or state.get("current_stage") in ("react_done", "optimize_done"):
        return "questions"
    if state.get("react_round", 0) >= 3:  # ReAct 最多 3 轮，与 max_react_rounds 一致
        return "questions"
    return "think"
