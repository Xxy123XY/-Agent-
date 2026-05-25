"""
Plan-and-Execute 范式图 —— 先由 Planner 拆解任务为子步骤，
再由 Executor 逐步执行，最后聚合结果。

工作流架构：

  [入口: analyze_job]
       │
       ▼
  [compute_similarity]
       │
       ▼
  [planner] ── 分析任务，拆解为子步骤列表 ──┐
       │                                    │
       ▼                                    │
  [executor] ── 执行第 N 步 ────────────────┘
       │
       ▼
  [aggregator] ── 汇总所有步骤结果 → 最终简历 + 面试题
       │
       ▼
  [generate_questions]

核心特点：
- Planner 用 gpt-4o 分析 JD 和简历，生成 3-5 个有序子任务
- Executor 每次从队列取出一个待执行步骤，调用对应 Agent 函数
- Aggregator 收集所有步骤结果，拼接为最终输出
"""

import functools
from typing import Literal

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

from src.state import AgentState
from src.utils.output_parser import parse_json_array


PLANNER_PROMPT = """你是一位资深的技术项目规划师。请将以下求职简历优化任务拆解为 3-5 个有序的执行步骤。

## 任务背景
职位描述（JD）：{job_description}

候选人的原始简历：{resume}

当前简历与 JD 相似度：{similarity}

## 拆解要求
1. 每个步骤应该是独立、可执行的任务。
2. 步骤应该按逻辑顺序排列（先分析，后执行，最后审核）。
3. 为每个步骤指定一个明确的名称和详细描述。
4. 步骤应该具体，不要过于抽象。

请输出一个 JSON 数组，每个元素包含：
- step_id: 步骤编号（从 1 开始）
- name: 步骤名称（如 "提取JD关键词"、"重写自我评价"）
- description: 步骤详细描述

直接输出 JSON 数组，不要包裹在 markdown 代码块中。"""


def create_plan_exec_graph(
    config: dict,
    vector_store=None,
    memory_manager=None,
) -> StateGraph:
    """创建 Plan-and-Execute 范式 LangGraph 编译图。

    Args:
        config: 全局配置字典。
        vector_store: VectorStoreManager 实例（可选）。
        memory_manager: MemoryManager 实例（可选）。

    Returns:
        编译后的 StateGraph。
    """
    from src.agents.job_analyzer import analyze_job_node, compute_similarity_node

    analyze_fn = functools.partial(analyze_job_node, config=config)
    similarity_fn = functools.partial(compute_similarity_node, config=config)
    planner_fn = functools.partial(_planner_node, config=config)
    executor_fn = functools.partial(
        _executor_node, config=config, vector_store=vector_store
    )
    aggregator_fn = functools.partial(_aggregator_node, config=config)

    workflow = StateGraph(AgentState)

    workflow.add_node("analyze_job", analyze_fn)
    workflow.add_node("compute_similarity", similarity_fn)
    workflow.add_node("planner", planner_fn)
    workflow.add_node("executor", executor_fn)
    workflow.add_node("aggregator", aggregator_fn)

    workflow.set_entry_point("analyze_job")
    workflow.add_edge("analyze_job", "compute_similarity")
    workflow.add_edge("compute_similarity", "planner")
    workflow.add_edge("planner", "executor")

    # executor → 条件路由：继续执行下一步，或进入聚合
    workflow.add_conditional_edges(
        "executor",
        _route_after_executor,
        {
            "executor": "executor",           # 继续执行下一步
            "aggregator": "aggregator",        # 全部完成，聚合
            "error": END,
        },
    )

    workflow.add_edge("aggregator", END)

    return workflow.compile()


# ── Planner 节点 ──

def _planner_node(state: AgentState, config: dict) -> dict:
    """Planner 节点：分析任务并拆解为子步骤列表。

    Args:
        state: 全局 AgentState。
        config: 配置字典。

    Returns:
        更新 state 的字典（包含 plan_steps）。
    """
    llm = config["llm"]
    jd_text = state.get("job_description", "")
    resume = state.get("original_resume", "")
    similarity = state.get("resume_jd_similarity", 0.0)

    prompt = PLANNER_PROMPT.format(
        job_description=jd_text[:1500],
        resume=resume[:1500],
        similarity=f"{similarity:.2%}",
    )

    try:
        response = llm.invoke(prompt)
        steps = parse_json_array(response.content, min_items=1)

        # 确保所有步骤初始状态为 pending
        for step in steps:
            step["status"] = "pending"

        trace = state.get("execution_trace") or []
        trace.append(f"[Planner] 已拆解为 {len(steps)} 个子任务：{[s['name'] for s in steps]}")

        return {
            "plan_steps": steps,
            "execution_trace": trace,
            "error_message": None,
        }
    except Exception as e:
        # 降级：使用默认步骤
        default_steps = [
            {"step_id": 1, "name": "分析JD关键词", "description": "提取JD核心技能和关键词", "status": "pending"},
            {"step_id": 2, "name": "重写自我评价", "description": "对标JD优化自我评价段落", "status": "pending"},
            {"step_id": 3, "name": "优化工作经历", "description": "重构工作经历突出量化成果", "status": "pending"},
            {"step_id": 4, "name": "审核与修正", "description": "审核简历质量并修正问题", "status": "pending"},
        ]
        trace = state.get("execution_trace") or []
        trace.append(f"[Planner] 规划异常，使用默认步骤：{str(e)}")
        return {
            "plan_steps": default_steps,
            "execution_trace": trace,
            "error_message": None,
        }


# ── Executor 节点 ──

def _executor_node(state: AgentState, config: dict, vector_store=None) -> dict:
    """Executor 节点：执行当前待处理步骤。

    每次调用执行一个 pending 步骤，标记为 done。

    Args:
        state: 全局 AgentState。
        config: 配置字典。
        vector_store: VectorStoreManager 实例。

    Returns:
        更新 state 的字典。
    """
    plan_steps = state.get("plan_steps") or []
    trace = state.get("execution_trace") or []

    # 找到第一个 pending 步骤
    current_step = None
    for step in plan_steps:
        if step.get("status") == "pending":
            current_step = step
            break

    if current_step is None:
        return {"error_message": "所有步骤已执行完毕", "execution_trace": trace}

    step_name = current_step.get("name", "")
    step_desc = current_step.get("description", "")

    trace.append(f"[Executor] 开始执行步骤 {current_step.get('step_id')}: {step_name}")

    # 根据步骤名称调用对应的 Agent
    try:
        if "JD" in step_name or "关键词" in step_name or "分析" in step_name:
            from src.agents.job_analyzer import analyze_job_node
            result = analyze_job_node(state, config)

        elif "自我评价" in step_name or "工作经历" in step_name or "优化" in step_name:
            from src.agents.writer import optimize_resume_node
            result = optimize_resume_node(state, config)

        elif "审核" in step_name or "修正" in step_name:
            from src.agents.reviewer import review_resume_node
            result = review_resume_node(state, config)

        else:
            # 通用步骤：调用 Writer 处理
            from src.agents.writer import optimize_resume_node
            result = optimize_resume_node(state, config)

        current_step["status"] = "done"
        current_step["result"] = result.get("optimized_resume", "")[:200] or "OK"
        trace.append(f"[Executor] 步骤 {current_step.get('step_id')} 完成")

        # 合并结果
        merged = dict(result)
        merged["plan_steps"] = plan_steps
        merged["execution_trace"] = trace

        # 保留已有的 optimized_resume
        if not merged.get("optimized_resume"):
            merged["optimized_resume"] = state.get("optimized_resume", "")

        return merged

    except Exception as e:
        current_step["status"] = "failed"
        current_step["result"] = str(e)
        trace.append(f"[Executor] 步骤失败：{str(e)}")
        return {
            "plan_steps": plan_steps,
            "execution_trace": trace,
            "error_message": None,  # 不中断流程
        }


# ── Aggregator 节点 ──

def _aggregator_node(state: AgentState, config: dict) -> dict:
    """Aggregator 节点：汇总所有步骤结果，生成最终输出。

    Args:
        state: 全局 AgentState。
        config: 配置字典。

    Returns:
        更新 state 的字典。
    """
    trace = state.get("execution_trace") or []
    plan_steps = state.get("plan_steps") or []

    # 统计
    done_count = sum(1 for s in plan_steps if s.get("status") == "done")
    failed_count = sum(1 for s in plan_steps if s.get("status") == "failed")

    trace.append(f"[Aggregator] 共 {len(plan_steps)} 步，完成 {done_count}，失败 {failed_count}")

    # 如果还没有优化后简历，调用一次 Writer
    optimized = state.get("optimized_resume", "")
    if not optimized:
        from src.agents.writer import optimize_resume_node
        result = optimize_resume_node(state, config)
        optimized = result.get("optimized_resume", "")

    return {
        "optimized_resume": optimized or state.get("original_resume", ""),
        "review_passed": True,
        "current_stage": "plan_exec_done",
        "execution_trace": trace,
        "error_message": None,
    }


# ── 路由函数 ──

def _route_after_executor(state: AgentState) -> Literal["executor", "aggregator", "error"]:
    """Executor 后的条件路由。

    - 还有 pending 步骤 → 继续执行 executor
    - 全部完成 → 进入 aggregator
    - 出错 → 终止
    """
    if state.get("error_message"):
        return "error"

    plan_steps = state.get("plan_steps") or []
    has_pending = any(s.get("status") == "pending" for s in plan_steps)

    return "executor" if has_pending else "aggregator"
