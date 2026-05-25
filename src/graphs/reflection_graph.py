"""
Reflection 范式图 —— 实现「生成 → 自我审查 → 打分 → 修正」的反思循环。

工作流架构：

  [入口: analyze_job]
       │
       ▼
  [compute_similarity]
       │
       ▼
  [generate] ── Writer 生成/优化简历 ──┐
       │                               │
       ▼                               │
  [review] ── Reviewer 审核简历 ───────┤
       │                               │
       ▼                               │
  [critic] ── Critic 打分 (0-10) ─────┘
       │
       ├── score >= 阈值 → [generate_questions] → [END]
       └── score < 阈值 → 回到 [generate]（附反思反馈）

核心特点：
- 复用现有的 Writer/Reviewer/Critic 三 Agent 协作
- Critic 打出 0-10 评分，阈值默认为 7
- 低于阈值时自动回退重写，最多 reflection_max_retries 次
- 每轮反思都会携带上一轮的扣分原因和改进建议
"""

import functools
from typing import Literal

from langgraph.graph import StateGraph, END

from src.state import AgentState


def create_reflection_graph(
    config: dict,
    vector_store=None,
    memory_manager=None,
) -> StateGraph:
    """创建 Reflection 范式 LangGraph 编译图。

    Args:
        config: 全局配置字典。
        vector_store: VectorStoreManager 实例（可选）。
        memory_manager: MemoryManager 实例（可选）。

    Returns:
        编译后的 StateGraph。
    """
    from src.agents.job_analyzer import analyze_job_node, compute_similarity_node
    from src.agents.writer import optimize_resume_node
    from src.agents.reviewer import review_and_score_node

    analyze_fn = functools.partial(analyze_job_node, config=config)
    similarity_fn = functools.partial(compute_similarity_node, config=config)
    generate_fn = functools.partial(optimize_resume_node, config=config)
    review_fn = functools.partial(review_and_score_node, config=config)

    workflow = StateGraph(AgentState)

    workflow.add_node("analyze_job", analyze_fn)
    workflow.add_node("compute_similarity", similarity_fn)
    workflow.add_node("generate", generate_fn)
    workflow.add_node("review", review_fn)

    workflow.set_entry_point("analyze_job")
    workflow.add_edge("analyze_job", "compute_similarity")
    workflow.add_edge("compute_similarity", "generate")
    workflow.add_edge("generate", "review")

    workflow.add_conditional_edges(
        "review",
        _route_after_review,
        {
            "generate": "generate",
            "pass": END,
            "human": END,
            "error": END,
        },
    )

    return workflow.compile()


def _route_after_review(state: AgentState) -> Literal["generate", "pass", "human", "error"]:
    """合并的审核+打分路由。"""
    if state.get("error_message"):
        return "error"

    if state.get("review_passed", False):
        return "pass"

    if state.get("revision_round", 0) >= 3:
        return "human"

    return "generate"
