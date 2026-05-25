"""
范式路由器 —— 根据前端传入的 execution_mode 参数，
返回对应的 LangGraph 编译图实例。

使用方式：
    from src.graphs.router import route_to_graph
    graph = route_to_graph("react", config, tools)
"""

from langgraph.graph import StateGraph

from src.state import AgentState


def route_to_graph(
    execution_mode: str,
    config: dict,
    vector_store=None,
    memory_manager=None,
) -> StateGraph:
    """根据范式名称路由到对应的 LangGraph 编译图。

    Args:
        execution_mode: "react" | "plan_exec" | "reflection"。
        config: 全局配置字典。
        vector_store: VectorStoreManager 实例（可选，RAG 启用时传入）。
        memory_manager: MemoryManager 实例（可选，记忆启用时传入）。

    Returns:
        编译后的 LangGraph StateGraph 实例。

    Raises:
        ValueError: 范式名称不合法时抛出。
    """
    if execution_mode == "react":
        from src.graphs.react_graph import create_react_graph
        return create_react_graph(config, vector_store, memory_manager)

    elif execution_mode == "plan_exec":
        from src.graphs.plan_exec_graph import create_plan_exec_graph
        return create_plan_exec_graph(config, vector_store, memory_manager)

    elif execution_mode == "reflection":
        from src.graphs.reflection_graph import create_reflection_graph
        return create_reflection_graph(config, vector_store, memory_manager)

    else:
        raise ValueError(
            f"不支持的执行范式: {execution_mode}，"
            f"请选择 react / plan_exec / reflection"
        )
