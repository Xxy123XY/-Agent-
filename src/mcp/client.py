"""
MCP Client 适配器 —— 将 MCP Server 的工具包装为 LangChain Tool，
供 LangGraph Agent 直接使用。

使用 langchain-mcp-adapters 自动启动 MCP Server 子进程，
通过 stdio 协议通信，完全解耦。

使用方式：
    from src.mcp.client import get_mcp_tools
    tools = get_mcp_tools()
    llm.bind_tools(tools)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_initialized = False
_mcp_tools = []


def get_mcp_tools() -> list:
    """获取 MCP Server 提供的所有工具（LangChain Tool 格式）。

    首次调用时启动 MCP Server 子进程并建立 stdio 连接。

    Returns:
        list: LangChain BaseTool 列表。连接失败时返回空列表。
    """
    global _initialized, _mcp_tools

    if _initialized:
        return _mcp_tools

    # 尝试 MCP，失败则回退到本地工具
    try:
        import asyncio
        from langchain_mcp_adapters.client import MultiServerMCPClient

        server_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "src", "mcp", "server.py"
        )

        client = MultiServerMCPClient({
            "job_tools": {
                "command": sys.executable,
                "args": [server_path],
                "transport": "stdio",
            }
        })

        # get_tools() 是异步方法，需要用 asyncio.run 包裹
        _mcp_tools = asyncio.run(client.get_tools())
    except Exception:
        # MCP 不可用时回退到本地工具
        from src.tools.job_tools import ALL_TOOLS
        _mcp_tools = list(ALL_TOOLS)

    _initialized = True
    return _mcp_tools
