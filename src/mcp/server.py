"""
MCP Server —— 将 5 个求职工具封装为标准 MCP 服务。

独立进程运行，通过 stdio 与 Agent 通信。
任何支持 MCP 协议的客户端（Claude Desktop / LangGraph / 自定义 Agent）
都能调用这些工具，完全与 LLM Provider 解耦。

启动方式：
    python src/mcp/server.py
"""

import sys
import os

# 确保项目根在路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mcp.server.fastmcp import FastMCP
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# ── 初始化 MCP Server ──
mcp = FastMCP("JobAgentTools")

# ── 初始化依赖（从环境变量读取） ──
_vector_store = None
_llm = None


def _get_vector_store():
    global _vector_store
    if _vector_store is None:
        from config import load_config
        cfg = load_config()
        from src.rag.vector_store import VectorStoreManager
        _vector_store = VectorStoreManager(cfg)
        _vector_store.initialize()
    return _vector_store


def _get_llm():
    global _llm
    if _llm is None:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        _llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0.1,
            api_key=api_key, base_url=base_url,
        )
    return _llm


# ═══════════════════════ MCP 工具定义 ═══════════════════════

@mcp.tool()
def search_industry_terms(query: str) -> str:
    """搜索行业术语和黑话的解释。当遇到不理解的 JD 术语时调用。

    Args:
        query: 要查询的术语（如 供应链金融、DDD、CAP理论）
    """
    vstore = _get_vector_store()
    result = vstore.retrieve(query, k=3)
    return result or f"未找到与 '{query}' 相关的行业术语。"


@mcp.tool()
def get_resume_template(job_type: str) -> str:
    """获取针对特定岗位的优秀简历模板和写作范例。

    Args:
        job_type: 岗位类型（如 后端开发、数据分析、产品经理）
    """
    vstore = _get_vector_store()
    query = f"{job_type} 简历模板 自我评价 工作经历"
    result = vstore.retrieve(query, k=3)
    return result or f"未找到 '{job_type}' 相关的简历模板。"


@mcp.tool()
def evaluate_resume_score(resume_text: str, job_requirements: str) -> str:
    """评估简历与岗位要求的匹配度，返回评分和修改建议。

    Args:
        resume_text: 简历全文
        job_requirements: 岗位核心要求（关键词列表）
    """
    llm = _get_llm()
    prompt = f"""请快速评估以下简历与岗位要求的匹配度。

岗位要求：{job_requirements}
简历内容：{resume_text[:1500]}

请从以下维度打分（每项 0-10）并给出简短修改建议：
1. 关键词覆盖度
2. 量化成果
3. 语言专业度

输出格式：
关键词覆盖度: X/10
量化成果: X/10
语言专业度: X/10
修改建议: <一句话建议>"""
    response = llm.invoke(prompt)
    return response.content


@mcp.tool()
def get_interview_question_bank(topic: str) -> str:
    """检索特定技术领域的常见面试题。

    Args:
        topic: 技术领域（如 Python、系统设计、数据库）
    """
    try:
        from config import load_config
        from src.interview.question_bank import InterviewQuestionBankManager

        manager = InterviewQuestionBankManager(load_config())
        result, _ = manager.retrieve_questions(f"{topic} 面试 面试题 技能", k=6)
        if result:
            return result
    except Exception as e:
        fallback_error = str(e)
    else:
        fallback_error = ""

    vstore = _get_vector_store()
    query = f"{topic} 面试 面试题 技能"
    result = vstore.retrieve(query, k=3)
    suffix = f"（题库检索异常：{fallback_error}）" if fallback_error else ""
    return result or f"未找到 '{topic}' 相关的面试题。{suffix}"


@mcp.tool()
def web_search(query: str) -> str:
    """联网搜索实时信息。查找最新面试题、面经、技术趋势。

    Args:
        query: 搜索关键词（如 Python后端面试题2025）
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return f"未找到与 '{query}' 相关的搜索结果。"
        snippets = []
        for i, r in enumerate(results, 1):
            snippets.append(f"[{i}] {r.get('title', '')}\n{r.get('body', '')}\n{r.get('href', '')}")
        return "\n\n".join(snippets)
    except Exception as e:
        return f"搜索失败：{str(e)}"


# ═══════════════════════ 入口 ═══════════════════════

if __name__ == "__main__":
    mcp.run(transport="stdio")
