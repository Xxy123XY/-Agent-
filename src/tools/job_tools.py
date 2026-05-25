"""
自定义 LangChain 工具集 —— 供 ReAct 范式 Agent 调用。

工具列表：
1. search_industry_terms  — 检索行业术语解释
2. get_resume_template     — 获取优秀简历模板片段
3. evaluate_resume_score   — 对简历与 JD 匹配度打分
4. get_interview_question_bank — 检索面试题库

所有工具使用 @tool 装饰器定义，支持 LangChain Tool Calling。
"""

from langchain_core.tools import tool


# ═══════════════════════ 全局引用（由外部注入） ═══════════════════════
# 这些变量在 init_tools() 中被赋值，避免循环导入
_vector_store = None
_llm = None


def init_tools(vector_store, llm):
    """初始化工具的全局依赖。

    Args:
        vector_store: VectorStoreManager 实例（用于 RAG 检索）。
        llm: gpt-4o 实例（用于打分等需要推理的操作）。
    """
    global _vector_store, _llm
    _vector_store = vector_store
    _llm = llm


# ═══════════════════════ 工具定义 ═══════════════════════

@tool
def search_industry_terms(query: str) -> str:
    """搜索行业术语和黑话的解释。当你遇到不理解的 JD 术语时调用此工具。

    Args:
        query: 要查询的术语或关键词（如 "供应链金融"、"DDD"）。
    """
    if _vector_store is None:
        return "RAG 知识库未启用，无法检索行业术语。"
    result = _vector_store.retrieve(query, k=3)
    return result or f"未找到与 '{query}' 相关的行业术语解释。"


@tool
def get_resume_template(job_type: str) -> str:
    """获取针对特定岗位类型的优秀简历模板和写作范例。
    在优化简历前调用，获取写作参考。

    Args:
        job_type: 岗位类型（如 "后端开发"、"数据分析"、"产品经理"）。
    """
    if _vector_store is None:
        return "RAG 知识库未启用，无法检索简历模板。"
    query = f"{job_type} 简历模板 自我评价 工作经历"
    result = _vector_store.retrieve(query, k=3)
    return result or f"未找到 '{job_type}' 相关的简历模板。"


@tool
def evaluate_resume_score(resume_text: str, job_requirements: str) -> str:
    """评估一份简历与岗位要求的匹配度，返回详细评分和修改建议。
    在简历优化完成后调用，验证优化效果。

    Args:
        resume_text: 简历全文。
        job_requirements: 岗位核心要求（关键词列表）。
    """
    if _llm is None:
        return "LLM 未配置，无法执行评估。"

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

    try:
        response = _llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"评估失败：{str(e)}"


@tool
def get_interview_question_bank(topic: str) -> str:
    """检索特定技术领域的常见面试题。在准备面试题时调用。

    Args:
        topic: 技术领域（如 "Python"、"系统设计"、"数据库"）。
    """
    try:
        from config import load_config
        from src.interview.question_bank import InterviewQuestionBankManager

        manager = InterviewQuestionBankManager(load_config())
        result, hits = manager.retrieve_questions(f"{topic} 面试 面试题 技能", k=6)
        if result:
            return result
    except Exception as e:
        fallback_error = str(e)
    else:
        fallback_error = ""

    if _vector_store is not None:
        query = f"{topic} 面试 面试题 技能"
        result = _vector_store.retrieve(query, k=3)
        if result:
            return result

    suffix = f"（题库检索异常：{fallback_error}）" if fallback_error else ""
    return f"未找到 '{topic}' 相关的面试题。{suffix}"


@tool
def web_search(query: str) -> str:
    """联网搜索实时信息。当你需要查找最新的面试题、面经、技术趋势时调用此工具。

    Args:
        query: 搜索关键词（如 "Python后端面试题2025"、"分布式系统面经"）。
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return f"未找到与 '{query}' 相关的搜索结果。"
        snippets = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            snippets.append(f"[{i}] {title}\n{body}\n链接: {href}")
        return "\n\n".join(snippets)
    except ImportError:
        return "web_search 不可用：请安装 duckduckgo-search。"
    except Exception as e:
        return f"搜索失败：{str(e)}"


# ── 工具列表（供 LangGraph agent 绑定用） ──

ALL_TOOLS = [
    search_industry_terms,
    get_resume_template,
    evaluate_resume_score,
    get_interview_question_bank,
    web_search,
]
