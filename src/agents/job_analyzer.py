"""
职位分析器 —— 将非结构化 JD 转化为结构化 JSON，
并计算原始简历与 JD 的向量余弦相似度。
"""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.state import AgentState, StructuredJD


def analyze_job_node(state: AgentState, config: dict) -> dict:
    """职位分析节点：调用 gpt-4o 将非结构化 JD 转为 StructuredJD。

    支持通过 rag_context 注入外部知识（行业术语等），提升提取精度。

    Args:
        state: 全局 AgentState。
        config: 配置字典。

    Returns:
        更新 state 的字典。
    """
    # 已有分析结果时跳过，避免重复 LLM 调用
    if state.get("structured_jd") is not None:
        return {"current_stage": "analyze_done", "error_message": None}

    llm = config["llm"]
    jd_text = state.get("job_description", "").strip()
    rag_context = state.get("rag_context", "")

    if not jd_text:
        return {"error_message": "职位描述为空，请输入有效的 JD 文本。", "current_stage": "analyze"}

    parser = PydanticOutputParser(pydantic_object=StructuredJD)

    # 如果启用了 RAG，在 System Prompt 中注入行业知识上下文
    rag_section = ""
    if rag_context:
        rag_section = f"\n## 参考行业知识（辅助理解 JD 中的专业术语）\n{rag_context}\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位资深的 HR 专家和职位分析师。请仔细阅读以下职位描述（JD），
提取出结构化的关键信息。

{format_instructions}
{rag_section}
要求：
1. core_skills: 提取所有明确要求的技术技能（编程语言、框架、工具等）。
2. hard_requirements: 提取学历、工作年限、证书、行业经验等硬性门槛。
3. soft_skills: 提取沟通、协作、领导力等软性素质要求。
4. keywords: 提取业务领域关键词和行业术语。
5. role_summary: 用一句话概括该岗位的核心职责。

请严格按 JSON 格式输出，不要添加任何额外的文字说明。"""),
        ("human", "以下是职位描述文本：\n\n{job_description}"),
    ]).partial(
        format_instructions=parser.get_format_instructions(),
        rag_section=rag_section,
    )

    try:
        response = llm.invoke(prompt.format(job_description=jd_text))
        structured_jd = parser.parse(response.content)
        # 记录执行轨迹
        trace = state.get("execution_trace") or []
        trace.append(f"[分析] 成功提取 {len(structured_jd.core_skills)} 项核心技能、{len(structured_jd.keywords)} 个关键词")
        return {
            "structured_jd": structured_jd,
            "current_stage": "analyze_done",
            "execution_trace": trace,
            "error_message": None,
        }
    except Exception as e:
        trace = state.get("execution_trace") or []
        trace.append(f"[分析] 失败：{str(e)}")
        return {
            "error_message": f"职位分析失败：{str(e)}",
            "current_stage": "analyze",
            "execution_trace": trace,
        }


# ═══════════════════════ 向量匹配 ═══════════════════════

def compute_similarity_node(state: AgentState, config: dict) -> dict:
    """向量匹配节点：计算原始简历与结构化 JD 的余弦相似度。

    Args:
        state: 全局 AgentState。
        config: 配置字典。

    Returns:
        更新 state 的字典。
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    # 已计算过相似度则跳过
    if state.get("resume_jd_similarity") is not None:
        return {"error_message": None}

    embedding_model = config["embedding_model"]
    resume_text = state.get("original_resume", "")
    structured_jd = state.get("structured_jd")

    if not resume_text or not structured_jd:
        return {"error_message": "简历或结构化 JD 缺失，无法计算相似度。"}

    try:
        # 将 StructuredJD 转为文本
        jd_data = structured_jd.model_dump() if hasattr(structured_jd, "model_dump") else structured_jd
        jd_parts = []
        for key in ["core_skills", "hard_requirements", "soft_skills", "keywords", "role_summary"]:
            val = jd_data.get(key, "")
            if isinstance(val, list):
                jd_parts.append(" ".join(val))
            elif isinstance(val, str):
                jd_parts.append(val)
        jd_text = " ".join(jd_parts)

        try:
            # 优先使用语义 Embedding；网络不可用时降级到本地 TF-IDF，避免主流程中断。
            resume_vec = embedding_model.embed_query(resume_text)
            jd_vec = embedding_model.embed_query(jd_text)
            a = np.array(resume_vec).reshape(1, -1)
            b = np.array(jd_vec).reshape(1, -1)
            similarity = round(float(cosine_similarity(a, b)[0][0]), 4)
            similarity_source = "Embedding"
        except Exception as embedding_error:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
            matrix = vectorizer.fit_transform([resume_text, jd_text])
            similarity = round(float(cosine_similarity(matrix[0], matrix[1])[0][0]), 4)
            similarity_source = f"本地TF-IDF降级（Embedding失败：{embedding_error}）"

        trace = state.get("execution_trace") or []
        trace.append(f"[相似度] 简历与JD余弦相似度 = {similarity:.2%}（{similarity_source}）")

        return {
            "resume_jd_similarity": similarity,
            "execution_trace": trace,
            "error_message": None,
        }
    except Exception as e:
        trace = state.get("execution_trace") or []
        trace.append(f"[相似度] 计算失败：{str(e)}")
        return {
            "resume_jd_similarity": 0.0,
            "execution_trace": trace,
            "error_message": f"相似度计算失败：{str(e)}",
        }
