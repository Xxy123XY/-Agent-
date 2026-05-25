"""
模拟面试官 Agent —— 负责面试题生成、多轮对话和评估报告三大功能。
"""

import json

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

from src.agents.supervisor import build_interview_question_query, route_interview_question_generation
from src.interview.question_bank import InterviewQuestionBankManager
from src.state import AgentState, InterviewQuestion, InterviewReport
from src.utils.output_parser import parse_json_array


# ── 面试官 System Prompt 模板 ──

INTERVIEW_SYSTEM_PROMPT = """你是一位专业且友善的技术面试官。你的任务是：

1. 基于候选人的简历和岗位 JD 进行有针对性的提问。
2. 根据候选人的回答质量，动态调整提问策略：
   - 如果回答过于浅显（缺少细节、没有实例），进行**深度追问**，要求补充具体案例或技术细节。
   - 如果回答偏题，进行**引导提示**，用一个更具体的问题把候选人拉回正轨。
3. 保持对话自然流畅，每次只问一个问题。
4. 不要一次性抛出多个问题，给候选人充分的回答空间。

当前面试的岗位背景：
{job_context}

候选人的简历概要：
{resume_context}

{memory_section}

请开始面试提问。如果是第一轮，先做一个简短的自我介绍，然后提出第一个问题。"""


# ═══════════════════════ 面试题生成 ═══════════════════════

def generate_interview_questions_node(state: AgentState, config: dict) -> dict:
    """基于结构化 JD 和优化后简历，生成 10-15 道高频面试题。

    Args:
        state: 全局 AgentState。
        config: 配置字典。

    Returns:
        更新 state 的字典。
    """
    llm = config["llm"]
    question_count = config.get("interview_question_count", 12)
    structured_jd = state.get("structured_jd")
    optimized_resume = state.get("optimized_resume", "") or state.get("original_resume", "")
    rag_context = state.get("rag_context", "")

    if not structured_jd:
        return {"error_message": "结构化 JD 缺失，无法生成面试题。"}

    jd_data = structured_jd.model_dump() if hasattr(structured_jd, "model_dump") else structured_jd
    core_skills = jd_data.get("core_skills", [])
    question_query = build_interview_question_query(jd_data, optimized_resume)

    bank_section = ""
    question_bank_hits: list[dict] = []
    bank_added_count = 0
    try:
        question_bank = InterviewQuestionBankManager(config)
        bank_text, question_bank_hits = question_bank.retrieve_questions(
            question_query,
            k=config.get("interview_bank_top_k", 6),
        )
        if bank_text:
            bank_section = f"\n## 向量题库命中（混合检索）\n{bank_text}\n"
    except Exception as e:
        trace = state.get("execution_trace") or []
        trace.append(f"[题库] 检索失败：{str(e)}")

    supervisor_decision = route_interview_question_generation(
        jd_data=jd_data,
        resume_text=optimized_resume,
        question_bank_hits=question_bank_hits,
        rag_context=rag_context,
        config=config,
    )
    route = supervisor_decision["route"]

    if not route["question_bank"]:
        bank_section = ""

    # ── 联网搜索真实面试题作为参考 ──
    web_section = ""
    web_results = []
    if route["web_search"]:
        try:
            from duckduckgo_search import DDGS
            search_query = f"{' '.join(core_skills[:3])} 面试题 面经"
            with DDGS() as ddgs:
                web_results = list(ddgs.text(search_query, max_results=config.get("interview_web_search_k", 5)))
            if web_results:
                web_snippets = []
                for r in web_results:
                    web_snippets.append(f"- {r.get('title', '')}: {r.get('body', '')[:200]}")
                web_section = "\n## 网上真实面试题参考\n" + "\n".join(web_snippets) + "\n"
                try:
                    question_bank = InterviewQuestionBankManager(config)
                    bank_added_count = question_bank.add_web_search_results(
                        web_results,
                        topic=" ".join(core_skills[:3]),
                    )
                except Exception as e:
                    trace = state.get("execution_trace") or []
                    trace.append(f"[题库] web 题目入库失败：{str(e)}")
        except Exception:
            pass  # 搜索失败不阻塞，继续用其他来源

    rag_section = ""
    if route["rag_context"] and rag_context:
        rag_section = f"\n## 本地知识库参考\n{rag_context}\n"

    prompt = f"""你是一位资深技术面试官。请根据以下岗位信息和候选人简历，生成 {question_count} 道高质量面试题。

## 岗位信息
- 核心技能：{', '.join(core_skills)}
- 硬性要求：{', '.join(jd_data.get('hard_requirements', []))}
- 软性素质：{', '.join(jd_data.get('soft_skills', []))}
- 业务关键词：{', '.join(jd_data.get('keywords', []))}
{rag_section}
{bank_section}
{web_section}
## 候选人简历
{optimized_resume}

## 出题要求
1. 题目类别分布：技术基础(40%)、项目经验(30%)、行为面试(20%)、情景题(10%)
2. 每道题需要标注 category、expected_points、source 三个字段
3. 题目应有区分度：包含基础题、进阶题和综合题
4. 贴近真实面试场景，参考网上真实面经，避免纯八股文题目
5. source 字段必须如实标注这道题的灵感来源，只能选以下之一：
   - "网上真实面经"（如果题目参考了网上搜索到的面试题）
   - "向量题库"（如果题目来自上方“向量题库命中”）
   - "本地知识库"（如果题目来自知识库检索结果）
   - "JD分析"（如果题目基于职位描述的核心技能提取）
   - "简历挖掘"（如果题目针对候选人简历中的项目经历）

请输出一个 JSON 数组，每个元素包含 question、category、expected_points、source 四个字段。
直接输出 JSON 数组，不要包裹在 markdown 代码块中。"""

    try:
        response = llm.invoke(prompt)
        questions = parse_json_array(response.content, min_items=1)
        _normalize_question_sources(questions)
        try:
            question_bank = InterviewQuestionBankManager(config)
            question_bank.add_questions(
                [
                    {
                        **q,
                        "topic": " ".join(core_skills[:3]),
                        "source": q.get("source") or "JD分析",
                        "quality_score": 0.75,
                    }
                    for q in questions
                ],
                default_source="JD分析",
            )
            question_bank.update_usage([hit.get("question_id") for hit in question_bank_hits if hit.get("question_id")])
        except Exception as e:
            trace = state.get("execution_trace") or []
            trace.append(f"[题库] 生成题入库失败：{str(e)}")

        trace = state.get("execution_trace") or []
        trace.append(
            "[Supervisor] 路由决策："
            f"题库={'启用' if route['question_bank'] else '跳过'}，"
            f"RAG={'启用' if route['rag_context'] else '跳过'}，"
            f"web_search={'启用' if route['web_search'] else '跳过'}"
        )
        for reason in supervisor_decision.get("reasons", []):
            trace.append(f"[Supervisor] {reason}")
        if question_bank_hits:
            trace.append(f"[题库] 混合检索命中 {len(question_bank_hits)} 道历史题")
        if bank_added_count:
            trace.append(f"[题库] 已从 web_search 新增 {bank_added_count} 道候选题")
        trace.append(f"[出题] 已生成 {len(questions)} 道面试题")

        return {
            "interview_questions": questions,
            "interview_question_bank_hits": question_bank_hits,
            "interview_question_bank_added": bank_added_count,
            "interview_supervisor_decision": supervisor_decision,
            "current_stage": "interview_ready",
            "execution_trace": trace,
            "error_message": None,
        }
    except Exception as e:
        trace = state.get("execution_trace") or []
        trace.append(f"[出题] 失败：{str(e)}")
        return {
            "execution_trace": trace,
            "error_message": f"面试题生成失败：{str(e)}",
        }


def _normalize_question_sources(questions: list[dict]) -> None:
    """Keep source labels stable for UI and later storage."""
    allowed = {"网上真实面经", "向量题库", "本地知识库", "JD分析", "简历挖掘"}
    aliases = {
        "题库": "向量题库",
        "本地题库": "向量题库",
        "面试题库": "向量题库",
        "web_search": "网上真实面经",
        "web": "网上真实面经",
        "RAG": "本地知识库",
    }
    for question in questions:
        source = str(question.get("source") or "JD分析").strip()
        source = aliases.get(source, source)
        if source not in allowed:
            source = "JD分析"
        question["source"] = source
        points = question.get("expected_points", [])
        if isinstance(points, str):
            question["expected_points"] = [p.strip() for p in points.replace("；", ";").split(";") if p.strip()]


# ═══════════════════════ 面试对话 ═══════════════════════

def conduct_interview_node(state: AgentState, config: dict, user_answer: str = "") -> dict:
    """处理一轮面试问答对话。

    - user_answer 为空：发起第一个问题（面试官自我介绍 + 第一问）。
    - user_answer 非空：面试官根据回答做出反应（追问 / 引导 / 下一题）。

    Args:
        state: 全局 AgentState。
        config: 配置字典。
        user_answer: 用户本轮的回答文本。

    Returns:
        更新 state 的字典。
    """
    llm = config["llm"]
    structured_jd = state.get("structured_jd")
    optimized_resume = state.get("optimized_resume", "") or state.get("original_resume", "")
    history = state.get("interview_history", [])
    memory_context = state.get("memory_context", "")

    # 构建 JD 上下文
    if hasattr(structured_jd, "model_dump"):
        jd_context = json.dumps(structured_jd.model_dump(), ensure_ascii=False)
    else:
        jd_context = json.dumps(structured_jd, ensure_ascii=False, default=str)

    # 记忆上下文
    memory_section = ""
    if memory_context:
        memory_section = f"## 该候选人历史面试记录（供参考）\n{memory_context}\n"

    # 构建 System Prompt
    system_msg = SystemMessage(content=INTERVIEW_SYSTEM_PROMPT.format(
        job_context=jd_context,
        resume_context=optimized_resume[:2000],
        memory_section=memory_section,
    ))

    messages = [system_msg] + list(history)

    if user_answer:
        messages.append(HumanMessage(content=user_answer))
    else:
        messages.append(HumanMessage(content="请开始面试。"))

    try:
        response = llm.invoke(messages)
        ai_msg = AIMessage(content=response.content)

        new_history = list(history)
        if user_answer:
            new_history.append(HumanMessage(content=user_answer))
        new_history.append(ai_msg)

        return {
            "interview_history": new_history,
            "current_stage": "interviewing",
            "error_message": None,
        }
    except Exception as e:
        return {"error_message": f"面试对话生成失败：{str(e)}"}


# ═══════════════════════ 面试评估 ═══════════════════════

def evaluate_interview_node(state: AgentState, config: dict) -> dict:
    """根据完整面试对话历史，生成结构化评估报告。

    Args:
        state: 全局 AgentState。
        config: 配置字典。

    Returns:
        更新 state 的字典。
    """
    llm = config["llm"]
    history = state.get("interview_history", [])
    structured_jd = state.get("structured_jd")

    if not history:
        return {"error_message": "面试历史为空，无法生成评估报告。"}

    # 格式化对话
    dialogue = []
    for msg in history:
        if hasattr(msg, "type"):
            role = "候选人" if msg.type == "human" else "面试官"
            dialogue.append(f"{role}: {msg.content}")

    if hasattr(structured_jd, "model_dump"):
        jd_context = json.dumps(structured_jd.model_dump(), ensure_ascii=False)
    else:
        jd_context = json.dumps(structured_jd, ensure_ascii=False, default=str)

    parser = PydanticOutputParser(pydantic_object=InterviewReport)

    prompt = f"""你是一位资深技术面试评估专家。请根据以下面试对话和岗位要求，生成一份结构化评估报告。

## 岗位要求
{jd_context}

## 面试对话记录
{chr(10).join(dialogue)}

{parser.get_format_instructions()}

评分标准：
- technical_depth(1-10)：考察候选人对技术原理的理解深度、解决问题的能力。
- expression_logic(1-10)：回答是否条理清晰、逻辑自洽、重点突出。
- job_match(1-10)：候选人的技能和经验与岗位的匹配程度。
- strengths：列出 3-5 条面试中展现的亮点。
- improvements：列出 3-5 条需要改进的地方。
- overall_suggestion：给出综合评价和后续面试建议。

请严格按 JSON 格式输出。"""

    try:
        response = llm.invoke(prompt)
        report = parser.parse(response.content)

        trace = state.get("execution_trace") or []
        trace.append(
            f"[评估] 技术深度={report.technical_depth}/10, "
            f"表达逻辑={report.expression_logic}/10, "
            f"岗位匹配={report.job_match}/10"
        )

        return {
            "interview_report": report,
            "current_stage": "interview_done",
            "execution_trace": trace,
            "error_message": None,
        }
    except Exception as e:
        trace = state.get("execution_trace") or []
        trace.append(f"[评估] 失败：{str(e)}")
        return {
            "execution_trace": trace,
            "error_message": f"面试评估生成失败：{str(e)}",
        }
