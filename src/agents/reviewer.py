"""
简历审核 Agent (Reviewer / Critic) —— 扮演挑剔的 HR 总监，
对优化后的简历进行严格审查，输出 PASS/REVISE 判决和详细反馈。

同时提供 reflection_score_node 用于 Reflection 范式中的自我打分。
"""

from src.agents.base import AgentTimer, failure_result, model_metadata, success_result
from src.state import AgentState
from src.utils.output_parser import parse_json_object


def _build_reviewer_prompt(optimized_resume: str, structured_jd) -> str:
    """构建审核 Agent 的提示词。

    Args:
        optimized_resume: 待审核的简历。
        structured_jd: 结构化 JD。

    Returns:
        组装好的提示词字符串。
    """
    jd_data = structured_jd.model_dump() if hasattr(structured_jd, "model_dump") else structured_jd

    return f"""你是一位极其挑剔的 HR 总监，拥有 15 年招聘经验。请严格审查以下"优化后的简历"。

## 目标岗位要求
- 核心技能：{', '.join(jd_data.get('core_skills', []))}
- 硬性要求：{', '.join(jd_data.get('hard_requirements', []))}
- 关键词：{', '.join(jd_data.get('keywords', []))}

## 审核标准（每一项都要检查）
1. **关键词匹配**：JD 关键词是否在简历中自然出现？是否有生硬堆砌感？
2. **逻辑自洽**：工作经历的时间线、职责描述是否合理？有没有矛盾之处？
3. **量化成果**：每段经历是否包含具体的量化成果（数字、百分比）？
4. **语言流畅**：语句是否通顺自然，像是一个真实的人写的？
5. **针对性**：简历是否明确针对该岗位做了定制？

## 优化后的简历
{optimized_resume}

请按以下格式输出审核结果：

如果审核通过（所有标准都满足），请回复：**PASS**
如果审核不通过，请回复：**REVISE**
然后详细列出需要修改的问题点。"""


def review_resume_node(state: AgentState, config: dict) -> dict:
    """审核节点：审查优化后的简历质量。

    输出 PASS 或 REVISE + 反馈，由 Graph 的条件边决定下一步。

    Args:
        state: 全局 AgentState。
        config: 配置字典。

    Returns:
        更新 state 的字典。
    """
    fast_llm = config["fast_llm"]
    max_rounds = config.get("max_revision_rounds", 3)

    optimized_resume = state.get("optimized_resume", "")
    structured_jd = state.get("structured_jd")
    revision_round = state.get("revision_round", 0)

    if not optimized_resume or not structured_jd:
        return {"error_message": "优化简历或结构化 JD 缺失，无法审核。"}

    review_prompt = _build_reviewer_prompt(optimized_resume, structured_jd)

    try:
        review_response = fast_llm.invoke(review_prompt)
        review_text = review_response.content
    except Exception as e:
        trace = state.get("execution_trace") or []
        trace.append(f"[Reviewer] 审核调用失败：{str(e)}")
        return {
            "review_feedback": f"审核调用失败：{str(e)}",
            "review_passed": False,
            "revision_round": revision_round + 1,
            "execution_trace": trace,
        }

    review_passed = review_text.strip().upper().startswith("PASS")
    new_round = revision_round + 1

    trace = state.get("execution_trace") or []
    trace.append(f"[Reviewer] 第 {new_round} 轮审核：{'PASS' if review_passed else 'REVISE'}")

    # 达到最大轮数仍未通过 → 触发人工介入
    if not review_passed and new_round >= max_rounds:
        trace.append("[Reviewer] 已达最大轮数，触发人工介入")
        return {
            "optimized_resume": optimized_resume,
            "review_feedback": review_text,
            "review_passed": False,
            "revision_round": new_round,
            "current_stage": "need_human_review",
            "execution_trace": trace,
        }

    return {
        "optimized_resume": optimized_resume,
        "review_feedback": review_text if not review_passed else "",
        "review_passed": review_passed,
        "revision_round": new_round,
        "current_stage": "optimize_done" if review_passed else "optimizing",
        "execution_trace": trace,
    }


def review_and_score_node(state: AgentState, config: dict) -> dict:
    """合并的审核+打分节点：一次 LLM 调用同时输出 PASS/REVISE + 0-10 分 + 反馈。
    替代 review_resume_node + reflection_score_node，每轮省一次 LLM 调用。
    """
    timer = AgentTimer()
    fast_llm = config["fast_llm"]
    optimized_resume = state.get("optimized_resume", "")
    structured_jd = state.get("structured_jd")
    revision_round = state.get("revision_round", 0)
    max_rounds = config.get("max_revision_rounds", 3)

    if not optimized_resume or not structured_jd:
        return failure_result(
            agent="reviewer",
            stage="resume_review_score",
            error="缺少简历或 JD",
            metadata=model_metadata(config, key="fast_model_name", latency_ms=timer.elapsed_ms()),
            legacy={"error_message": "缺少简历或 JD"},
        )

    jd_data = structured_jd.model_dump() if hasattr(structured_jd, "model_dump") else structured_jd

    prompt = f"""你是一位极其挑剔的 HR 总监。请审查简历并打出分数。

## 岗位要求
- 核心技能：{', '.join(jd_data.get('core_skills', []))}
- 硬性要求：{', '.join(jd_data.get('hard_requirements', []))}
- 关键词：{', '.join(jd_data.get('keywords', []))}

## 简历
{optimized_resume}

## 审查标准
1. 关键词自然覆盖（非堆砌）
2. 量化成果（数字/百分比）
3. 语言流畅专业
4. 结构清晰有重点

输出一个 JSON：
{{"passed": true/false, "score": 0-10, "feedback": "未通过时的问题点，通过时写一句肯定"}}
只输出 JSON，不要其他文字。"""

    try:
        response = fast_llm.invoke(prompt)
        result = parse_json_object(
            response.content,
            defaults={"passed": False, "score": 5, "feedback": ""},
            required_keys=["passed", "score", "feedback"],
        )

        passed = result.get("passed", False)
        score = result.get("score", 5)
        feedback = result.get("feedback", "")
        new_round = revision_round + 1

        trace = state.get("execution_trace") or []
        trace.append(f"[审核] 第{new_round}轮: {'PASS' if passed else 'REVISE'}, 评分{score}/10")

        if not passed and new_round >= max_rounds:
            trace.append("[审核] 已达最大轮数")

        current_stage = "optimize_done" if passed else ("need_human_review" if new_round >= max_rounds else "optimizing")
        data = {
            "review_passed": passed,
            "review_feedback": feedback if not passed else "",
            "reflection_score": int(score),
            "revision_round": new_round,
            "current_stage": current_stage,
        }
        return success_result(
            agent="reviewer",
            stage="resume_review_score",
            data=data,
            trace=trace,
            metadata=model_metadata(
                config,
                key="fast_model_name",
                parser="json_object",
                latency_ms=timer.elapsed_ms(),
            ),
            legacy={
                **data,
                "execution_trace": trace,
                "error_message": None,
            },
        )
    except Exception as e:
        trace = state.get("execution_trace") or []
        trace.append(f"[审核] 失败：{e}")
        return failure_result(
            agent="reviewer",
            stage="resume_review_score",
            error=f"审核失败：{e}",
            trace=trace,
            metadata=model_metadata(config, key="fast_model_name", latency_ms=timer.elapsed_ms()),
            legacy={"execution_trace": trace, "error_message": f"审核失败：{e}"},
        )


def reflection_score_node(state: AgentState, config: dict) -> dict:
    """Reflection 自评打分节点：Critic 对当前简历质量进行 0-10 打分。

    用于 Reflection 范式：分数 >= 阈值 → 输出；分数 < 阈值 → 重写。

    Args:
        state: 全局 AgentState。
        config: 配置字典。

    Returns:
        更新 state 的字典（包含 reflection_score）。
    """
    fast_llm = config["fast_llm"]
    optimized_resume = state.get("optimized_resume", "")
    structured_jd = state.get("structured_jd")

    if not optimized_resume or not structured_jd:
        return {"error_message": "缺少简历或 JD，无法自评打分。"}

    jd_data = structured_jd.model_dump() if hasattr(structured_jd, "model_dump") else structured_jd

    prompt = f"""你是一位客观公正的简历评审专家。请对以下简历与目标岗位的匹配度打分（0-10分）。

## 目标岗位要求
- 核心技能：{', '.join(jd_data.get('core_skills', []))}
- 关键词：{', '.join(jd_data.get('keywords', []))}

## 简历内容
{optimized_resume}

## 评分维度（每项 0-10）
1. 关键词覆盖度：JD 关键词是否在简历中自然出现
2. 量化成果：是否包含具体的数字和百分比
3. 语言质量：是否流畅、专业、有说服力
4. 结构清晰：是否层次分明、重点突出

请只输出一个 0-10 的整数分数，不要输出任何其他内容。

示例输出格式：
8"""

    try:
        response = fast_llm.invoke(prompt)
        score_text = response.content.strip()
        # 提取数字
        import re
        numbers = re.findall(r'\d+', score_text)
        score = int(numbers[0]) if numbers else 5
        score = max(0, min(10, score))  # clamp [0, 10]

        trace = state.get("execution_trace") or []
        threshold = config.get("reflection_score_threshold", 7)
        status = "合格" if score >= threshold else "不合格"
        trace.append(f"[Critic] 自评分数：{score}/10（{status}，阈值：{threshold}）")

        return {
            "reflection_score": score,
            "execution_trace": trace,
            "error_message": None,
        }
    except Exception as e:
        trace = state.get("execution_trace") or []
        trace.append(f"[Critic] 打分失败：{str(e)}")
        return {
            "reflection_score": 5,
            "execution_trace": trace,
            "error_message": f"自评打分失败：{str(e)}",
        }
