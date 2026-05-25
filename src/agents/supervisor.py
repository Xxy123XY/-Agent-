"""Lightweight supervisor/router decisions for agent workflows."""

from __future__ import annotations

from typing import Any


def build_interview_question_query(jd_data: dict[str, Any], resume_text: str) -> str:
    """Build a compact query for question-bank retrieval."""
    core_skills = jd_data.get("core_skills", []) or []
    query_parts = [
        " ".join(core_skills[:5]),
        " ".join((jd_data.get("hard_requirements", []) or [])[:5]),
        " ".join((jd_data.get("keywords", []) or [])[:5]),
        resume_text[:500],
    ]
    return " ".join(part for part in query_parts if part).strip()


def route_interview_question_generation(
    jd_data: dict[str, Any],
    resume_text: str,
    question_bank_hits: list[dict],
    rag_context: str,
    config: dict,
) -> dict[str, Any]:
    """Decide which context sources the interview question agent should use.

    This is intentionally rule-first instead of LLM-first: the supervisor should
    save cost, not add another model call before every generation.
    """
    core_skills = jd_data.get("core_skills", []) or []
    min_bank_hits = int(config.get("interview_bank_min_hits", 4))
    bank_hit_count = len(question_bank_hits or [])
    has_rag = bool((rag_context or "").strip())
    has_enough_bank = bank_hit_count >= min_bank_hits

    use_web_search = bool(core_skills) and not has_enough_bank
    use_question_bank = bank_hit_count > 0
    use_rag_context = has_rag

    reasons = []
    if use_question_bank:
        reasons.append(f"题库命中 {bank_hit_count} 道，可复用历史题降低生成成本")
    else:
        reasons.append("题库暂无有效命中，需要更多外部或 JD/简历上下文")

    if use_web_search:
        reasons.append(f"题库命中低于阈值 {min_bank_hits}，启用 web_search 补充真实面经")
    else:
        reasons.append("题库命中已足够或缺少核心技能，跳过 web_search")

    if use_rag_context:
        reasons.append("存在本地知识库上下文，纳入出题参考")
    else:
        reasons.append("未提供本地知识库上下文，本轮跳过通用 RAG")

    return {
        "route": {
            "question_bank": use_question_bank,
            "web_search": use_web_search,
            "rag_context": use_rag_context,
            "llm_generation": True,
        },
        "bank_hit_count": bank_hit_count,
        "min_bank_hits": min_bank_hits,
        "core_skill_count": len(core_skills),
        "resume_length": len(resume_text or ""),
        "reasons": reasons,
    }
