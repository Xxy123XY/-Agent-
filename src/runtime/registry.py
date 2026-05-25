"""Runtime registry for the multi-agent system.

The registry gives the project a single place to describe available agents,
MCP/local tools, and workflows.  Business code can still call existing
functions directly, but higher-level orchestration now has a clean abstraction
to grow into.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


AgentCallable = Callable[[dict, dict], dict]
WorkflowCallable = Callable[..., dict]


@dataclass(frozen=True)
class AgentSpec:
    name: str
    role: str
    description: str
    entrypoint: AgentCallable
    input_keys: tuple[str, ...] = ()
    output_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    provider: str
    entrypoint: Any


@dataclass(frozen=True)
class WorkflowSpec:
    name: str
    description: str
    entrypoint: WorkflowCallable
    nodes: tuple[str, ...] = ()


@dataclass
class ExecutionContext:
    config: dict
    state: dict = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_trace(self, message: str) -> None:
        self.trace.append(message)


class RuntimeRegistry:
    """In-memory registry for agents, tools, and workflows."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentSpec] = {}
        self._tools: dict[str, ToolSpec] = {}
        self._workflows: dict[str, WorkflowSpec] = {}

    def register_agent(self, spec: AgentSpec) -> None:
        self._agents[spec.name] = spec

    def register_tool(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def register_workflow(self, spec: WorkflowSpec) -> None:
        self._workflows[spec.name] = spec

    def agent(self, name: str) -> AgentSpec:
        return self._agents[name]

    def tool(self, name: str) -> ToolSpec:
        return self._tools[name]

    def workflow(self, name: str) -> WorkflowSpec:
        return self._workflows[name]

    def list_agents(self) -> list[AgentSpec]:
        return list(self._agents.values())

    def list_tools(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def list_workflows(self) -> list[WorkflowSpec]:
        return list(self._workflows.values())


def create_default_registry() -> RuntimeRegistry:
    """Register the current project capabilities."""
    registry = RuntimeRegistry()

    from src.agents.interviewer import (
        conduct_interview_node,
        evaluate_interview_node,
        generate_interview_questions_node,
    )
    from src.agents.supervisor import route_interview_question_generation
    from src.agents.job_analyzer import analyze_job_node, compute_similarity_node
    from src.agents.reviewer import review_and_score_node, review_resume_node
    from src.agents.writer import chat_edit_node, optimize_resume_node
    from src.orchestrator.job_workflow import run_resume_optimization
    from src.tools.job_tools import ALL_TOOLS

    registry.register_agent(
        AgentSpec(
            name="job_analyzer",
            role="analysis",
            description="Parse raw JD text into structured role requirements.",
            entrypoint=analyze_job_node,
            input_keys=("job_description", "rag_context"),
            output_keys=("structured_jd",),
        )
    )
    registry.register_agent(
        AgentSpec(
            name="similarity_analyzer",
            role="analysis",
            description="Compute resume-JD similarity with embedding/TF-IDF fallback.",
            entrypoint=compute_similarity_node,
            input_keys=("original_resume", "structured_jd"),
            output_keys=("resume_jd_similarity",),
        )
    )
    registry.register_agent(
        AgentSpec(
            name="writer",
            role="generation",
            description="Generate or rewrite resumes according to JD and user requirements.",
            entrypoint=optimize_resume_node,
            input_keys=("original_resume", "structured_jd", "user_requirements"),
            output_keys=("optimized_resume",),
        )
    )
    registry.register_agent(
        AgentSpec(
            name="chat_writer",
            role="generation",
            description="Conversational resume editor with structured JSON output.",
            entrypoint=chat_edit_node,
            input_keys=("original_resume", "user_requirements", "rag_context"),
            output_keys=("optimized_resume", "reply"),
        )
    )
    registry.register_agent(
        AgentSpec(
            name="reviewer",
            role="reflection",
            description="Review optimized resume quality and provide revision feedback.",
            entrypoint=review_resume_node,
            input_keys=("optimized_resume", "structured_jd"),
            output_keys=("review_passed", "review_feedback"),
        )
    )
    registry.register_agent(
        AgentSpec(
            name="reviewer_score",
            role="reflection",
            description="Review and score resume quality in one LLM call.",
            entrypoint=review_and_score_node,
            input_keys=("optimized_resume", "structured_jd"),
            output_keys=("review_passed", "reflection_score"),
        )
    )
    registry.register_agent(
        AgentSpec(
            name="interview_supervisor",
            role="supervisor",
            description="Route interview question generation across question bank, RAG, web_search, and LLM generation.",
            entrypoint=route_interview_question_generation,
            input_keys=("structured_jd", "optimized_resume", "interview_question_bank_hits", "rag_context"),
            output_keys=("route", "reasons"),
        )
    )
    registry.register_agent(
        AgentSpec(
            name="interview_questioner",
            role="interview",
            description="Generate targeted interview questions from JD and resume.",
            entrypoint=generate_interview_questions_node,
            input_keys=("structured_jd", "optimized_resume"),
            output_keys=("interview_questions",),
        )
    )
    registry.register_agent(
        AgentSpec(
            name="interviewer",
            role="interview",
            description="Run a multi-turn simulated interview.",
            entrypoint=conduct_interview_node,
            input_keys=("structured_jd", "optimized_resume", "interview_history"),
            output_keys=("interview_history",),
        )
    )
    registry.register_agent(
        AgentSpec(
            name="interview_evaluator",
            role="interview",
            description="Evaluate interview history and produce a structured report.",
            entrypoint=evaluate_interview_node,
            input_keys=("structured_jd", "interview_history"),
            output_keys=("interview_report",),
        )
    )

    for tool in ALL_TOOLS:
        registry.register_tool(
            ToolSpec(
                name=getattr(tool, "name", ""),
                description=getattr(tool, "description", ""),
                provider="local_langchain_tool",
                entrypoint=tool,
            )
        )

    registry.register_workflow(
        WorkflowSpec(
            name="resume_optimization",
            description="LangGraph workflow for RAG/tool-augmented resume optimization.",
            entrypoint=run_resume_optimization,
            nodes=(
                "plan_user_request",
                "search_industry_terms",
                "get_resume_template",
                "retrieve_rag",
                "search_web",
                "write_resume",
                "evaluate_resume_score",
                "reflect_review",
                "finalize",
            ),
        )
    )

    return registry
