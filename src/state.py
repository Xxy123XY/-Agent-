"""
全局状态定义 —— 使用 Pydantic 定义 AgentState，
作为 LangGraph 各节点间传递数据的统一载体。

增强内容（v2.0）：
- 新增 execution_mode / rag_enabled / memory_enabled 开关
- 新增 rag_context / memory_context 检索上下文
- 新增 execution_trace 执行轨迹（前端实时展示）
- 新增 plan_steps / reflection_score 范式专属字段
"""

from typing import Annotated, Sequence, TypedDict
try:
    from langgraph.graph.message import add_messages
except ModuleNotFoundError:
    def add_messages(left, right):
        return list(left or []) + list(right or [])
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


# ═══════════════════════ 业务数据模型 ═══════════════════════

class StructuredJD(BaseModel):
    """结构化职位描述"""
    core_skills: list[str] = Field(description="核心技术技能要求，如 Python、Docker")
    hard_requirements: list[str] = Field(description="硬性要求，如学历、工作年限、证书")
    soft_skills: list[str] = Field(description="软性素质要求，如沟通能力、团队协作")
    keywords: list[str] = Field(description="业务关键词，如 供应链金融、风控模型")
    role_summary: str = Field(description="一句话概括岗位核心职责")


class InterviewQuestion(BaseModel):
    """一道面试题"""
    question: str = Field(description="面试题目")
    category: str = Field(description="题目类别：技术基础 / 项目经验 / 行为面试 / 情景题")
    expected_points: list[str] = Field(description="期望回答中应包含的要点")
    source: str = Field(default="JD分析", description="题目来源：网上真实面经 / 本地知识库 / JD分析 / 简历挖掘")


class InterviewReport(BaseModel):
    """结构化面试评估报告"""
    technical_depth: int = Field(description="技术深度评分 1-10")
    expression_logic: int = Field(description="表达逻辑评分 1-10")
    job_match: int = Field(description="岗位匹配度评分 1-10")
    strengths: list[str] = Field(description="面试亮点")
    improvements: list[str] = Field(description="待改进项")
    overall_suggestion: str = Field(description="综合评价与建议")


class PlanStep(BaseModel):
    """Plan-and-Execute 中的单个执行步骤"""
    step_id: int = Field(description="步骤编号")
    name: str = Field(description="步骤名称，如 分析JD关键词")
    description: str = Field(description="步骤详细描述")
    status: str = Field(default="pending", description="执行状态: pending / running / done / failed")
    result: str | None = Field(default=None, description="步骤执行结果")


# ═══════════════════════ LangGraph 全局 AgentState ═══════════════════════

class AgentState(TypedDict):
    """LangGraph 全局状态。

    字段分为三层：
    - 公共层：所有范式共用
    - 范式专属层：仅特定范式读写
    - 基础设施层：RAG / 记忆 / 执行追踪
    """

    # ═══════════════════ 公共层（所有范式共用） ═══════════════════

    # ── 用户输入 ──
    job_description: str
    original_resume: str
    user_requirements: str        # 用户对简历优化的自定义需求

    # ── 运行配置 ──
    execution_mode: str          # "react" | "plan_exec" | "reflection"
    rag_enabled: bool
    memory_enabled: bool

    # ── 职位分析结果（Tab1 产出，所有范式共享） ──
    structured_jd: StructuredJD | None
    resume_jd_similarity: float | None

    # ── 简历优化结果（Tab2 产出） ──
    optimized_resume: str | None

    # ── 面试结果（Tab3 产出） ──
    interview_questions: list[dict] | None
    interview_history: Annotated[Sequence[BaseMessage], add_messages]
    interview_report: InterviewReport | None

    # ── 进度标记 ──
    current_stage: str       # "analyze" | "optimize" | "interview" | "done"
    error_message: str | None


    # ═══════════════════ Reflection 范式专属 ═══════════════════
    # 生成 → 审查 → Critic打分 → {>=7: 输出, <7: 回退重写}

    review_passed: bool                 # 审核是否通过
    review_feedback: str | None         # 审核不通过时的具体修改意见
    reflection_score: int | None        # Critic 自评分数 0-10
    revision_round: int                 # 当前修改轮数


    # ═══════════════════ ReAct 范式专属 ═════════════════════
    # Think → Act → Observe 循环

    react_observation: str | None       # 工具调用返回的观察结果
    react_round: int                    # ReAct 循环当前轮数


    # ═══════════════════ Plan-and-Execute 范式专属 ═══════════
    # Planner 拆解 → Executor 逐步执行 → Aggregator 汇总

    plan_steps: list[dict] | None       # Planner 拆解出的任务列表


    # ═══════════════════ 基础设施层 ═══════════════════════════

    rag_context: str | None             # RAG 检索到的知识片段
    memory_context: str | None          # 长期记忆检索到的历史摘要
    execution_trace: list[str] | None   # Agent 执行轨迹（前端展示）

    
