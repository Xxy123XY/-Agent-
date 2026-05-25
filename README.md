# 求职辅助多智能体 Agent 系统

基于 `LangGraph`、`LangChain`、`ChromaDB` 和 `Streamlit` 构建的求职辅助多智能体项目，围绕同一份职位 JD 和候选人简历，提供 **职位分析、简历优化、模拟面试、面试评估** 的一体化能力。

项目重点不只是调用大模型生成文本，而是把求职场景拆成多个可协作的 Agent，并结合 **RAG 检索增强、面试题向量题库、Supervisor 动态路由、MCP 工具调用** 等机制，让系统更接近一个可演示、可扩展的 Agent 应用。

## 项目亮点

- **多 Agent 协作**：将 `JDAnalyzer`、`Writer`、`Reviewer`、`Interviewer`、`Supervisor` 拆分为独立节点，职责清晰，便于扩展。
- **LangGraph 编排**：在简历优化链路中使用 `Plan-and-Execute + Reflection` 工作流，支持任务拆解、执行、评分和修正。
- **混合检索 RAG**：知识库与面试题库均采用 `向量检索 + 关键词检索 + RRF 融合排序`，兼顾语义和关键词召回。
- **面试题向量题库**：支持题目沉淀、来源标注、使用次数更新和阈值清理，降低重复生成成本。
- **Supervisor 动态路由**：根据题库命中数量、RAG 上下文和任务目标决定是否启用题库、`web_search` 或 LLM 生成。
- **MCP 工具解耦**：将行业术语检索、简历模板检索、简历评分、面试题库检索和联网搜索封装为标准工具服务。
- **可视化演示**：Streamlit 页面实时展示 RAG 命中、题库命中、Supervisor 决策和执行轨迹。

## 功能概览

### 1. 职位分析器

- 输入职位 JD，提取核心技能、硬性要求、软性素质、业务关键词和岗位概要
- 支持 RAG 注入行业术语与知识库上下文
- 计算简历与 JD 的匹配度

### 2. 简历优化师

- 基于对话方式持续修改简历
- 首轮优化支持 `Plan-and-Execute + Reflection`
- 展示 RAG 命中文档、执行步骤和结构化 Agent 结果
- 支持版本回退、保存最终简历

### 3. 模拟面试官

- 基于 JD、简历、RAG 和题库生成个性化面试题
- 支持多轮问答与最终面试评估
- 支持长期记忆注入历史面试摘要
- 展示 Supervisor 路由决策、题库命中和面试题来源

## 系统架构

### Agent 层

- `JDAnalyzer`：将原始 JD 转为结构化岗位画像
- `Writer`：根据 JD、用户要求和检索上下文优化简历
- `Reviewer`：从关键词覆盖、语言质量和岗位匹配度等维度审查结果
- `Interviewer`：生成面试题、发起多轮面试、输出评估报告
- `Supervisor`：在面试题生成阶段决定是否启用题库、RAG 和 `web_search`

### 基础能力层

- `RAG`：负责文档分块、向量化、混合检索与结果组织
- `Interview Question Bank`：独立的面试题向量题库
- `MemoryManager`：存储和检索历史面试摘要
- `MCP Server`：对外暴露标准工具能力

### 前端层

- `Streamlit` 三页签：职位分析 / 简历优化 / 模拟面试

## 检索与题库设计

### 知识库 RAG

项目使用本地知识库提供岗位术语、简历模板和面试准备资料，当前实现包括：

- 文档切分：`chunk_size + chunk_overlap`
- 向量存储：`ChromaDB`
- 向量检索：embedding 相似度召回
- 关键词检索：本地 `TF-IDF`
- 结果融合：`RRF`

### 面试题向量题库

项目将历史生成题和 `web_search` 获取的真实面经沉淀到独立题库中，每条题目会记录：

- `question_id`
- `category`
- `topic`
- `source`
- `expected_points`
- `quality_score`
- `used_count`
- `created_at / last_used_at`

当题库命中足够时，Supervisor 会优先复用题库内容，减少联网搜索和重复生成。

## MCP 工具设计

当前 MCP 服务位于 [src/mcp/server.py](src/mcp/server.py)，通过 `stdio` 提供工具能力，客户端适配器位于 [src/mcp/client.py](src/mcp/client.py)。

当前内置工具包括：

- `search_industry_terms`
- `get_resume_template`
- `evaluate_resume_score`
- `get_interview_question_bank`
- `web_search`

如果 MCP 启动失败，系统会自动回退到本地工具实现，保证主流程可用。

## 项目目录

```text
.
├── app.py                         # Streamlit 入口
├── config.py                      # 模型与环境配置
├── components/                    # 侧边栏、展示组件、上传解析
├── tabs/
│   ├── analyze.py                 # 职位分析页
│   ├── optimize.py                # 简历优化页
│   └── interview.py               # 模拟面试页
├── src/
│   ├── agents/                    # 各类 Agent 节点
│   ├── orchestrator/              # LangGraph 工作流
│   ├── rag/                       # 知识库与混合检索
│   ├── interview/                 # 面试题向量题库
│   ├── memory/                    # 长期记忆模块
│   ├── mcp/                       # MCP 服务端 / 客户端
│   ├── tools/                     # LangChain 工具封装
│   ├── runtime/                   # Registry 与运行描述
│   └── utils/                     # 输出解析等通用工具
├── data/
│   ├── knowledge_base/            # 本地知识库
│   ├── chroma_store/              # 知识库向量索引
│   ├── interview_question_bank/   # 面试题题库索引
│   ├── memory_store/              # 长期记忆向量存储
│   └── uploads/                   # 上传后的 JD / 简历缓存
├── docs/                          # 项目文档与面试问答
└── scripts/                       # 导出和辅助脚本
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，并填写你的模型配置：

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_FAST_MODEL=gpt-4o-mini
OPENAI_TOOL_MODEL=gpt-4o-mini

# 可选：单独指定 embedding 服务
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=
EMBEDDING_MODEL=embedding-3
```

项目支持 **OpenAI-compatible** 接口，聊天模型和 embedding 模型可以来自不同服务商。

### 3. 启动应用

```bash
streamlit run app.py
```

默认会在本地打开 Streamlit 页面。首次启用知识库检索时，会自动构建向量索引。

## 常用配置项

| 配置项 | 说明 | 默认值 |
| --- | --- | --- |
| `RAG_TOP_K` | 知识库 RAG 返回条数 | `3` |
| `RAG_CHUNK_SIZE` | 文档切块大小 | `800` |
| `RAG_CHUNK_OVERLAP` | 文档切块重叠长度 | `120` |
| `RAG_RRF_K` | RRF 融合参数 | `60` |
| `MEMORY_TOP_K` | 记忆检索返回条数 | `3` |
| `INTERVIEW_QUESTION_COUNT` | 面试题生成数量 | `12` |
| `INTERVIEW_BANK_TOP_K` | 面试题题库返回条数 | `6` |
| `INTERVIEW_BANK_MIN_HITS` | Supervisor 触发 web_search 的阈值 | `4` |
| `INTERVIEW_BANK_MAX_SIZE` | 面试题题库最大容量 | `1000` |
| `INTERVIEW_WEB_SEARCH_K` | web_search 返回条数 | `5` |

完整配置见 [.env.example](.env.example)。

## 运行模式

侧边栏支持以下执行策略：

- `auto`：推荐模式。职位分析使用 ReAct 思路做检索增强，简历优化使用 `Plan-and-Execute + Reflection`
- `react`：实验模式，强调思考-行动-观察循环
- `plan_exec`：实验模式，强调先规划后执行
- `reflection`：实验模式，强调生成后的审查和修正

同时支持按需启用：

- `RAG 知识库`
- `长期记忆`

## 演示建议

如果你准备把项目放到 GitHub 或简历中展示，推荐按照下面顺序演示：

1. 上传一份 JD 和一份简历
2. 在“职位分析器”里展示结构化 JD 和匹配度
3. 在“简历优化师”里输入修改需求，展示工作流、RAG 命中和版本更新
4. 在“模拟面试官”里生成题目，展示题库命中、Supervisor 决策和最终评估

这样最容易体现这个项目的多 Agent 架构和上下文增强能力。

## 相关文档

- 面试问答整理：[docs/agent_interview_qa.md](docs/agent_interview_qa.md)
- 打印版问答文档：`docs/agent_interview_qa_print_v2.docx`

## 后续可优化方向

- 将当前关键词检索从 `TF-IDF` 升级到 `BM25`
- 增加更标准化的 RAG 离线评测指标，如 `Recall@K / MRR / Precision@K`
- 为长期记忆增加用户身份绑定和分层记忆设计
- 扩展更多 MCP 工具，如公司研究、题目去重、简历 diff 等
- 增加更完整的运行观测能力，如节点级耗时和错误分类

## 适合写在简历上的关键词

`LangGraph` `LangChain` `Streamlit` `ChromaDB` `RAG` `MCP` `Multi-Agent` `Supervisor Routing` `Vector Retrieval` `TF-IDF` `RRF` `OpenAI-compatible API`

## License

如果你准备公开仓库，建议补充 `MIT` 或 `Apache-2.0` 许可证文件。
