# LangGraph Demo

基于 LangGraph 框架的 6 个渐进式学习示例，覆盖从基础图构建到高级并行模式的完整实践。

## 环境要求

- Python >= 3.11
- DeepSeek API Key

## 快速开始

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 DEEPSEEK_API_KEY

# 运行示例
python demo1_intent_router.py
```

## 项目结构

```
.
├── .env.example                    # 环境变量模板
├── requirements.txt                # 依赖列表
├── demo1_intent_router.py          # 示例1：意图分类路由
├── demo2_code_review.py            # 示例2：代码审查流水线
├── demo3_human_in_loop.py         # 示例3：人机协作审批
├── demo4_rag_customer_service.py   # 示例4：RAG智能客服
├── demo5_self_correcting_agent.py  # 示例5：自我纠错Agent
└── demo6_parallel_research.py      # 示例6：并行研究(Map-Reduce)
```

## 示例说明

### Demo1 - 意图分类路由

**核心概念：** StateGraph 基础、条件路由

客服系统根据用户输入自动分类为「技术支持/账单查询/一般咨询」，路由到对应的专业 Agent 处理。

```
START → classify → [tech_support | billing | general] → END
```

**LangGraph 要点：**
- `StateGraph` 创建与编译
- `TypedDict` 定义 State
- `add_node` / `add_edge` / `add_conditional_edges` 构建图
- 条件路由：根据分类结果走不同分支

---

### Demo2 - 代码审查流水线

**核心概念：** 串行节点、Reducer 累积

自动化代码审查流水线，多节点串行协作完成安全审查、风格检查、自动修复和报告生成。

```
START → security_review → style_review → auto_fix → generate_report → END
```

**LangGraph 要点：**
- 多个 Agent 节点串行协作
- `Annotated[list, reducer_list]` 实现 State 累积
- 每个节点负责不同职责

---

### Demo3 - 人机协作审批

**核心概念：** Human-in-the-Loop、检查点暂停/恢复

请假审批系统，经理审批节点暂停等待人工决策，支持审批通过/驳回。

```
START → submit_leave → manager_approve (⏸️ 人工审批) → [hr_record | reject_notify] → END
```

**LangGraph 要点：**
- `MemorySaver` 持久化检查点
- `interrupt()` 在节点中暂停等待人工输入
- `Command(resume=...)` 恢复执行

---

### Demo4 - RAG 智能客服

**核心概念：** RAG 集成、多轮对话记忆

企业智能客服，基于知识库检索增强生成回答，支持多轮对话记忆。

```
START → retrieve → generate → END
```

**LangGraph 要点：**
- 检索节点 + 生成节点组合成图
- `MemorySaver` + `checkpointer` 跨 invoke 保持对话状态
- `Annotated[list[BaseMessage], reducer_messages]` 实现消息历史累积

---

### Demo5 - 自我纠错 Agent

**核心概念：** 循环结构、重试逻辑

数据分析 Agent，生成 SQL → 执行 → 验证，结果有问题则自动反思重试（最多 3 次）。

```
START → generate_sql → execute_sql → validate
                 ↑                        ↓
                 └── increment_retry ←── retry (重试)
                 ↓                        ↓
            max_retries: handle_max_retries → END
            valid: → END
```

**LangGraph 要点：**
- 条件边实现循环结构
- 最大重试次数控制
- 自我反思与纠错模式

---

### Demo6 - 并行研究 Agent

**核心概念：** Map-Reduce 模式、动态并行

市场研究报告生成，输入多个研究主题，并行研究每个主题，汇总生成完整报告。

```
START → [research_topic (并行 N 个实例)] → synthesize → END
```

**LangGraph 要点：**
- `Send()` API 动态创建并行节点实例
- Map-Reduce 模式：并行处理 → 汇总结果
- 多个 State Schema（`OverallState` + `TopicState`）
- Reducer 聚合并行结果

---

## 依赖

| 包名 | 用途 |
|------|------|
| `langgraph` | 图状态管理框架 |
| `langchain-core` | 消息类型、基础抽象 |
| `langchain-openai` | DeepSeek API 的 ChatOpenAI 适配 |
| `python-dotenv` | 加载 .env 环境变量 |

## 配置

在 `.env` 文件中配置：

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | - |
| `DEEPSEEK_MODEL` | 使用的模型名称 | `deepseek-v4-flash` |
