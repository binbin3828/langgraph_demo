"""
Demo8: Supervisor编排者 - LLM动态路由多Agent

学习要点:
- Supervisor模式: 一个"经理"Agent动态决定调用哪个子Agent
- 与Demo1静态路由的区别: 路由规则由LLM实时决定，而非写死的if-else
- 结构化输出: Supervisor的决策用Pydantic约束，确保路由合法
- 循环调度: Supervisor可以多次调度不同Agent，直到任务完成
- 最大调度次数: 防止Supervisor陷入死循环

场景: 智能项目助手
  用户提需求 → Supervisor(经理)分析需求，决定派给哪个子Agent:
    - Researcher(研究员): 调研分析
    - Coder(程序员): 写代码
    - Writer(文档员): 写文档
  子Agent完成后 → 回到Supervisor → 决定是否需要继续调度 → 直到任务完成
"""

import json
import os
import re
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
import httpx

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

http_client = httpx.Client(verify=False)

llm = ChatOpenAI(
    model="deepseek-v4-pro",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    http_client=http_client,
    max_retries=3,
    timeout=30,
)

MAX_DISPATCH_ROUNDS = 5


def reducer_list(old: list, new: list) -> list:
    return old + new


class SupervisorDecision(BaseModel):
    """Supervisor的结构化决策"""
    reasoning: str = Field(description="决策理由：为什么选择这个Agent")
    next_agent: Literal["researcher", "coder", "writer", "FINISH"] = Field(
        description="下一个要调度的Agent，任务完成时返回FINISH"
    )
    task_description: str = Field(description="给子Agent的任务描述")


class SupervisorState(TypedDict):
    user_request: str
    messages: Annotated[list[str], reducer_list]
    dispatch_count: int
    next_agent: str
    task_description: str
    final_response: str


def _extract_json(text: str) -> dict | None:
    """从LLM响应中提取JSON对象（兼容DeepSeek思考模型的输出格式）"""
    # 尝试匹配 ```json ... ``` 代码块
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试匹配包含 next_agent 字段的 JSON 对象
    match = re.search(r'\{[^{}]*"next_agent"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # 尝试匹配任意 JSON 对象
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def _parse_decision(text: str, user_request: str) -> SupervisorDecision:
    """解析LLM响应为SupervisorDecision，JSON失败时fallback到关键词匹配"""
    json_data = _extract_json(text)

    if json_data:
        try:
            return SupervisorDecision(
                reasoning=json_data.get("reasoning", text[:100]),
                next_agent=json_data.get("next_agent", "FINISH"),
                task_description=json_data.get("task_description", user_request),
            )
        except Exception:
            pass

    # fallback: 关键词匹配
    text_lower = text.lower()
    for key in ["researcher", "coder", "writer"]:
        if key in text_lower:
            return SupervisorDecision(
                reasoning=text[:100],
                next_agent=key,
                task_description=user_request,
            )

    return SupervisorDecision(reasoning=text[:100], next_agent="FINISH", task_description="")


def supervisor(state: SupervisorState) -> dict:
    """节点1: 编排者 - 分析当前状态，决定下一步调度谁"""
    history = "\n".join(f"- {m}" for m in state["messages"]) if state["messages"] else "暂无"

    prompt = f"""你是一个项目经理(Supervisor)，负责协调团队完成用户需求。

你的团队有3个成员:
- researcher: 研究员，擅长调研分析、竞品分析、技术选型
- coder: 程序员，擅长写代码、技术方案设计、架构设计
- writer: 文档员，擅长写文档、用户手册、API文档

用户需求: {state['user_request']}

已完成的工作:
{history}

当前已调度 {state['dispatch_count']} 次。

请决定下一步，只输出一个JSON对象（不要加任何解释）:
{{
    "reasoning": "决策理由",
    "next_agent": "researcher 或 coder 或 writer 或 FINISH",
    "task_description": "给子Agent的具体任务描述"
}}

注意:
- 如果还需要某个Agent工作，next_agent 选择对应成员
- 如果任务已经完成，next_agent 设为 "FINISH"
- 不要重复已完成的工作，每次调度要有明确的增量目标"""

    result = llm.invoke([HumanMessage(content=prompt)])
    decision = _parse_decision(result.content.strip(), state["user_request"])

    next_agent = decision.next_agent
    print(f"[Supervisor] 决策: {next_agent} | 理由: {decision.reasoning[:50]}")

    return {
        "messages": [f"[Supervisor] 调度 {next_agent}: {decision.task_description}"],
        "next_agent": next_agent,
        "task_description": decision.task_description,
    }


def researcher(state: SupervisorState) -> dict:
    """节点2a: 研究员 - 调研分析"""
    task = state.get("task_description", state["user_request"])
    prompt = f"""你是资深研究员。请根据任务要求进行调研分析，给出专业的研究结论。

任务: {task}

请用中文回答，200字以内，给出具体可操作的结论。"""

    result = llm.invoke([HumanMessage(content=prompt)])
    print(f"[研究员] 完成调研")
    return {
        "messages": [f"[研究员] {result.content.strip()}"],
        "dispatch_count": state["dispatch_count"] + 1,
    }


def coder(state: SupervisorState) -> dict:
    """节点2b: 程序员 - 写代码/技术方案"""
    task = state.get("task_description", state["user_request"])
    prompt = f"""你是资深程序员。请根据任务要求给出技术方案或代码实现。

任务: {task}

请用中文回答，200字以内，给出具体的技术方案或核心代码片段。"""

    result = llm.invoke([HumanMessage(content=prompt)])
    print(f"[程序员] 完成技术方案")
    return {
        "messages": [f"[程序员] {result.content.strip()}"],
        "dispatch_count": state["dispatch_count"] + 1,
    }


def writer(state: SupervisorState) -> dict:
    """节点2c: 文档员 - 写文档"""
    task = state.get("task_description", state["user_request"])
    prompt = f"""你是资深技术文档工程师。请根据任务要求撰写文档。

任务: {task}

请用中文回答，200字以内，给出清晰的文档内容。"""

    result = llm.invoke([HumanMessage(content=prompt)])
    print(f"[文档员] 完成文档")
    return {
        "messages": [f"[文档员] {result.content.strip()}"],
        "dispatch_count": state["dispatch_count"] + 1,
    }


def route_from_supervisor(state: SupervisorState) -> str:
    """条件边: 根据Supervisor的决策路由"""
    next_agent = state.get("next_agent", "FINISH")

    if next_agent == "FINISH" or state["dispatch_count"] >= MAX_DISPATCH_ROUNDS:
        if state["dispatch_count"] >= MAX_DISPATCH_ROUNDS:
            print(f"[路由] 已达最大调度次数 {MAX_DISPATCH_ROUNDS}，强制结束")
        else:
            print(f"[路由] Supervisor决定结束")
        return "summarize"

    return next_agent


def summarize(state: SupervisorState) -> dict:
    """节点3: 汇总 - 整理所有Agent的输出，生成最终回复"""
    work_items = []
    for msg in state["messages"]:
        if msg.startswith("[研究员]") or msg.startswith("[程序员]") or msg.startswith("[文档员]"):
            work_items.append(msg)

    work_summary = "\n\n".join(work_items) if work_items else "无"

    prompt = f"""你是项目助理。请根据团队的工作成果，整理一份简洁的最终回复给用户。

用户需求: {state['user_request']}

团队工作成果:
{work_summary}

请用中文回答，直接给出最终回复，不要重复工作过程。"""

    result = llm.invoke([HumanMessage(content=prompt)])
    print(f"[汇总] 生成最终回复")
    return {"final_response": result.content.strip()}


# ========== 构建状态图 ==========
graph = StateGraph(SupervisorState)

graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher)
graph.add_node("coder", coder)
graph.add_node("writer", writer)
graph.add_node("summarize", summarize)

graph.add_edge(START, "supervisor")

graph.add_conditional_edges("supervisor", route_from_supervisor, {
    "researcher": "researcher",
    "coder": "coder",
    "writer": "writer",
    "summarize": "summarize",
})

graph.add_edge("researcher", "supervisor")
graph.add_edge("coder", "supervisor")
graph.add_edge("writer", "supervisor")
graph.add_edge("summarize", END)

app = graph.compile(checkpointer=MemorySaver())

# ========== 运行示例 ==========
if __name__ == "__main__":
    requests = [
        "帮我开发一个用户登录功能，需要调研安全方案、写代码实现、并输出API文档",
    ]

    for req in requests:
        config = {"configurable": {"thread_id": "supervisor-001"}}

        print(f"\n{'='*60}")
        print(f"📋 用户需求: {req}")
        print(f"{'='*60}")

        result = app.invoke({
            "user_request": req,
            "messages": [],
            "dispatch_count": 0,
            "final_response": "",
        }, config=config)

        print(f"\n{'='*60}")
        print("📄 最终回复")
        print("=" * 60)
        print(result["final_response"])
        print(f"\n📊 调度次数: {result['dispatch_count']}")
        print(f"\n📝 调度历史:")
        for msg in result["messages"]:
            if msg.startswith("[Supervisor]"):
                print(f"  {msg}")