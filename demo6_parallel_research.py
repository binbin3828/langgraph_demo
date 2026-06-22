"""
Demo6: 并行研究Agent - Map-Reduce模式

学习要点:
- Send() API: 动态创建并行节点实例
- Map-Reduce 模式: 并行处理 -> 汇总结果
- 动态路由: 根据输入数量动态创建节点

场景: 市场研究报告生成
  输入多个研究主题 -> 并行研究每个主题 -> 汇总生成完整报告
"""

import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


def reducer_list(old: list, new: list) -> list:
    return old + new


class OverallState(TypedDict):
    topics: list[str]
    research_results: Annotated[list[str], reducer_list]
    final_report: str


class TopicState(TypedDict):
    topic: str


def route_to_researchers(state: OverallState) -> list[Send]:
    """条件边: 为每个主题创建一个并行研究节点"""
    return [Send("research_topic", {"topic": t}) for t in state["topics"]]


def research_topic(state: TopicState) -> dict:
    """节点: 研究单个主题（会被并行调用）"""
    prompt = f"""你是市场研究分析师。请针对以下主题进行简要分析，包含:
1. 当前市场趋势
2. 主要竞争者
3. 未来6个月预测

主题: {state['topic']}

请用中文回答，200字以内。"""
    result = llm.invoke([HumanMessage(content=prompt)])
    print(f"[研究] 完成主题: {state['topic']}")
    return {"research_results": [f"## {state['topic']}\n{result.content}"]}


def synthesize_report(state: OverallState) -> dict:
    """节点: 汇总所有研究结果生成最终报告"""
    all_research = "\n\n".join(state["research_results"])
    prompt = f"""你是首席分析师。请根据以下各主题的研究结果，生成一份综合市场分析报告。
要求: 添加执行摘要、关键发现和战略建议。

各主题研究:
{all_research}"""
    result = llm.invoke([HumanMessage(content=prompt)])
    print(f"[汇总] 报告生成完成")
    return {"final_report": result.content}


# ========== 构建状态图 ==========
graph = StateGraph(OverallState)

graph.add_node("research_topic", research_topic)
graph.add_node("synthesize", synthesize_report)

graph.add_conditional_edges(START, route_to_researchers, ["research_topic"])
graph.add_edge("research_topic", "synthesize")
graph.add_edge("synthesize", END)

app = graph.compile()

# ========== 运行示例 ==========
if __name__ == "__main__":
    topics = [
        "AI大模型市场",
        "新能源汽车市场",
        "云计算市场",
    ]

    print("=" * 60)
    print(f"📊 并行研究 {len(topics)} 个市场主题")
    print("=" * 60)

    result = app.invoke({
        "topics": topics,
        "research_results": [],
        "final_report": "",
    })

    print(f"\n{'='*60}")
    print("📋 最终报告")
    print("=" * 60)
    print(result["final_report"][:500])