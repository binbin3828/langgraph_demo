"""
Demo1: 客户意图分类路由 - LangGraph 基础状态图

学习要点:
- StateGraph 的创建与编译
- 定义 State (TypedDict)
- 添加节点 (add_node) 和边 (add_edge / add_conditional_edges)
- 条件路由: 根据分类结果走不同分支

场景: 客服系统，根据用户输入自动分类为"技术支持/账单查询/一般咨询"，
      然后路由到对应的专业Agent处理
"""

import os
from typing import TypedDict, Literal
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


class CustomerServiceState(TypedDict):
    user_input: str
    category: str
    response: str


def classify_intent(state: CustomerServiceState) -> CustomerServiceState:
    """节点1: 意图分类 - 判断用户属于哪类问题"""
    prompt = f"""你是一个客服意图分类器。请将以下用户输入分类为以下三类之一:
- tech_support: 技术支持(产品故障、使用问题、bug反馈)
- billing: 账单查询(费用、退款、发票、订阅)
- general: 一般咨询(其他所有问题)

只回复分类标签，不要任何额外内容。

用户输入: {state['user_input']}"""
    result = llm.invoke([HumanMessage(content=prompt)])
    category = result.content.strip().lower()
    if category not in ["tech_support", "billing", "general"]:
        category = "general"
    print(f"[分类结果] {state['user_input']!r} -> {category}")
    return {**state, "category": category}


def handle_tech_support(state: CustomerServiceState) -> CustomerServiceState:
    """节点2a: 技术支持Agent"""
    prompt = f"""你是技术支持专家。用户遇到了技术问题，请提供专业的技术帮助。
用户问题: {state['user_input']}"""
    result = llm.invoke([HumanMessage(content=prompt)])
    print(f"[技术支持] 已处理")
    return {**state, "response": result.content}


def handle_billing(state: CustomerServiceState) -> CustomerServiceState:
    """节点2b: 账单查询Agent"""
    prompt = f"""你是账单客服专员。用户有账单相关的问题，请耐心解答。
用户问题: {state['user_input']}"""
    result = llm.invoke([HumanMessage(content=prompt)])
    print(f"[账单客服] 已处理")
    return {**state, "response": result.content}


def handle_general(state: CustomerServiceState) -> CustomerServiceState:
    """节点2c: 一般咨询Agent"""
    prompt = f"""你是友好的客服代表。请回答用户的一般咨询。
用户问题: {state['user_input']}"""
    result = llm.invoke([HumanMessage(content=prompt)])
    print(f"[一般咨询] 已处理")
    return {**state, "response": result.content}


def route_by_category(state: CustomerServiceState) -> Literal["tech_support", "billing", "general"]:
    """条件边: 根据分类结果路由到不同节点"""
    return state["category"]


# ========== 构建状态图 ==========
graph = StateGraph(CustomerServiceState)

# 添加节点
graph.add_node("classify", classify_intent)
graph.add_node("tech_support", handle_tech_support)
graph.add_node("billing", handle_billing)
graph.add_node("general", handle_general)

# 添加边: START -> classify
graph.add_edge(START, "classify")

# 添加条件边: classify -> 根据分类结果路由
graph.add_conditional_edges("classify", route_by_category, {
    "tech_support": "tech_support",
    "billing": "billing",
    "general": "general",
})

# 添加边: 各处理节点 -> END
graph.add_edge("tech_support", END)
graph.add_edge("billing", END)
graph.add_edge("general", END)

# 编译图
app = graph.compile()

# ========== 运行示例 ==========
if __name__ == "__main__":
    test_inputs = [
        "我的App打开后一直闪退怎么办？",
        "上个月的扣费为什么多扣了50块？",
        "你们公司周末上班吗？",
    ]

    for user_input in test_inputs:
        print(f"\n{'='*60}")
        print(f"用户: {user_input}")
        result = app.invoke({
            "user_input": user_input,
            "category": "",
            "response": "",
        })
        print(f"分类: {result['category']}")
        print(f"回复: {result['response'][:200]}...")