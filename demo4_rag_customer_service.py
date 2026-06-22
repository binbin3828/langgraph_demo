"""
Demo4: 智能客服 - RAG + LangGraph 带记忆

学习要点:
- LangGraph 中集成 RAG (检索增强生成)
- MemorySaver 实现多轮对话记忆
- checkpointer 跨 invoke 保持对话状态
- 将检索节点与生成节点组合成图

场景: 企业智能客服，基于知识库回答问题，支持多轮对话
"""

import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import httpx

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

http_client = httpx.Client(verify=False)

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    http_client=http_client,
)

# ========== 模拟知识库 ==========
KNOWLEDGE_BASE = {
    "退货政策": "我们支持7天无理由退货。商品需保持原包装完好，配件齐全。退货运费由买家承担，质量问题除外。退款将在收到退货后3个工作日内原路退回。",
    "配送时间": "标准配送3-5个工作日，加急配送1-2个工作日。偏远地区可能额外增加2-3天。订单满99元免运费，否则收取8元运费。",
    "会员权益": "银卡会员享9.5折，金卡会员享9折，钻石会员享8.5折。所有会员享有专属客服、生日礼包和优先发货权。积分可兑换优惠券。",
    "支付方式": "支持微信支付、支付宝、银行卡、信用卡和花呗分期。大额订单(>5000元)支持对公转账，请联系客服获取对公账户信息。",
    "保修服务": "电子产品保修1年，家电保修3年，服装鞋帽保修30天。保修期内非人为损坏免费维修。延保服务可在购买时加购。",
}


def simple_retrieve(query: str) -> str:
    """简易检索: 基于关键词匹配（实际项目应使用向量检索）"""
    results = []
    for key, value in KNOWLEDGE_BASE.items():
        for word in key:
            if word in query:
                results.append(f"【{key}】{value}")
                break
    if not results:
        for key, value in KNOWLEDGE_BASE.items():
            for word in value[:20]:
                if word in query and len(word) > 1:
                    results.append(f"【{key}】{value}")
                    break
    return "\n\n".join(results) if results else "未找到相关知识库内容。"


def reducer_messages(old: list, new: list) -> list:
    return old + new


class CustomerServiceState(TypedDict):
    messages: Annotated[list[BaseMessage], reducer_messages]
    context: str
    query: str


def retrieve_knowledge(state: CustomerServiceState) -> dict:
    """节点1: 从知识库检索相关内容"""
    last_msg = state["messages"][-1]
    query = last_msg.content if isinstance(last_msg, HumanMessage) else ""
    context = simple_retrieve(query)
    print(f"[检索] 查询: {query[:30]}... | 找到上下文: {'是' if '未找到' not in context else '否'}")
    return {"context": context, "query": query}


def generate_answer(state: CustomerServiceState) -> dict:
    """节点2: 基于检索结果生成回答"""
    context = state.get("context", "")
    history = state["messages"][:-1]
    query = state["query"]

    system_prompt = f"""你是企业智能客服。请根据知识库内容回答用户问题。
如果知识库中没有相关信息，请诚实说"抱歉，我暂时无法回答这个问题，建议联系人工客服"。
回答要简洁友好。

知识库内容:
{context}"""

    messages = [SystemMessage(content=system_prompt)]
    for msg in history[-6:]:
        messages.append(msg)
    messages.append(HumanMessage(content=query))

    result = llm.invoke(messages)
    print(f"[生成] 回答长度: {len(result.content)}字")
    return {"messages": [AIMessage(content=result.content)]}


# ========== 构建状态图 ==========
graph = StateGraph(CustomerServiceState)

graph.add_node("retrieve", retrieve_knowledge)
graph.add_node("generate", generate_answer)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# ========== 运行示例 ==========
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "customer-001"}}

    questions = [
        "你们的退货政策是什么？",
        "那运费怎么算？",
        "我是金卡会员有什么优惠？",
        "你们支持花呗吗？",
    ]

    print("=" * 60)
    print("智能客服对话 (多轮记忆)")
    print("=" * 60)

    for q in questions:
        print(f"\n👤 用户: {q}")
        result = app.invoke({
            "messages": [HumanMessage(content=q)],
            "context": "",
            "query": "",
        }, config=config)

        last_ai = [m for m in result["messages"] if isinstance(m, AIMessage)][-1]
        print(f"🤖 客服: {last_ai.content[:150]}...")

