"""
Demo10: 多Agent辩论协商 - 平等博弈达成共识

学习要点:
- 辩论模式: 两个对等Agent就同一议题持不同立场，通过多轮辩论达成共识
- 与Demo7的区别: Demo7是上下级(评审→修订)，Demo10是平等博弈(正方↔反方)
- 共识机制: 辩论N轮后由Moderator(主持人)总结双方观点，提炼共识
- Annotated reducer累积完整辩论记录
- 最大辩论轮次限制

场景: 技术方案辩论
  给定一个争议性议题 → 正方(支持)和反方(反对)交替发言 → 
  多轮辩论后 → Moderator总结共识和分歧

辩论流程:
  Proponent(正方) → Opponent(反方) → Proponent → Opponent → ... → Moderator → END
"""

import os
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
import httpx

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

http_client = httpx.Client(verify=False)

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    http_client=http_client,
    max_retries=3,
    timeout=30,
)

MAX_DEBATE_ROUNDS = 3


def reducer_list(old: list, new: list) -> list:
    return old + new


class DebateState(TypedDict):
    topic: str
    current_round: int
    proponent_history: Annotated[list[str], reducer_list]
    opponent_history: Annotated[list[str], reducer_list]
    consensus: str
    dissent: str


def proponent(state: DebateState) -> dict:
    """节点1: 正方 - 支持议题"""
    round_num = state["current_round"] + 1
    opponent_last = state["opponent_history"][-1] if state["opponent_history"] else ""

    print(f"\n{'─'*50}")
    print(f"🟢 正方发言 (第{round_num}轮)")
    print(f"{'─'*50}")

    if state["current_round"] == 0 and not opponent_last:
        prompt = f"""你是一位善于论证的支持者。请就以下议题提出你的开篇立论，阐述支持的理由。

议题: {state['topic']}

要求:
1. 提出2-3个核心论点，每个论点要有理有据
2. 语言简洁有力，150字以内
3. 不要攻击对方，只阐述己方观点"""
    else:
        prompt = f"""你是一位善于论证的支持者。请针对反方的最新发言进行反驳，并重申或补充己方论点。

议题: {state['topic']}

反方最新发言:
{opponent_last}

要求:
1. 直接回应反方的论点，指出其漏洞
2. 补充新的支持论据
3. 150字以内，不要重复之前说过的内容"""

    result = llm.invoke([HumanMessage(content=prompt)])
    argument = result.content.strip()
    print(argument)
    return {"proponent_history": [argument]}


def opponent(state: DebateState) -> dict:
    """节点2: 反方 - 反对议题"""
    round_num = state["current_round"] + 1
    proponent_last = state["proponent_history"][-1]

    print(f"\n{'─'*50}")
    print(f"🔴 反方发言 (第{round_num}轮)")
    print(f"{'─'*50}")

    if state["current_round"] == 0:
        prompt = f"""你是一位善于质疑的反对者。请针对正方的开篇立论提出反驳。

议题: {state['topic']}

正方发言:
{proponent_last}

要求:
1. 逐条反驳正方的论点，指出其不合理之处
2. 提出反对的核心论据
3. 150字以内"""
    else:
        prompt = f"""你是一位善于质疑的反对者。请针对正方的最新发言进行反驳。

议题: {state['topic']}

正方最新发言:
{proponent_last}

要求:
1. 直接回应正方的论点，指出其漏洞
2. 补充新的反对论据
3. 150字以内，不要重复之前说过的内容"""

    result = llm.invoke([HumanMessage(content=prompt)])
    argument = result.content.strip()
    print(argument)
    return {
        "opponent_history": [argument],
        "current_round": state["current_round"] + 1,
    }


def should_continue_debate(state: DebateState) -> Literal["continue", "moderate"]:
    """条件边: 判断是否继续辩论"""
    if state["current_round"] >= MAX_DEBATE_ROUNDS:
        print(f"\n{'═'*50}")
        print(f"⚖️ 已达最大辩论轮次 {MAX_DEBATE_ROUNDS}，请主持人总结")
        print(f"{'═'*50}")
        return "moderate"
    next_round = state["current_round"] + 1
    print(f"\n>>> 进入第{next_round}轮辩论 <<<")
    return "continue"


def moderator(state: DebateState) -> dict:
    """节点3: 主持人 - 总结共识与分歧"""
    pro_args = "\n\n".join(
        f"第{i+1}轮: {arg}" for i, arg in enumerate(state["proponent_history"])
    )
    con_args = "\n\n".join(
        f"第{i+1}轮: {arg}" for i, arg in enumerate(state["opponent_history"])
    )

    prompt = f"""你是一位公正的主持人。双方已就以下议题进行了{state['current_round']}轮辩论，请总结共识与分歧。

议题: {state['topic']}

正方观点:
{pro_args}

反方观点:
{con_args}

请分别输出:
1. 共识: 双方都认同的观点（如有）
2. 分歧: 双方无法达成一致的核心争议点

格式:
共识: ...
分歧: ..."""

    result = llm.invoke([HumanMessage(content=prompt)])
    content = result.content.strip()

    consensus = ""
    dissent = ""
    current_section = ""
    for line in content.split("\n"):
        if "共识" in line and ":" in line:
            current_section = "consensus"
            consensus = line.split(":", 1)[1].strip()
        elif "分歧" in line and ":" in line:
            current_section = "dissent"
            dissent = line.split(":", 1)[1].strip()
        elif current_section == "consensus":
            consensus += line.strip()
        elif current_section == "dissent":
            dissent += line.strip()

    print(f"\n{'═'*50}")
    print(f"⚖️ 主持人总结")
    print(f"{'═'*50}")
    print(f"✅ 共识: {consensus}")
    print(f"❌ 分歧: {dissent}")

    return {"consensus": consensus, "dissent": dissent}


# ========== 构建状态图 ==========
graph = StateGraph(DebateState)

graph.add_node("proponent", proponent)
graph.add_node("opponent", opponent)
graph.add_node("moderator", moderator)

graph.add_edge(START, "proponent")
graph.add_edge("proponent", "opponent")

graph.add_conditional_edges("opponent", should_continue_debate, {
    "continue": "proponent",
    "moderate": "moderator",
})

graph.add_edge("moderator", END)

app = graph.compile(checkpointer=MemorySaver())

# ========== 运行示例 ==========
if __name__ == "__main__":
    topic = "AI是否会取代大部分人类工作？"
    config = {"configurable": {"thread_id": "debate-001"}}

    print(f"\n{'═'*50}")
    print(f"🎤 辩论议题: {topic}")
    print(f"📋 辩论轮次: {MAX_DEBATE_ROUNDS}轮")
    print(f"{'═'*50}")
    print(f"🟢 正方: 支持该观点")
    print(f"🔴 反方: 反对该观点")
    print(f"⚖️ 主持人: 总结共识与分歧")

    result = app.invoke({
        "topic": topic,
        "current_round": 0,
        "proponent_history": [],
        "opponent_history": [],
        "consensus": "",
        "dissent": "",
    }, config=config)

    print(f"\n{'═'*50}")
    print(f"📊 辩论统计")
    print(f"{'═'*50}")
    print(f"总轮次: {result['current_round']}")
    print(f"正方发言次数: {len(result['proponent_history'])}")
    print(f"反方发言次数: {len(result['opponent_history'])}")