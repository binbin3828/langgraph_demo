"""
Demo7: 反思式多Agent协作 - Reflective Multi-Agent Pattern

学习要点:
- 多Agent角色分工: Writer(写作者) + Critic(评论家) + Reviser(修订者)
- 反思循环: 生成 -> 评审 -> 修订 -> 再评审，直到质量达标
- 条件边控制反思轮次，最大反思次数限制
- Annotated reducer 累积评审意见与修订历史
- 结构化输出: with_structured_output 替代字符串解析，评分100%可靠
- 退化保护: 评分严重下降时提前终止，避免越改越差
- 最佳版本保留: 即使最终版本退化，也能输出历史最优稿件
- 持久化/断点恢复: MemorySaver保存反思进度，崩溃后可从断点继续
- 超时/重试: LLM调用超时保护 + API层面自动重试

场景: 文章写作工作流
  用户给题 -> Writer写初稿 -> Critic评审打分 -> 
    如果分数不够 -> Reviser根据意见修订 -> Critic再评审 -> ... 
    如果分数达标 -> 输出终稿
"""

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
    model="deepseek-v4-flash",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    http_client=http_client,
    max_retries=3,
    timeout=30,
)

MAX_REFLECTION_ROUNDS = 3
PASSING_SCORE = 8
DEGRADATION_THRESHOLD = 2


def reducer_list(old: list, new: list) -> list:
    return old + new


class CritiqueResult(BaseModel):
    """结构化评审结果，确保评分解析可靠"""
    strengths: str = Field(description="文章的具体优点")
    weaknesses: str = Field(description="文章的具体不足之处")
    suggestions: str = Field(description="具体可操作的改进建议")
    score: int = Field(description="综合评分1-10，10分最高", ge=1, le=10)


structured_critic_llm = llm.with_structured_output(CritiqueResult, method="function_calling")


class ReflectiveWritingState(TypedDict):
    topic: str
    current_draft: str
    initial_draft: str
    critique: str
    score: int
    best_draft: str
    best_score: int
    prev_score: int
    revision_history: Annotated[list[str], reducer_list]
    critique_history: Annotated[list[str], reducer_list]
    reflection_round: int
    final_article: str
    summary: str


def writer(state: ReflectiveWritingState) -> dict:
    """节点1: 写作者 - 根据主题生成初稿"""
    prompt = f"""你是一位资深内容写作者。请根据以下主题撰写一篇高质量的文章，要求:
1. 观点鲜明，逻辑清晰
2. 语言流畅，有感染力
3. 字数300-500字

主题: {state['topic']}

直接输出文章内容，不要加标题前缀。"""

    result = llm.invoke([HumanMessage(content=prompt)])
    draft = result.content.strip()
    print(f"[写作者] 生成初稿，字数: {len(draft)}")
    return {
        "current_draft": draft,
        "initial_draft": draft,
        "best_draft": draft,
    }


def critic(state: ReflectiveWritingState) -> dict:
    """节点2: 评论家 - 评审当前稿件并打分（结构化输出）"""
    prompt = f"""你是一位严格的文章评审专家。请评审以下文章。

评审标准: 观点深度、逻辑连贯、语言表达、说服力

文章内容:
---
{state['current_draft']}
---

请给出:
- strengths: 具体优点
- weaknesses: 具体不足之处
- suggestions: 具体可操作的改进建议
- score: 综合评分（1-10整数，10分最高）"""

    try:
        result = structured_critic_llm.invoke([HumanMessage(content=prompt)])
        score = max(1, min(10, result.score))
        critique_text = (
            f"优点: {result.strengths}\n"
            f"不足: {result.weaknesses}\n"
            f"建议: {result.suggestions}"
        )
    except Exception:
        critique_text = _fallback_critic(state["current_draft"])
        score = _parse_score_from_text(critique_text)

    round_num = state["reflection_round"] + 1
    print(f"[评论家] 第{round_num}轮评审 | 评分: {score}/10")

    best_draft = state["best_draft"]
    best_score = state["best_score"]
    if score > best_score:
        best_draft = state["current_draft"]
        best_score = score

    return {
        "critique": critique_text,
        "score": score,
        "prev_score": state["score"],
        "best_draft": best_draft,
        "best_score": best_score,
        "critique_history": [f"第{round_num}轮评审 (评分:{score}):\n{critique_text}"],
    }


def _fallback_critic(draft: str) -> str:
    """结构化输出失败时的降级方案"""
    prompt = f"""请评审以下文章，按格式输出:
优点: ...
不足: ...
建议: ...
评分: X (1-10)

文章:
---
{draft}
---"""
    result = llm.invoke([HumanMessage(content=prompt)])
    return result.content.strip()


def _parse_score_from_text(text: str) -> int:
    """从文本中正则提取评分"""
    match = re.search(r'评分[：:]\s*(\d+)', text)
    if match:
        return max(1, min(10, int(match.group(1))))
    return 5


def reviser(state: ReflectiveWritingState) -> dict:
    """节点3: 修订者 - 根据评审意见修改稿件"""
    round_num = state["reflection_round"] + 1
    prompt = f"""你是一位文章修订专家。请根据评审意见修改以下文章，要求:
1. 针对评审中指出的不足逐一改进
2. 保留原文的优点
3. 字数保持在300-500字

原文章:
---
{state['current_draft']}
---

评审意见:
---
{state['critique']}
---

直接输出修改后的完整文章，不要加任何说明。"""

    result = llm.invoke([HumanMessage(content=prompt)])
    revised = result.content.strip()
    print(f"[修订者] 完成第{round_num}轮修订，字数: {len(revised)}")
    return {
        "current_draft": revised,
        "revision_history": [revised],
        "reflection_round": state["reflection_round"] + 1,
    }


def should_continue(state: ReflectiveWritingState) -> Literal["revise", "finalize"]:
    """条件边: 判断是否需要继续反思修订（含退化保护）"""
    if state["score"] >= PASSING_SCORE:
        print(f"[决策] 评分 {state['score']} >= {PASSING_SCORE}，质量达标!")
        return "finalize"
    if state["reflection_round"] >= MAX_REFLECTION_ROUNDS:
        print(f"[决策] 已达最大反思轮次 {MAX_REFLECTION_ROUNDS}，输出当前版本")
        return "finalize"
    if (state["prev_score"] > 0
            and state["score"] <= state["prev_score"] - DEGRADATION_THRESHOLD):
        print(f"[决策] 评分严重退化 ({state['prev_score']}→{state['score']})，提前终止")
        return "finalize"
    print(f"[决策] 评分 {state['score']} < {PASSING_SCORE}，进入第{state['reflection_round'] + 2}轮反思")
    return "revise"


def finalize(state: ReflectiveWritingState) -> dict:
    """节点4: 定稿 - 输出最优版本并生成摘要"""
    use_best = state["score"] < state["best_score"]
    final = state["best_draft"] if use_best else state["current_draft"]
    final_score = state["best_score"] if use_best else state["score"]
    source = "历史最优版本" if use_best else "当前版本"

    summary = (
        f"{'='*50}\n"
        f"  反思式写作完成\n"
        f"{'='*50}\n"
        f"  主题: {state['topic']}\n"
        f"  反思轮次: {state['reflection_round']}\n"
        f"  最终评分: {final_score}/10\n"
        f"  初稿字数: {len(state['initial_draft'])}\n"
        f"  终稿字数: {len(final)}\n"
        f"  终稿来源: {source}\n"
        f"{'='*50}"
    )

    print(f"\n{summary}")
    return {"final_article": final, "summary": summary}


# ========== 构建状态图 ==========
graph = StateGraph(ReflectiveWritingState)

graph.add_node("writer", writer)
graph.add_node("critic", critic)
graph.add_node("reviser", reviser)
graph.add_node("finalize", finalize)

graph.add_edge(START, "writer")
graph.add_edge("writer", "critic")

graph.add_conditional_edges("critic", should_continue, {
    "revise": "reviser",
    "finalize": "finalize",
})

graph.add_edge("reviser", "critic")
graph.add_edge("finalize", END)

app = graph.compile(checkpointer=MemorySaver())

# ========== 运行示例 ==========
if __name__ == "__main__":
    topic = "AI时代，人类的核心竞争力是什么？"
    config = {"configurable": {"thread_id": "reflective-writing-001"}}

    print(f"\n{'='*60}")
    print(f"📝 写作主题: {topic}")
    print(f"{'='*60}")

    result = app.invoke({
        "topic": topic,
        "current_draft": "",
        "initial_draft": "",
        "critique": "",
        "score": 0,
        "best_draft": "",
        "best_score": 0,
        "prev_score": 0,
        "revision_history": [],
        "critique_history": [],
        "reflection_round": 0,
        "final_article": "",
        "summary": "",
    }, config=config)

    print(f"\n{'='*60}")
    print("📄 最终文章")
    print("=" * 60)
    print(result["final_article"])
    print(f"\n📊 反思轮次: {result['reflection_round']} | 最终评分: {result['score']}/10")
    if result["summary"]:
        print(f"\n{result['summary']}")

    # ========== 断点恢复演示 ==========
    print(f"\n\n{'#'*60}")
    print("🔄 断点恢复演示")
    print(f"{'#'*60}")
    print("模拟场景: 第1轮反思后进程崩溃，用相同thread_id恢复执行")
    print("LangGraph会从上次完成的节点继续，不重复已完成的工作\n")

    recovery_config = {"configurable": {"thread_id": "reflective-writing-001"}}
    recovered_result = app.invoke(None, config=recovery_config)

    print(f"[断点恢复] 已恢复，当前反思轮次: {recovered_result['reflection_round']}")
    print(f"[断点恢复] 当前评分: {recovered_result['score']}/10")
    print(f"[断点恢复] 当前稿件字数: {len(recovered_result['current_draft'])}")