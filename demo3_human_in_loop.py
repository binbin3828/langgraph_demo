"""
Demo3: 审批工作流 - 人机协作 (Human-in-the-Loop)

学习要点:
- MemorySaver: 持久化检查点，支持中断与恢复
- interrupt_before / interrupt: 在指定节点前暂停，等待人工审批
- 从检查点恢复执行
- Command(resume=...): 人工审批后恢复流程

场景: 请假审批系统
  员工提交请假 -> 经理审批(人工) -> HR备案 / 驳回通知
"""

import os
from typing import TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


class LeaveApprovalState(TypedDict):
    employee: str
    leave_type: str
    days: int
    reason: str
    manager_decision: str
    manager_comment: str
    hr_record: str
    notification: str


def submit_leave(state: LeaveApprovalState) -> dict:
    """节点1: 提交请假申请"""
    print(f"[提交申请] {state['employee']} 申请 {state['days']}天{state['leave_type']}")
    print(f"  原因: {state['reason']}")
    return {}


def manager_approve(state: LeaveApprovalState) -> dict:
    """节点2: 经理审批 - 这里会暂停等待人工输入"""
    decision = interrupt(
        f"📋 请假审批待处理:\n"
        f"  员工: {state['employee']}\n"
        f"  类型: {state['leave_type']}\n"
        f"  天数: {state['days']}天\n"
        f"  原因: {state['reason']}\n"
        f"请输入 'approve' 或 'reject':"
    )
    if isinstance(decision, dict):
        return {
            "manager_decision": decision.get("decision", "reject"),
            "manager_comment": decision.get("comment", ""),
        }
    return {"manager_decision": "reject", "manager_comment": "未提供审批"}


def hr_record(state: LeaveApprovalState) -> dict:
    """节点3a: HR备案(审批通过)"""
    prompt = f"""你是HR系统。请为以下请假生成一条备案记录，格式简洁:
员工: {state['employee']}, 类型: {state['leave_type']}, 天数: {state['days']}天, 原因: {state['reason']}"""
    result = llm.invoke([HumanMessage(content=prompt)])
    print(f"[HR备案] 已记录")
    return {"hr_record": result.content, "notification": f"✅ {state['employee']}，你的请假已批准！"}


def reject_notify(state: LeaveApprovalState) -> dict:
    """节点3b: 驳回通知(审批不通过)"""
    notification = f"❌ {state['employee']}，你的请假未批准。原因: {state.get('manager_comment', '未说明')}"
    print(f"[驳回通知] 已发送")
    return {"notification": notification}


def route_by_decision(state: LeaveApprovalState) -> str:
    """条件边: 根据经理决定路由"""
    return "hr_record" if state["manager_decision"] == "approve" else "reject_notify"


# ========== 构建状态图 ==========
graph = StateGraph(LeaveApprovalState)

graph.add_node("submit_leave", submit_leave)
graph.add_node("manager_approve", manager_approve)
graph.add_node("hr_record", hr_record)
graph.add_node("reject_notify", reject_notify)

graph.add_edge(START, "submit_leave")
graph.add_edge("submit_leave", "manager_approve")
graph.add_conditional_edges("manager_approve", route_by_decision, {
    "hr_record": "hr_record",
    "reject_notify": "reject_notify",
})
graph.add_edge("hr_record", END)
graph.add_edge("reject_notify", END)

# 使用 MemorySaver 支持中断与恢复
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# ========== 运行示例 ==========
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "leave-001"}}

    # 第一阶段: 提交请假，流程会在经理审批节点暂停
    print("=" * 60)
    print("第一阶段: 提交请假申请")
    print("=" * 60)
    result = app.invoke({
        "employee": "张三",
        "leave_type": "年假",
        "days": 3,
        "reason": "家中有事需要处理",
        "manager_decision": "",
        "manager_comment": "",
        "hr_record": "",
        "notification": "",
    }, config=config)

    # 模拟经理审批 - 从中断处恢复
    print("\n" + "=" * 60)
    print("第二阶段: 经理审批 (模拟通过)")
    print("=" * 60)
    result = app.invoke(
        Command(resume={"decision": "approve", "comment": "同意，注意安全"}),
        config=config,
    )

    print(f"\n📧 通知: {result['notification']}")
    if result.get('hr_record'):
        print(f"📁 HR备案: {result['hr_record'][:100]}...")
