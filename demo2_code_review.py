"""
Demo2: 代码审查工作流 - 串行节点+ reducer累积

学习要点:
- 多个Agent节点串行协作
- State 在节点间传递与累积
- 每个节点负责不同职责(审查/修复/验证)

场景: 自动化代码审查流水线
  代码提交 -> 安全审查 -> 风格审查 -> 自动修复 -> 最终报告
"""

import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import httpx

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

http_client = httpx.Client(verify=False)

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    http_client=http_client,
)


def reducer_list(old: list, new: list) -> list:
    return old + new


class CodeReviewState(TypedDict):
    code: str
    security_issues: Annotated[list[str], reducer_list]
    style_issues: Annotated[list[str], reducer_list]
    fixed_code: str
    final_report: str


def security_review(state: CodeReviewState) -> dict:
    """节点1: 安全审查 - 检查SQL注入、XSS等安全问题"""
    prompt = f"""你是安全审查专家。请审查以下代码中的安全问题(如SQL注入、XSS、硬编码密钥等)。
如果没有问题，回复"无安全问题"。如果有问题，每行列出一个问题。

代码:
```
{state['code']}
```"""
    result = llm.invoke([HumanMessage(content=prompt)])
    issues = [line.strip("- ").strip() for line in result.content.split("\n") if line.strip() and line.strip() != "无安全问题"]
    print(f"[安全审查] 发现 {len(issues)} 个问题")
    return {"security_issues": issues}


def style_review(state: CodeReviewState) -> dict:
    """节点2: 风格审查 - 检查命名规范、代码风格等"""
    prompt = f"""你是代码风格审查专家。请审查以下代码的风格问题(如命名不规范、缺少注释、过长函数等)。
如果没有问题，回复"无风格问题"。如果有问题，每行列出一个问题。

代码:
```
{state['code']}
```"""
    result = llm.invoke([HumanMessage(content=prompt)])
    issues = [line.strip("- ").strip() for line in result.content.split("\n") if line.strip() and line.strip() != "无风格问题"]
    print(f"[风格审查] 发现 {len(issues)} 个问题")
    return {"style_issues": issues}


def auto_fix(state: CodeReviewState) -> dict:
    """节点3: 自动修复 - 根据审查结果尝试修复代码"""
    all_issues = state.get("security_issues", []) + state.get("style_issues", [])
    if not all_issues:
        print("[自动修复] 无需修复")
        return {"fixed_code": state["code"]}

    prompt = f"""你是代码修复专家。根据以下问题列表修复代码，只输出修复后的完整代码，不要解释。

原始代码:
```
{state['code']}
```

问题列表:
{chr(10).join(f'- {issue}' for issue in all_issues)}"""
    result = llm.invoke([HumanMessage(content=prompt)])
    print(f"[自动修复] 已生成修复代码")
    return {"fixed_code": result.content}


def generate_report(state: CodeReviewState) -> dict:
    """节点4: 生成最终审查报告"""
    security_count = len(state.get("security_issues", []))
    style_count = len(state.get("style_issues", []))

    report = f"""{'='*50}
📋 代码审查报告
{'='*50}
🔒 安全问题: {security_count} 个
{chr(10).join(f'  ⚠️ {i}' for i in state.get('security_issues', [])) if security_count else '  ✅ 无安全问题'}

🎨 风格问题: {style_count} 个
{chr(10).join(f'  ⚠️ {i}' for i in state.get('style_issues', [])) if style_count else '  ✅ 无风格问题'}

📝 修复后代码已生成: {'是' if state.get('fixed_code') else '否'}
{'='*50}"""
    print(f"[报告] 审查完成: 安全{security_count}个, 风格{style_count}个")
    return {"final_report": report}


# ========== 构建状态图 ==========
graph = StateGraph(CodeReviewState)

graph.add_node("security_review", security_review)
graph.add_node("style_review", style_review)
graph.add_node("auto_fix", auto_fix)
graph.add_node("generate_report", generate_report)

graph.add_edge(START, "security_review")
graph.add_edge("security_review", "style_review")
graph.add_edge("style_review", "auto_fix")
graph.add_edge("auto_fix", "generate_report")
graph.add_edge("generate_report", END)

app = graph.compile()

# ========== 运行示例 ==========
if __name__ == "__main__":
    test_code = '''def login(username, password):
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    result = db.execute(query)
    if result:
        session["user"] = username
        API_KEY = "sk-1234567890abcdef"
        return True
    return False'''

    print("📝 提交代码审查...")
    result = app.invoke({
        "code": test_code,
        "security_issues": [],
        "style_issues": [],
        "fixed_code": "",
        "final_report": "",
    })

    print(result["final_report"])
    print(result['fixed_code'])

