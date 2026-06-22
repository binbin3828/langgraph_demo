"""
Demo5: 自我纠错数据分析Agent - 条件分支 + 循环

学习要点:
- 图中的循环结构 (节点可以回到之前的节点)
- 条件边实现"重试"逻辑
- 最大重试次数控制
- 自我反思与纠错模式

场景: 数据分析Agent，生成SQL查询 -> 执行 -> 验证结果
      如果结果有问题，自动反思并重试（最多3次）
"""

import os
import re
from typing import TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# ========== 模拟数据库 ==========
MOCK_DB = {
    "employees": [
        {"id": 1, "name": "张三", "department": "工程部", "salary": 25000, "join_date": "2022-03-15"},
        {"id": 2, "name": "李四", "department": "工程部", "salary": 30000, "join_date": "2021-06-01"},
        {"id": 3, "name": "王五", "department": "市场部", "salary": 20000, "join_date": "2023-01-10"},
        {"id": 4, "name": "赵六", "department": "市场部", "salary": 22000, "join_date": "2022-09-20"},
        {"id": 5, "name": "钱七", "department": "财务部", "salary": 28000, "join_date": "2020-11-05"},
    ]
}


def execute_mock_sql(sql: str) -> str:
    """模拟SQL执行（仅支持简单的SELECT查询演示）"""
    sql_lower = sql.lower().strip()

    if "avg(salary)" in sql_lower and "department" in sql_lower and "group by" in sql_lower:
        return "工程部: 27500, 市场部: 21000, 财务部: 28000"
    elif "count" in sql_lower and "department" in sql_lower and "group by" in sql_lower:
        return "工程部: 2, 市场部: 2, 财务部: 1"
    elif "max(salary)" in sql_lower:
        return "30000 (李四)"
    elif "min(salary)" in sql_lower:
        return "20000 (王五)"
    elif "avg(salary)" in sql_lower:
        return "25000"
    elif "count" in sql_lower:
        return "5"
    else:
        return "ERROR: 无法解析的SQL查询"


MAX_RETRIES = 3


class DataAnalysisState(TypedDict):
    question: str
    sql: str
    sql_result: str
    analysis: str
    retry_count: int
    is_valid: bool
    error_message: str


def generate_sql(state: DataAnalysisState) -> dict:
    """节点1: 根据问题生成SQL"""
    retry_hint = ""
    if state["retry_count"] > 0:
        retry_hint = f"\n注意: 之前的SQL有问题 - {state.get('error_message', '')}\n请修正SQL。"

    prompt = f"""你是SQL专家。根据用户问题生成一条SQL查询。

数据库表: employees (id, name, department, salary, join_date)
数据示例: 张三/工程部/25000, 李四/工程部/30000, 王五/市场部/20000, 赵六/市场部/22000, 钱七/财务部/28000

只输出SQL语句，不要解释。
{retry_hint}

用户问题: {state['question']}"""

    result = llm.invoke([HumanMessage(content=prompt)])
    sql = result.content.strip()
    sql = re.sub(r'^```sql\s*', '', sql)
    sql = re.sub(r'\s*```$', '', sql)
    print(f"[生成SQL] 第{state['retry_count'] + 1}次尝试: {sql[:80]}...")
    return {"sql": sql}


def execute_sql(state: DataAnalysisState) -> dict:
    """节点2: 执行SQL"""
    result = execute_mock_sql(state["sql"])
    is_valid = "ERROR" not in result
    print(f"[执行SQL] 结果: {result[:60]}... | 有效: {is_valid}")
    return {"sql_result": result, "is_valid": is_valid}


def validate_and_analyze(state: DataAnalysisState) -> dict:
    """节点3: 验证结果并生成分析"""
    if not state["is_valid"]:
        error_msg = f"SQL执行失败: {state['sql_result']}"
        print(f"[验证] ❌ {error_msg}")
        return {"error_message": error_msg, "analysis": ""}

    prompt = f"""你是数据分析专家。根据SQL查询结果，用中文简洁地回答用户问题。

用户问题: {state['question']}
SQL: {state['sql']}
查询结果: {state['sql_result']}"""

    result = llm.invoke([HumanMessage(content=prompt)])
    print(f"[分析] ✅ 生成分析报告")
    return {"analysis": result.content, "error_message": ""}


def should_retry(state: DataAnalysisState) -> str:
    """条件边: 判断是否需要重试"""
    if state["is_valid"]:
        return "done"
    if state["retry_count"] >= MAX_RETRIES:
        print(f"[重试] ❌ 已达最大重试次数 {MAX_RETRIES}")
        return "max_retries"
    return "retry"


def handle_max_retries(state: DataAnalysisState) -> dict:
    """节点: 超过最大重试次数的处理"""
    return {"analysis": f"抱歉，经过{MAX_RETRIES}次尝试仍无法生成正确的SQL查询。请尝试换一种方式描述您的问题。"}


def increment_retry(state: DataAnalysisState) -> dict:
    """重试前增加计数"""
    return {"retry_count": state["retry_count"] + 1}


# ========== 构建状态图 ==========
graph = StateGraph(DataAnalysisState)

graph.add_node("generate_sql", generate_sql)
graph.add_node("execute_sql", execute_sql)
graph.add_node("validate", validate_and_analyze)
graph.add_node("increment_retry", increment_retry)
graph.add_node("handle_max_retries", handle_max_retries)

graph.add_edge(START, "generate_sql")
graph.add_edge("generate_sql", "execute_sql")
graph.add_edge("execute_sql", "validate")

graph.add_conditional_edges("validate", should_retry, {
    "done": END,
    "retry": "increment_retry",
    "max_retries": "handle_max_retries",
})

graph.add_edge("increment_retry", "generate_sql")
graph.add_edge("handle_max_retries", END)

app = graph.compile()

# ========== 运行示例 ==========
if __name__ == "__main__":
    questions = [
        "各部门的平均工资是多少？",
        "哪个部门人数最多？",
        "公司最高薪水的员工是谁？",
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"📊 问题: {q}")
        result = app.invoke({
            "question": q,
            "sql": "",
            "sql_result": "",
            "analysis": "",
            "retry_count": 0,
            "is_valid": False,
            "error_message": "",
        })
        print(f"📝 分析: {result['analysis'][:200]}")
        print(f"🔄 重试次数: {result['retry_count']}")
