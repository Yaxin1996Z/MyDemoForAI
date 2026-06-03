"""
LangChain Agent Demo —— ReAct 循环：LLM 思考 + 调用工具 + 观察结果

功能：数学计算助手，Agent 自动决定是否用计算器
演示：自定义 Tool + create_agent + ReAct 循环
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool
from langchain.agents import create_agent

# ============================================================
# 1. 定义 Tool（Agent 的"手脚"）
# ============================================================
@tool
def calculator(expression: str) -> str:
    """计算数学表达式的结果。输入应为合法的数学表达式，如 '2 + 3 * 4'"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"

@tool
def get_current_time() -> str:
    """获取当前的日期和时间"""
    from datetime import datetime
    now = datetime.now()
    return f"当前时间是 {now.strftime('%Y年%m月%d日 %H:%M:%S')}"

# 组装工具包
tools = [calculator, get_current_time]

# ============================================================
# 2. 定义 LLM（Agent 的大脑）
# ============================================================
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.3,
)

# ============================================================
# 3. 创建 Agent
# ============================================================
# create_agent 自动构建 ReAct 循环：
#   Thought -> Action（选工具） -> Observation（看结果） -> 循环 -> Final Answer
#
# 注意：LangGraph 1.0+ 已将 create_react_agent 迁移到 langchain.agents
#       新名称改为 create_agent，API 兼容
agent = create_agent(llm, tools)

# ============================================================
# 4. 运行 Agent
# ============================================================
def run_agent(query: str):
    """运行 Agent 并打印详细过程"""
    print(f"\n[用户提问] {query}")
    print("-" * 60)

    result = agent.invoke({"messages": [("human", query)]})

    # 打印完整的执行轨迹
    for msg in result["messages"]:
        role = msg.type.upper()
        content = msg.content

        if role == "HUMAN":
            print(f"\n[用户] {content}")

        elif role == "AI":
            if content:
                print(f"\n[AI 思考] {content}")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"  [调用工具] {tc['name']}({tc['args']})")

        elif role == "TOOL":
            print(f"  [工具返回] {content[:80]}...")

    final = result["messages"][-1].content
    print(f"\n{'=' * 60}")
    print(f"[最终答案] {final}")
    print("=" * 60)

# ============================================================
# 5. 测试
# ============================================================
if __name__ == "__main__":
    # 测试1：需要调计算器
    run_agent("计算 (25 + 37) * 3 - 18 等于多少？")

    # 测试2：不需要工具
    run_agent("用中文介绍一下你自己")

    # 测试3：调时间工具
    run_agent("现在几点了？")

    # 测试4：混合问题
    run_agent("如果我有 150 块钱，买 3 个单价 28 块的东西，还剩多少钱？")
