"""
LangGraph Demo —— 手动构建 StateGraph，理解 Agent 内部运行机制

对比 create_agent（快捷方式）vs StateGraph（手动实现）：
  create_agent(llm, tools) 内部就是帮我们做了下面这些事情。
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Literal, Sequence, Annotated

# ============================================================
# 0. 准备 LLM 和 Tool（和之前一样）
# ============================================================
llm = ChatDeepSeek(model="deepseek-chat", temperature=0.3)

@tool
def calculator(expression: str) -> str:
    """计算数学表达式的结果"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"

@tool
def get_current_time() -> str:
    """获取当前的日期和时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

tools = [calculator, get_current_time]
# 给 LLM 绑定工具（让模型知道有哪些工具可用）
llm_with_tools = llm.bind_tools(tools)

# ============================================================
# 1. 定义 State（状态）—— 图的"血液"
# ============================================================

class AgentState(TypedDict):
    """整个 Agent 执行过程中传递的状态"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    step_count: int  # 记录当前执行到第几步

# ============================================================
# 2. 定义 Node（节点）—— 图中的每个处理步骤
# ============================================================

def call_llm(state: AgentState) -> AgentState:
    """Node 1: LLM 推理节点（对应 ReAct 的 Thought + Action）"""
    response = llm_with_tools.invoke(state["messages"])
    step = state.get("step_count", 0) + 1
    print(f"\n  [第{step}步 · LLM 思考]")
    if response.content:
        print(f"    文本: {response.content[:60]}...")
    if response.tool_calls:
        for tc in response.tool_calls:
            print(f"    决定调用工具: {tc['name']}({tc['args']})")
    return {"messages": [response], "step_count": step}


def call_tool(state: AgentState) -> AgentState:
    """Node 2: 工具执行节点（对应 ReAct 的 Observation）"""
    last_message = state["messages"][-1]
    tool_messages = []

    for tc in last_message.tool_calls:
        # 根据工具名找到对应的函数并执行
        tool_name = tc["name"]
        tool_args = tc["args"]
        print(f"  [执行工具] {tool_name}({tool_args})")

        if tool_name == "calculator":
            result = calculator.invoke(tool_args["expression"])
        elif tool_name == "get_current_time":
            result = get_current_time.invoke({})
        else:
            result = f"未知工具: {tool_name}"

        print(f"    -> 返回: {result[:60]}...")

        tool_messages.append(ToolMessage(
            content=result,
            tool_call_id=tc["id"]
        ))

    return {"messages": tool_messages}

# ============================================================
# 3. 定义条件路由函数（判断 LLM 的下一步）
# ============================================================

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """
    条件路由：根据 LLM 的输出决定下一步

    如果 LLM 调用了工具 -> 走 tools 节点
    如果 LLM 直接回答了  -> 结束（END）
    """
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"  [路由] LLM 要求调用工具 → 进入 tools 节点")
        return "tools"
    else:
        print(f"  [路由] LLM 已给出最终答案 → 结束")
        return "__end__"

# ============================================================
# 4. 构建图（Graph）—— 把节点连起来
# ============================================================

builder = StateGraph(AgentState)

# 4.1 注册节点
builder.add_node("llm", call_llm)      # LLM 推理节点
builder.add_node("tools", call_tool)   # 工具执行节点

# 4.2 连接边
builder.add_edge(START, "llm")         # 开始 → LLM
builder.add_conditional_edges(         # LLM → 条件判断
    "llm",
    should_continue,
    {"tools": "tools", "__end__": END}
)
builder.add_edge("tools", "llm")       # 工具执行完 → 回到 LLM（这就是 ReAct 循环！）

# 4.3 编译图
graph = builder.compile()

# ============================================================
# 5. 运行
# ============================================================

def run_langgraph(query: str):
    print(f"\n{'=' * 60}")
    print(f"[用户提问] {query}")
    print(f"{'=' * 60}")

    result = graph.invoke({
        "messages": [HumanMessage(content=query)],
        "step_count": 0
    })

    final = result["messages"][-1].content
    total_steps = result["step_count"]
    print(f"\n{'=' * 60}")
    print(f"[最终答案] {final}")
    print(f"[总步数] {total_steps} 步（LLM 调用了 {total_steps - 1} 次工具）")
    print(f"{'=' * 60}")


# ============================================================
# 6. 测试
# ============================================================
if __name__ == "__main__":
    # 测试1：需要调计算器
    run_langgraph("计算 (25 + 37) * 3 - 18 等于多少？")

    # 测试2：不需要工具
    run_langgraph("你好，用中文介绍一下你自己")

    # 测试3：连续调工具
    run_langgraph("如果我有100块钱，买2个单价30块的东西和一个单价15块的东西，还剩多少钱？")
