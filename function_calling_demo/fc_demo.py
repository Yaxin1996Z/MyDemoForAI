"""
Function Calling 手写实战 —— 不用任何框架，直接用 API 实现工具调用循环

对比 LangChain @tool + create_agent：
  @tool        → 帮你生成 JSON Schema 工具定义
  create_agent → 帮你管理 tool_calls → 执行 → 回送的循环

本 demo 手写这两件事，理解底层原理。

API：DeepSeek（兼容 OpenAI 格式）
模型：deepseek-v4-flash
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import json
import os
from openai import OpenAI

# ============================================================
# 0. 初始化客户端
# ============================================================
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


# ============================================================
# 1. 定义工具（裸 JSON Schema —— @tool 帮我们做的事）
# ============================================================
# 每个工具 = type + function(name/description/parameters)
# parameters 是 JSON Schema 格式，描述参数类型和约束
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名，如 上海、北京、深圳",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位，默认 celsius",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式的结果",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 (25+37)*3-18",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "发送邮件给指定收件人",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "收件人邮箱"},
                    "subject": {"type": "string", "description": "邮件主题"},
                    "body": {"type": "string", "description": "邮件正文"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
]

# ============================================================
# 2. 工具执行函数（真正的业务逻辑）
# ============================================================
# 用带参数名的函数定义 —— 和 @tool 一样的风格
# 关键是：如何把 JSON Schema 的 args dict 传入这些函数？
#   看下面的 call_tool() 分发器 —— 用 **kwargs 解包
#   这就是 @tool 在背后做的事


def get_weather(location: str, unit: str = "celsius") -> str:
    """获取指定城市的天气信息"""
    db = {"上海": 25, "北京": 18, "深圳": 30, "哈尔滨": 10}
    temp = db.get(location, 22)
    if unit == "fahrenheit":
        temp = round(temp * 9 / 5 + 32)
        return f"{location} 当前 {temp}°F"
    return f"{location} 当前 {temp}°C，{'☀️' if temp > 20 else '⛅'}"


def calculator(expression: str) -> str:
    """计算数学表达式的结果"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"


def send_email(to: str, subject: str, body: str = "") -> str:
    """发送邮件给指定收件人"""
    return f"邮件已发送至 {to}，主题：「{subject}」"


# ============================================================
# 2.5 工具分发器 —— 这就是 @tool 的核心机制
# ============================================================
# LangChain 的 @tool 装饰器本质上做了两件事：
#   ① 偷看你的函数签名 → 生成 JSON Schema（函数名/参数名/类型/注解）
#   ② 收到 LLM 的 tool_calls 时 → 用 **kwargs 把 args dict 解包传入
#
# 下面就是手写版本：

TOOL_FUNCTIONS: dict[str, callable] = {
    "get_weather": get_weather,
    "calculator": calculator,
    "send_email": send_email,
}


def call_tool(name: str, args: dict) -> str:
    """
    通用工具分发器：把 JSON args dict 解包为函数的命名参数。

    这行就是核心：executor(**args)  ← @tool 帮你做的也是这件事
    """
    executor = TOOL_FUNCTIONS.get(name)
    if not executor:
        return f"错误：未知工具 {name}"
    try:
        return executor(**args)
    except Exception as e:
        return f"工具执行错误：{e}"


# ============================================================
# 3. 工具调用循环（手写 —— create_agent 帮我们做的事）
# ============================================================
def run(query: str, max_turns: int = 5, verbose: bool = True):
    """
    完整 Function Calling 循环：
    ① 调 LLM（带工具描述）
    ② 解析响应 → 有 tool_calls？→ 执行 → 结果回送 → 回到①
    ③ 无 tool_calls？→ 输出最终回答
    """
    messages = [{"role": "user", "content": query}]

    print(f"\n{'='*60}")
    print(f"[用户] {query}")
    print(f"{'='*60}\n")

    for turn in range(max_turns):
        # ---------- ① 调 LLM ----------
        if verbose:
            print(f"── 第 {turn+1} 轮调用 LLM ──")

        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.3,
        )

        msg = response.choices[0].message

        # 把 AI 的回复追加到消息列表
        messages.append(msg)

        # ---------- ② 检查是否有工具调用 ----------
        if not msg.tool_calls:
            # 没有工具调用 → 这是最终回答
            if msg.content:
                print(f"\n[AI 最终回答] {msg.content}")
            print(f"\n{'='*60}\n")
            return msg.content

        # ---------- ③ 有工具调用 → 逐个执行 ----------
        for tc in msg.tool_calls:
            func_name = tc.function.name
            func_args = json.loads(tc.function.arguments)

            if verbose:
                thought = msg.content or "(直接调工具)"
                print(f"  🤔 AI 思考: {thought}")
                print(f"  🛠  调工具: {func_name}({json.dumps(func_args, ensure_ascii=False)})")

            # 执行真实的函数（通过通用分发器解包参数）
            result = call_tool(func_name, func_args)

            if verbose:
                print(f"  📦  结果: {result}\n")

            # 将工具结果作为 ToolMessage 送回 LLM
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    print("[达到最大轮数，结束循环]")
    return messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])


# ============================================================
# 4. 测试场景
# ============================================================
if __name__ == "__main__":
    # 场景1：需要调计算器
    run("计算 (25 + 37) * 3 - 18 等于多少？")

    # 场景2：不需要工具
    run("用中文介绍一下你自己")

    # 场景3：查天气
    run("上海今天天气怎么样？用摄氏度")

    # 场景4：多工具调用 —— 先查天气，判断后再发邮件
    run("北京今天多少度？如果低于15度就发邮件提醒我穿外套，邮箱是 test@example.com")

    # 场景5：一轮多个独立工具（并行调用）
    run("上海天气怎么样？顺便帮我算一下 (8+2)*5 等于多少？")
