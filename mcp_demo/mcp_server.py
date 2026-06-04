"""
MCP Server Demo —— 展示 Tools / Resources / Prompts 三大核心概念

运行方式（stdio 模式）：
  python mcp_server.py          # 启动 Server（等待 Client 连接）
  python mcp_demo_client.py     # 另一个终端启动 Client 测试

依赖：fastmcp>=3.3.1
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from fastmcp import FastMCP

# 创建 MCP Server
server = FastMCP("AI 助手工具箱")


# ============================================================
# Tools（工具）—— LLM 可以调用的操作
# ============================================================
@server.tool()
def get_weather(location: str, unit: str = "celsius") -> str:
    """获取指定城市的天气信息"""
    db = {"上海": 25, "北京": 18, "深圳": 30, "哈尔滨": 10}
    temp = db.get(location, 22)
    if unit == "fahrenheit":
        temp = round(temp * 9 / 5 + 32)
        return f"{location} 当前 {temp}°F"
    return f"{location} 当前 {temp}°C，{'☀️' if temp > 20 else '⛅'}"


@server.tool()
def calculator(expression: str) -> str:
    """计算数学表达式的结果"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"


# ============================================================
# Resources（资源）—— LLM 可以读取的数据
# ============================================================
@server.resource("config://help")
def get_help() -> str:
    """Server 使用说明（自动暴露给 Client）"""
    return """
这是一台 MCP 工具箱服务器，提供以下能力：

【工具】
- get_weather(location, unit) — 查天气
- calculator(expression) — 数学计算

【资源】
- config://help — 本说明
- data://tips — 使用技巧
"""


@server.resource("data://tips")
def get_tips() -> str:
    """使用技巧"""
    return """
使用技巧：
1. 查询天气时支持摄氏度和华氏度（unit="fahrenheit"）
2. 计算器支持加减乘除和括号
"""


# ============================================================
# Prompts（提示模板）—— 预设指令
# ============================================================
@server.prompt()
def weather_assistant() -> str:
    """天气助手角色设定"""
    return """你是一位天气助手。
- 当用户查询天气时，调用 get_weather 工具
- 根据温度给出穿衣建议
- 用友好的语气回复"""


if __name__ == "__main__":
    print("🌤  MCP 工具箱服务器启动中...")
    print("    传输方式: stdio\n")
    server.run(transport="stdio")
