"""
MCP Client Demo —— 通过 stdio 连接 MCP Server，展示 MCP 协议的自动发现和调用

运行方式（MCP 原生，不经过 LLM）：
  python mcp_client.py

流程：启动 Server 子进程 → MCP 自动发现 → 调用工具 → 读取资源
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    # ============================================================
    # 1. 启动 Server 子进程（stdio 模式）
    # ============================================================
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_demo/mcp_server.py"],
    )

    print("\n📡 连接 MCP Server...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # ============================================================
            # 2. 自动发现：Server 暴露了什么能力
            # ============================================================
            print("\n" + "="*50)
            print("  🔍 自动发现 Server 能力")
            print("="*50)

            # 列出 Tools
            tools = await session.list_tools()
            print(f"\n  🛠  Tools（{len(tools.tools)} 个）:")
            for t in tools.tools:
                print(f"     - {t.name}: {t.description}")

            # 列出 Resources
            resources = await session.list_resources()
            print(f"\n  📦  Resources（{len(resources.resources)} 个）:")
            for r in resources.resources:
                print(f"     - {r.uri}: {r.description or '(无描述)'}")

            # ============================================================
            # 3. 调用工具
            # ============================================================
            print("\n" + "="*50)
            print("  🎯 调用工具")
            print("="*50)

            # 3a. 查天气
            print(f"\n  [call] get_weather(location='上海')")
            result = await session.call_tool("get_weather", {"location": "上海"})
            for content in result.content:
                print(f"  [结果] {content.text}")

            # 3b. 计算器
            print(f"\n  [call] calculator(expression='(25+37)*3-18')")
            result = await session.call_tool("calculator", {"expression": "(25+37)*3-18"})
            for content in result.content:
                print(f"  [结果] {content.text}")

            # ============================================================
            # 4. 读取资源
            # ============================================================
            print("\n" + "="*50)
            print("  📖 读取资源")
            print("="*50)

            print(f"\n  [read] config://help")
            result = await session.read_resource("config://help")
            for content in result.contents:
                print(f"  {content.text}")

    print("\n✅ MCP 通信完成\n")


if __name__ == "__main__":
    asyncio.run(main())
