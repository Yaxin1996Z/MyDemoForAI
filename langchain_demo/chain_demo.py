"""
LangChain Chain Demo —— 最简单的 Prompt -> LLM -> 输出 流水线

功能：中文翻译成英文
演示：ChatPromptTemplate + ChatDeepSeek + StrOutputParser
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ============================================================
# 1. 定义 LLM（大脑）
# ============================================================
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,
)

# ============================================================
# 2. 定义 Prompt 模板（流水线的输入口）
# ============================================================
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的翻译助手。请将用户输入的内容翻译成{target_language}。"),
    ("human", "{text}"),
])

# ============================================================
# 3. 定义输出解析器（流水线的输出口）
# ============================================================
output_parser = StrOutputParser()

# ============================================================
# 4. 组装 Chain（Pipe 操作符 |）
# ============================================================
# Chain = Prompt -> LLM -> Parser
# 数据流：用户输入 -> 填充模板 -> LLM调用 -> 解析文本输出
chain = prompt | llm | output_parser

# ============================================================
# 5. 运行 Chain
# ============================================================
if __name__ == "__main__":
    # 传入参数
    result = chain.invoke({
        "target_language": "英语",
        "text": "今天天气真好，我想去公园散步。"
    })

    print("=" * 50)
    print("[Chain 执行结果]")
    print("=" * 50)
    print(f"输入: 今天天气真好，我想去公园散步。")
    print(f"输出: {result}")
    print("=" * 50)

    # 再来一次——换个参数，Chain 可以重复使用
    result2 = chain.invoke({
        "target_language": "日语",
        "text": "请给我来一杯美式咖啡。"
    })

    print(f"\n输入: 请给我来一杯美式咖啡。")
    print(f"输出: {result2}")
    print("=" * 50)
