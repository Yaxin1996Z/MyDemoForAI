"""
CrewAI 框架 Demo —— 直接用库，不手写
======================================

对比手写版 handcraft_crew.py：
  手写版 = 理解原理
  本文件 = 用框架，更简洁

配置：通过环境变量指向 DeepSeek（OpenAI 兼容）
  OPENAI_API_KEY    = DEEPSEEK_API_KEY
  OPENAI_BASE_URL   = https://api.deepseek.com/v1
  OPENAI_MODEL_NAME = deepseek-chat
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import os
from dotenv import load_dotenv
load_dotenv()

# 配置 CrewAI 使用 DeepSeek
os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY", "")
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com/v1"
os.environ["OPENAI_MODEL_NAME"] = "deepseek-chat"


# ============================================================
# 导入 CrewAI
# ============================================================

from crewai import Agent, Task, Crew, Process


# ============================================================
# Demo 1：顺序流程 — 跟手写版一样的场景
# ============================================================

def demo_sequential():
    print("\n" + "="*60)
    print("Demo 1: 顺序流程 — AI 技术文章写作团队 (CrewAI)")
    print("="*60)

    # 定义 Agent
    researcher = Agent(
        role="高级AI技术研究员",
        goal="深入调研技术主题，提取关键信息",
        backstory="你是某顶级AI实验室的研究员，擅长快速理解新技术并提炼核心要点。",
        verbose=True,
    )

    writer = Agent(
        role="技术文章写手",
        goal="将研究结果写成清晰易懂的技术文章",
        backstory="你是资深技术写手，擅长把复杂技术概念讲得通俗易懂。",
        verbose=True,
    )

    reviewer = Agent(
        role="技术文章审核员",
        goal="检查文章的质量、准确性和可读性，提出改进意见",
        backstory="你是严谨的技术审核专家，对技术准确性和表达清晰度要求极高。",
        verbose=True,
    )

    # 定义 Task
    research_task = Task(
        description=(
            "调研 MCP（Model Context Protocol）协议，包括：\n"
            "1. 核心概念（Resources / Tools / Prompts）\n"
            "2. 架构设计（Host / Client / Server）\n"
            "3. 传输层（stdio / SSE / Streamable HTTP）\n"
            "4. 与 Function Calling 的关系\n"
            "输出详细的调研要点。"
        ),
        agent=researcher,
        expected_output="一份结构化的 MCP 技术调研报告",
    )

    write_task = Task(
        description=(
            "基于调研结果，写一篇面向开发者的 MCP 入门文章。\n"
            "要求：300 字以上，有代码示例，有架构说明。"
        ),
        agent=writer,
        expected_output="一篇完整的 MCP 技术文章",
    )

    review_task = Task(
        description="审核文章质量，检查技术准确性、结构清晰度，给出具体的修改建议。",
        agent=reviewer,
        expected_output="审核意见列表",
    )

    # 组建 Crew 并运行
    crew = Crew(
        agents=[researcher, writer, reviewer],
        tasks=[research_task, write_task, review_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()

    print(f"\n{'='*50}")
    print("最终输出：")
    print(f"{'='*50}\n")
    print(result)


# ============================================================
# Demo 2：层级流程 — Manager 管理团队
# ============================================================

def demo_hierarchical():
    print("\n" + "="*60)
    print("Demo 2: 层级流程 — 数学解题团队 (CrewAI)")
    print("="*60)

    # Manager Agent
    manager = Agent(
        role="数学解题团队负责人",
        goal="分析问题、分解步骤、分配任务、汇总结果",
        backstory="你是经验丰富的数学老师，擅长把复杂问题拆解成小步骤分配给团队成员。",
        verbose=True,
        allow_delegation=True,
    )

    # 专家 Agent
    calculator = Agent(
        role="计算专家",
        goal="执行精确的数学计算",
        backstory="你擅长各种数学计算，细心且准确。",
        verbose=True,
    )

    verifier = Agent(
        role="验证专家",
        goal="验证计算结果是否正确",
        backstory="你对数字敏感，善于发现计算错误。",
        verbose=True,
    )

    # 任务
    calc_task = Task(
        description="计算 (235 * 47 + 8921) / 3 - 156 * 2，请给出每一步的详细计算过程。",
        agent=calculator,
        expected_output="详细的计算步骤和最终结果",
    )

    verify_task = Task(
        description="验证上一步的计算结果是否正确，如果有误请指出并纠正。",
        agent=verifier,
        expected_output="验证结论（正确或错误及纠正）",
    )

    # 组建 Crew（层级模式需要 manager）
    crew = Crew(
        agents=[calculator, verifier],
        tasks=[calc_task, verify_task],
        process=Process.hierarchical,
        manager_agent=manager,
        verbose=True,
    )

    result = crew.kickoff()

    print(f"\n最终结果：\n{result}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print()
    print("CrewAI 框架 Demo")
    print()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", type=int, choices=[1, 2], default=1,
                        help="1=顺序流程(默认), 2=层级流程")
    args = parser.parse_args()

    if args.demo == 1:
        demo_sequential()
    else:
        demo_hierarchical()

    print("\n" + "="*60)
    print("演示完成！")
    print()
    print("手写版 vs 框架版对比：")
    print("  手写版 handcraft_crew.py  = 理解原理")
    print("  框架版 real_crew.py       = 用 CrewAI，代码更少")
    print()
    print("本质一样：Agent + Task + Crew + kickoff")
