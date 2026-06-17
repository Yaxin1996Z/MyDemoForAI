"""
手写多 Agent 编排 —— 理解 CrewAI 的本质
=========================================

CrewAI 的本质 = 多个 LLM 角色 + 任务依赖链 + 工具

  手写版                  CrewAI
  ─────────              ─────────
  Agent(role, goal)  →   Agent(role, goal)
  Task(desc, agent)  →   Task(desc, agent)
  Crew(agents, tasks) →  Crew(agents, tasks)
  crew.kickoff()     →   crew.kickoff()

你学过的拓扑排序 → 任务依赖链就是一个 DAG
你学过的 LangGraph StateGraph → Agent 之间的状态传递
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import os
import re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()


# ============================================================
# LLM 调用
# ============================================================

def call_llm(prompt: str, system: str = "") -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.3,
        max_tokens=2048,
    )
    return resp.choices[0].message.content.strip()


# ============================================================
# Agent（= LLM + System Prompt + Tools）
# ============================================================

class Agent:
    """一个 Agent = 身份设定 + 目标 + 可用工具"""

    def __init__(self, role: str, goal: str, tools: list = None):
        self.role = role
        self.goal = goal
        self.tools = tools or []

    def build_system_prompt(self) -> str:
        """生成 system prompt"""
        prompt = f"你是{self.role}。你的目标是{self.goal}。"
        if self.tools:
            tool_descs = "\n".join(
                f"  - {t['name']}: {t['description']}" for t in self.tools
            )
            prompt += f"\n\n你有以下工具可用：\n{tool_descs}\n需要时请调用。"
        return prompt

    def run(self, task: str, context: str = "") -> str:
        """执行任务（带工具调用循环）"""
        system = self.build_system_prompt()

        # 构造本次请求的 prompt
        user_msg = f"当前任务：{task}\n"
        if context:
            user_msg += f"\n可以参考以下上下文：\n{context}\n"

        if self.tools:
            user_msg += (
                "\n如果需要工具，请按以下格式调用：\n"
                "TOOL_CALL: 工具名 | 参数\n\n"
                "然后根据工具返回结果继续你的回答。\n"
            )

        print(f"  \U0001f916 [{self.role}] 正在执行：{task[:40]}...")
        result = call_llm(user_msg, system)

        # 工具调用循环：解析 TOOL_CALL -> 执行 -> 结果送回 LLM
        max_tool_rounds = 3
        for _ in range(max_tool_rounds):
            tool_match = re.search(
                r"TOOL_CALL:\s*(\w+)\s*\|\s*(.+)", result, re.MULTILINE
            )
            if not tool_match:
                break  # 没有工具调用，结束

            tool_name = tool_match.group(1)
            tool_args = tool_match.group(2).strip()

            # 查找并执行工具
            matched_tool = next((t for t in self.tools if t["name"] == tool_name), None)
            if not matched_tool:
                tool_output = f"错误：未知工具 {tool_name}"
            else:
                try:
                    tool_output = str(matched_tool["func"](tool_args))
                except Exception as e:
                    tool_output = f"工具执行错误：{e}"

            print(f"    \U0001f527 调工具 [{tool_name}]({tool_args}) -> {tool_output[:60]}...")

            # 把工具结果喂回 LLM，让它继续
            result = call_llm(
                f"{user_msg}\n\n---\n{result}\n\n工具返回：{tool_output}\n\n请基于工具结果继续你的回答。",
                system,
            )

        return result

    def __repr__(self):
        return f"Agent({self.role})"


# ============================================================
# Task（= 任务描述 + 分配给哪个 Agent）
# ============================================================

class Task:
    """一个任务单元"""

    def __init__(self, description: str, agent: Agent, expected_output: str = ""):
        self.description = description
        self.agent = agent
        self.expected_output = expected_output
        self.output = None

    def execute(self, context: str = "") -> str:
        self.output = self.agent.run(self.description, context)
        return self.output

    def __repr__(self):
        return f"Task({self.description[:30]}... → {self.agent.role})"


# ============================================================
# Crew（= 团队 + 任务链 + 调度器）
# ============================================================

class Crew:
    """
    团队调度器。

    两种流程：
      sequential:  顺序执行，上一个的输出是下一个的上下文
      hierarchical: 层级管理，manager 分配任务给子 Agent
    """

    def __init__(self, agents: list, tasks: list, process: str = "sequential"):
        self.agents = agents
        self.tasks = tasks
        self.process = process
        self._context = ""

    def kickoff(self) -> str:
        """启动任务执行"""
        print(f"\n{'='*50}")
        print(f"🚀 Crew 启动！{len(self.agents)} 个 Agent，{len(self.tasks)} 个任务")
        print(f"流程模式：{self.process}")
        print(f"{'='*50}\n")

        for agent in self.agents:
            print(f"  👤 {agent.role} → {agent.goal}")

        print()

        if self.process == "sequential":
            return self._run_sequential()
        elif self.process == "hierarchical":
            return self._run_hierarchical()
        else:
            raise ValueError(f"未知流程: {self.process}")

    def _run_sequential(self) -> str:
        """串行：任务按顺序执行，前一个输出传给下一个"""
        final_output = ""
        for i, task in enumerate(self.tasks):
            print(f"\n  📋 任务 {i+1}: {task.description[:50]}...")
            output = task.execute(context=self._context)
            print(f"  ✅ 完成 ✓ ({len(output)} 字)")
            # 输出摘要
            print(f"     └─ {output[:80]}...")
            self._context = output  # 传给下一个任务
            final_output = output
        return final_output

    def _run_hierarchical(self) -> str:
        """层级：第一个 Agent 作为 manager，分配任务"""
        if len(self.agents) < 2:
            print("  ⚠️ 层级模式至少需要 2 个 Agent")

        manager = self.agents[0]
        workers = self.agents[1:]

        print(f"  🎯 Manager [{manager.role}] 分配任务...\n")

        for i, worker in enumerate(workers):
            task_desc = self.tasks[i].description if i < len(self.tasks) else "继续处理"
            # Manager 分配任务
            assign_prompt = (
                f"你是一个团队管理者。将以下任务分配给 {worker.role} 执行：\n"
                f"任务：{task_desc}\n"
                f"当前上下文：{self._context[:200] if self._context else '无'}\n"
                f"请明确告诉 {worker.role} 需要做什么。"
            )
            instruction = manager.run(assign_prompt)

            # Worker 执行
            print(f"\n  📋 {worker.role} 收到任务...")
            output = worker.run(task_desc, context=instruction)
            print(f"  ✅ {worker.role} 完成 ✓ ({len(output)} 字)")
            self._context = output

        return self._context


# ============================================================
# 工具（可选，复用你之前写的）
# ============================================================

def create_calculator_tool():
    return {
        "name": "calculator",
        "description": "数学计算，输入表达式返回结果，如 '2 + 3 * 4'",
        "func": lambda expr: str(eval(expr)),
    }


def create_summary_tool():
    return {
        "name": "summarizer",
        "description": "总结长文本的核心要点",
        "func": lambda text: call_llm(f"用三句话总结：{text}", "你是一个摘要助手"),
    }


# ============================================================
# Demo 1：顺序流程 —— 技术文章写作团队
# ============================================================

def demo_sequential():
    print("\n" + "="*60)
    print("📌 Demo 1: 顺序流程 — AI 技术文章写作团队")
    print("="*60)

    # 3 个 Agent
    researcher = Agent(
        role="高级AI技术研究员",
        goal="深入调研技术主题，提取关键信息",
    )
    writer = Agent(
        role="技术文章写手",
        goal="将研究结果写成清晰易懂的技术文章",
    )
    reviewer = Agent(
        role="技术文章审核员",
        goal="检查文章质量，提出修改意见，然后输出修改后的最终版本文章",
    )

    # 3 个任务，串行
    tasks = [
        Task(
            description="调研 RAG（检索增强生成）技术，包括：基本原理、主流方案（Naive RAG / Advanced RAG / Modular RAG）、适用场景。输出调研要点。",
            agent=researcher,
        ),
        Task(
            description="基于调研结果，写一篇 300-500 字的技术文章，面向有编程经验的读者。要求：有代码示例、有架构说明。",
            agent=writer,
        ),
        Task(
            description="审核文章质量，检查：技术准确性、结构清晰度、代码正确性。输出修改后的最终版本文章（不是只给意见，而是输出你审核修改后的完整文章）。",
            agent=reviewer,
        ),
    ]

    crew = Crew(
        agents=[researcher, writer, reviewer],
        tasks=tasks,
        process="sequential",
    )

    final = crew.kickoff()

    print(f"\n{'='*50}")
    print("📄 最终输出：")
    print(f"{'='*50}\n")
    print(final[:500] + "...\n" if len(final) > 500 else final)


# ============================================================
# Demo 2：层级流程 —— Manager 分配任务
# ============================================================

def demo_hierarchical():
    print("\n" + "="*60)
    print("📌 Demo 2: 层级流程 — 数学解题团队")
    print("="*60)

    manager = Agent(
        role="数学解题团队负责人",
        goal="分析数学问题，分解步骤，分配给合适的成员",
    )
    solver = Agent(
        role="数学解题专家",
        goal="执行具体的数学计算和推理",
        tools=[create_calculator_tool()],
    )
    verifier = Agent(
        role="数学验证专家",
        goal="验证计算结果的正确性",
        tools=[create_calculator_tool()],
    )

    tasks = [
        Task(description="计算表达式：235 * 47 + 8921 / 3，给出详细步骤", agent=solver),
        Task(description="验证上一步的计算结果是否正确", agent=verifier),
    ]

    crew = Crew(
        agents=[manager, solver, verifier],
        tasks=tasks,
        process="hierarchical",
    )

    final = crew.kickoff()
    print(f"\n📄 最终结果：{final[:200]}...")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("")
    print("  手写多 Agent 编排 - 理解 CrewAI")
    print("")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()

    if args.demo == 1:
        demo_sequential()
    else:
        demo_hierarchical()

    print("\n" + "="*60)
    print("✅ 演示完成！")
    print()
    print("📌 跟 CrewAI 的对应关系：")
    print("  手写 Agent  → crewai.Agent(role, goal, tools)")
    print("  手写 Task   → crewai.Task(description, agent)")
    print("  手写 Crew   → crewai.Crew(agents, tasks, process)")
    print("  手写 kickoff → crew.kickoff()")
    print()
    print("💡 CrewAI 的本质 = 多个 LLM 角色 + 任务依赖链 + 工具")
    print("   跟 LangGraph 的区别：LangGraph 管状态图，CrewAI 管团队协作")
