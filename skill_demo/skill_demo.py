"""
Skill 系统 Demo
===============
演示 Agent 三层架构中 Skill（流程层）如何与 Tool（执行层）协同工作。

核心概念：
  Level 1: 技能元信息（名称+描述）-> 常驻上下文，~100 tokens/个
  Level 2: 完整 SKILL.md 指令     -> 匹配到任务才加载
  Level 3: 外部脚本/知识          -> 按需读取
"""

from dataclasses import dataclass
from typing import Callable
import time


# ============================================================
# Tool 层 —— 原子操作（类比 MCP 暴露的工具）
# ============================================================

@dataclass
class Tool:
    """一个工具 = 名字 + 描述 + 可调用函数"""
    name: str
    description: str
    fn: Callable

    def __call__(self, **kwargs) -> str:
        print(f"  [Tool] {self.name}({kwargs})")
        result = self.fn(**kwargs)
        print(f"  [Result] {result}")
        return result


def build_tool_registry() -> dict[str, Tool]:
    """注册所有可用工具"""

    def get_customer(user_id: str) -> str:
        db = {"u001": "ZhangSan(普通会员)", "u002": "LiSi(高级会员)", "u003": "WangWu(普通会员)"}
        return db.get(user_id, "未知用户")

    def get_order(order_id: str) -> str:
        db = {
            "ord001": "订单001: 商品A CNY299, 购买于2026-06-01",
            "ord002": "订单002: 商品B CNY899, 购买于2026-05-15",
            "ord003": "订单003: 商品C CNY1299, 购买于2026-06-10",
        }
        return db.get(order_id, "未知订单")

    def refund(order_id: str, amount: str) -> str:
        return f"退款成功: {order_id}, 金额 CNY{amount}，预计3个工作日到账"

    def escalate(reason: str) -> str:
        return f"已升级人工客服，原因: {reason}，工单号 TK-{int(time.time())}"

    def check_weather(city: str) -> str:
        return f"{city} 今日天气: 晴, 25-32C"

    return {
        "get_customer": Tool("get_customer", "查询用户信息", get_customer),
        "get_order": Tool("get_order", "查询订单详情", get_order),
        "refund": Tool("refund", "执行退款", refund),
        "escalate": Tool("escalate", "升级到人工客服", escalate),
        "check_weather": Tool("check_weather", "查询天气", check_weather),
    }


# ============================================================
# Skill 层 —— 流程逻辑 + 决策规则
# ============================================================

@dataclass
class Skill:
    """
    Level 1（常驻上下文）：名称 + 一行描述
    Level 2（按需加载）：完整指令
    """
    name: str
    description: str
    instructions: str          # Level 2: 类似 SKILL.md 的内容
    required_tools: list[str]  # 这个 Skill 依赖的工具列表

    def __repr__(self):
        return f"  [{self.name}] {self.description}"


def build_skill_library() -> dict[str, Skill]:
    """注册技能库"""

    return {
        "handle-refund": Skill(
            name="handle-refund",
            description="处理退款请求：验证用户->查订单->按条件退款或升级人工",
            required_tools=["get_customer", "get_order", "refund", "escalate"],
            instructions="""
[退款处理流程]

步骤1: 调用 get_customer 验证用户身份
步骤2: 调用 get_order 查询订单详情
步骤3: 条件判断
  - 购买30天内 + 金额<500 -> 调用 refund 执行退款
  - 金额>=500 -> 调用 escalate 升级人工处理（不退款）
  - 不符合条件 -> 调用 escalate 并说明原因

[约束]
  - 禁止在步骤1、2完成前执行退款
  - 退款金额不能超过原支付金额
  - 高级会员优先处理
"""
        ),
        "code-review": Skill(
            name="code-review",
            description="代码审查：检查代码风格、安全漏洞、性能问题",
            required_tools=[],
            instructions="""
[代码审查流程]

步骤1: 检查代码风格和命名规范
步骤2: 检查潜在的安全漏洞（SQL注入、XSS等）
步骤3: 检查性能问题（循环嵌套、不必要的计算等）
步骤4: 汇总审查意见

输出格式:
  - [PASS] 通过
  - [WARN] 建议改进
  - [FAIL] 必须修复
"""
        ),
        "customer-greeting": Skill(
            name="customer-greeting",
            description="客户问候：根据用户身份和时段生成个性化问候",
            required_tools=["get_customer"],
            instructions="""
[客户问候流程]

步骤1: 调用 get_customer 获取用户信息
步骤2: 根据会员等级生成不同语气问候
  - 高级会员 -> 尊称+感谢+主动服务
  - 普通会员 -> 礼貌问候+询问需求
"""
        ),
    }


# ============================================================
# Skill 引擎 —— 匹配 Skill + 执行工作流
# ============================================================

class SkillEngine:
    """
    Agent 的大脑：匹配任务 -> 展开 Skill 指令 -> 编排工具调用
    """

    def __init__(self, tools: dict[str, Tool], skills: dict[str, Skill]):
        self.tools = tools
        self.skills = skills

    # -- Level 1: 展示所有可用技能（模拟"常驻上下文"） --
    def list_skills(self) -> list[Skill]:
        """所有技能的名称+描述，模拟 Level 1 加载"""
        print("\n[Level 1] 当前可用技能（名称+描述，~100 tokens/个）:")
        for s in self.skills.values():
            print(f"   {s}")
        return list(self.skills.values())

    # -- Level 2: 匹配任务到技能 --
    def match_skill(self, task: str) -> Skill | None:
        """根据任务描述匹配最合适的技能"""
        task_lower = task.lower()

        keyword_map = [
            (("退款", "退货", "退钱"), "handle-refund"),
            (("审查", "review", "检查代码"), "code-review"),
            (("问候", "打招呼", "欢迎"), "customer-greeting"),
        ]

        for keywords, skill_name in keyword_map:
            if any(kw in task_lower for kw in keywords):
                skill = self.skills.get(skill_name)
                if skill:
                    print(f"\n[Level 2] 命中技能: {skill.name}")
                    print(f"   展开完整指令 ({len(skill.instructions)} chars):")
                    print(f"   {skill.instructions.strip()}")
                    return skill

        print(f"\n[失败] 未找到匹配 {task!r} 的技能")
        return None

    # -- Level 3: 执行 Skill 工作流 --
    def execute(self, skill: Skill, context: dict) -> str:
        """
        按 Skill 指令编排工具调用。
        真正的执行逻辑由 LLM 驱动，这里用预定义流程模拟。
        """
        print(f"\n[Level 3] 执行技能: {skill.name}")

        if skill.name == "handle-refund":
            return self._execute_refund(context)
        elif skill.name == "customer-greeting":
            return self._execute_greeting(context)
        else:
            return f"执行 {skill.name}：技能已激活，等待执行..."

    def _execute_refund(self, ctx: dict) -> str:
        """退款流程——演示 Skill 如何控制工具调用顺序和条件判断"""
        user_id = ctx.get("user_id", "u001")
        order_id = ctx.get("order_id", "ord001")

        print("\n 按 SKILL.md 流程执行退款:")

        # 步骤1: 验证用户（按 Skill 指令，必须第一步）
        print("\n  [步骤1/3] 验证用户...")
        user_info = self.tools["get_customer"](user_id=user_id)
        is_vip = "高级会员" in user_info

        # 步骤2: 查订单（按 Skill 指令，必须第二步）
        print("\n  [步骤2/3] 查询订单...")
        order_info = self.tools["get_order"](order_id=order_id)

        # 步骤3: 条件判断（Skill 的决策规则）
        print("\n  [步骤3/3] 执行退款决策...")
        import re
        match = re.search(r'CNY(\d+)', order_info)
        amount = int(match.group(1)) if match else 0

        if is_vip:
            print("  [VIP] 高级会员 -> 优先处理")

        if amount >= 500:
            result = self.tools["escalate"](reason=f"订单{order_id}金额CNY{amount}>=500，需人工审核")
        elif amount > 0:
            result = self.tools["refund"](order_id=order_id, amount=str(amount))
        else:
            result = self.tools["escalate"](reason=f"订单{order_id}信息异常")

        return f"\n[最终结果] {result}"

    def _execute_greeting(self, ctx: dict) -> str:
        """客户问候流程"""
        user_id = ctx.get("user_id", "u001")
        print("\n 按 SKILL.md 流程执行问候:")

        print("\n  [步骤1/2] 获取用户信息...")
        user_info = self.tools["get_customer"](user_id=user_id)

        print("\n  [步骤2/2] 生成问候语...")
        if "高级会员" in user_info:
            greeting = f"[尊贵]{user_info}，您好！感谢您一直以来的支持，请问有什么可以帮您的？"
        else:
            greeting = f"{user_info}，您好！欢迎咨询，请告诉我您的需求。"

        return f"\n[最终结果] {greeting}"


# ============================================================
# Agent —— 入口层
# ============================================================

class Agent:
    """Agent 入口：接收用户请求，匹配 Skill，执行"""

    def __init__(self):
        self.tools = build_tool_registry()
        self.skills = build_skill_library()
        self.engine = SkillEngine(self.tools, self.skills)

    def handle(self, task: str, context: dict | None = None):
        """处理用户请求的完整流程"""
        print(f"\n{'='*60}")
        print(f"[Agent] 用户请求: {task}")
        print(f"{'='*60}")

        # Level 1: 扫描技能
        self.engine.list_skills()

        # Level 2: 匹配技能 + 展开指令
        skill = self.engine.match_skill(task)

        if not skill:
            return f"抱歉，没有找到处理「{task}」的技能"

        # Level 3: 按 Skill 指令执行工作流
        result = self.engine.execute(skill, context or {})
        print(f"\n{result}")
        return result


# ============================================================
# 运行 Demo
# ============================================================

if __name__ == "__main__":
    agent = Agent()

    # ---- 场景 1: 普通退款 ----
    agent.handle(
        task="我要退款",
        context={"user_id": "u001", "order_id": "ord001"}
    )

    # ---- 场景 2: 高金额退款（触发升级人工） ----
    agent.handle(
        task="退货退款",
        context={"user_id": "u002", "order_id": "ord002"}
    )

    # ---- 场景 3: 客户问候 ----
    agent.handle(
        task="你好",
        context={"user_id": "u003"}
    )

    # ---- 场景 4: 不存在的技能 ----
    agent.handle(task="帮我写一首诗")

    print(f"\n{'='*60}")
    print("Demo 完成！展示了 Skill 三层架构:")
    print("  Level 1: 技能名称+描述（常驻上下文，~100 tokens/个）")
    print("  Level 2: 匹配到任务后展开完整 SKILL.md 指令")
    print("  Level 3: 按指令编排工具调用（顺序/条件/约束）")
    print(f"{'='*60}")
