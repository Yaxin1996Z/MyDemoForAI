"""
自定义 Memory 系统 —— 从零实现，不依赖 LangChain Memory 组件
========================================================

三层记忆架构：
  1. CircularBufferMemory —— 短期记忆（环形缓冲区，自动丢弃最早的消息）
  2. EntityMemory         —— 实体记忆（从对话中提取关键信息）
  3. SummaryMemory        —— 摘要记忆（长对话用 LLM 压缩）

核心思想：
  Memory 的本质不是组件，而是数据流模式——
  要么全量塞，要么检索后塞，最终都变成 Prompt 的一部分。
"""

import re
from typing import List, Optional, Callable, Dict
from dataclasses import dataclass, field
from datetime import datetime


# ============================================================
# 消息模型
# ============================================================

@dataclass
class Message:
    """单条消息"""
    role: str       # "user" | "assistant" | "system" | "tool"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def format(self) -> str:
        return f"{self.role}: {self.content}"


# ============================================================
# Memory 基类（统一接口）
# ============================================================

class BaseMemory:
    """所有 Memory 的抽象基类，定义统一接口"""

    def add_user_message(self, content: str):
        raise NotImplementedError

    def add_ai_message(self, content: str):
        raise NotImplementedError

    def get_context(self) -> str:
        """返回拼装好的上下文文本，直接塞进 Prompt"""
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError


# ============================================================
# 1. CircularBufferMemory —— 短期记忆（环形缓冲区）
# ============================================================

class CircularBufferMemory(BaseMemory):
    """
    环形缓冲区记忆。
    只保留最近 n 轮对话（n 轮 = 2n 条消息），超出的自动丢弃最早的消息。

    ┌──────────────────────────────────────────┐
    │  CircularBuffer (k=3)                    │
    │  ┌─────┬─────┬─────┬─────┬─────┬─────┐   │
    │  │ m1  │ m2  │ m3  │ m4  │ m5  │ m6  │   │
    │  └─────┴─────┴─────┴─────┴─────┴─────┘   │
    │   ↑丢弃             ↑保留                 │
    │  最早的消息        最新 k 轮               │
    └──────────────────────────────────────────┘
    """

    def __init__(self, k: int = 10):
        assert k >= 1, "k 必须 >= 1"
        self.k = k
        self._messages: List[Message] = []

    def add_user_message(self, content: str):
        self._messages.append(Message("user", content))
        self._trim()

    def add_ai_message(self, content: str):
        self._messages.append(Message("assistant", content))
        self._trim()

    def _trim(self):
        """超出上限时丢弃最早的消息"""
        max_msgs = self.k * 2  # k 轮 = 2k 条消息（user + assistant 各一条）
        while len(self._messages) > max_msgs:
            self._messages.pop(0)

    def get_history(self) -> List[Message]:
        """获取所有消息（通常用于调试/统计）"""
        return self._messages.copy()

    def get_context(self) -> str:
        """格式化为 Prompt 上下文"""
        if not self._messages:
            return ""
        return "\n".join(m.format() for m in self._messages)

    def clear(self):
        self._messages.clear()

    # ---- 统计属性 ----
    @property
    def message_count(self) -> int:
        return len(self._messages)

    @property
    def round_count(self) -> int:
        """对话轮数（向上取整）"""
        return (len(self._messages) + 1) // 2

    @property
    def token_estimate(self) -> int:
        """粗略估计 token 数：中文 ≈ 2 token，英文/数字 ≈ 1 token"""
        total = 0
        for m in self._messages:
            for ch in m.content:
                total += 2 if ord(ch) > 127 else 1
        return total


# ============================================================
# 2. EntityMemory —— 实体记忆
# ============================================================

class EntityMemory(BaseMemory):
    """
    实体记忆。
    从对话中提取关键实体（人名、地名、偏好、技能等），结构化存储。

    每次调用 extract_from() → 扫描文本 → 合并到实体字典

    两种提取策略：
      ① 正则提取（内置基础模式）
      ② LLM 提取（更精准，需传入 llm_extract 函数）

    使用方式：
        memory = EntityMemory()
        memory.extract_from("我叫张三，是AI工程师")
        memory.get_context()
        # → "[用户信息] name=张三, job=AI工程师"
    """

    # 内置正则模式
    PATTERNS: Dict[str, List[str]] = {
        "name": [
            r"我叫(.{1,8}?)(?:[，。、,.!！]|$)",
            r"我的名字是(.{1,8}?)(?:[，。、,.!！]|$)",
        ],
        "city": [
            r"我住在(.{1,10}?)(?:[，。、,.!！]|$)",
            r"来自(.{1,10}?)(?:[，。、,.!！]|$)",
            r"家在(.{1,10}?)(?:[，。、,.!！]|$)",
        ],
        "job": [
            r"([^，。、,.!！]{1,15}(?:工程师|员|师))",
            r"我从事(.{1,15}?)(?:工作|$)",
        ],
        "hobby": [
            r"我喜欢(.{1,20}?)(?:[，。、,.!！]|$)",
            r"我的爱好是(.{1,20}?)(?:[，。、,.!！]|$)",
        ],
    }

    def __init__(self, llm_extract: Optional[Callable] = None):
        self._entities: Dict[str, str] = {}
        self._llm_extract = llm_extract

    def add_user_message(self, content: str):
        """添加用户消息时自动提取实体"""
        self.extract_from(content)

    def add_ai_message(self, content: str):
        """AI 消息也扫描（有些信息可能在 AI 回复中确认）"""
        self.extract_from(content)

    def extract_from(self, text: str):
        """从文本中提取实体"""
        # 方法一：正则提取
        for key, patterns in self.PATTERNS.items():
            if key in self._entities:
                continue  # 已提取的不重复提取
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    self._entities[key] = match.group(1).strip()
                    break

        # 方法二：LLM 提取（补充正则没覆盖到的）
        if self._llm_extract and not self._entities.get("name"):
            # 只在正则没提取到名字时才用 LLM 补充
            extra = self._llm_extract(text)
            if extra:
                self._entities.update(extra)

    def get_context(self) -> str:
        """返回实体上下文片段（空则返回空字符串）"""
        if not self._entities:
            return ""
        parts = [f"{k}={v}" for k, v in self._entities.items()]
        return f"[用户信息] {'; '.join(parts)}"

    def get_all(self) -> dict:
        return self._entities.copy()

    def clear(self):
        self._entities.clear()


# ============================================================
# 3. SummaryMemory —— 摘要记忆
# ============================================================

class SummaryMemory(BaseMemory):
    """
    摘要记忆。
    对话达到阈值轮数后，调用 LLM 生成摘要压缩历史。

    策略：
        context = summary（历史摘要） + recent_k 轮完整对话

    这样既保留了关键历史信息，又不会让 Prompt 无限膨胀。

    ┌────────────────────────────────────────┐
    │  [摘要] 用户询问了产品价格...            │  ← 压缩的历史
    │  ──────────────────────────────────    │
    │  [最近 2 轮]                           │  ← 完整保留
    │  user: 那支持多轮对话吗？               │
    │  assistant: 支持...                    │
    └────────────────────────────────────────┘
    """

    def __init__(self, llm: Callable[[str], str], k: int = 4, threshold: int = 6):
        """
        Args:
            llm: LLM 调用函数，接收 prompt 返回文本
            k: 保留最近多少轮完整对话
            threshold: 总轮数超过此值触发摘要
        """
        self.llm = llm
        self.k = k
        self.threshold = threshold
        self._summary: Optional[str] = None
        self._buffer = CircularBufferMemory(k=threshold * 2)

    def add_user_message(self, content: str):
        self._buffer.add_user_message(content)

    def add_ai_message(self, content: str):
        self._buffer.add_ai_message(content)

    def should_summarize(self) -> bool:
        """判断是否应该触发摘要"""
        total = self._buffer.round_count
        # 如果已有摘要，新对话又累积了 threshold 轮，再次压缩
        if self._summary:
            return total >= self.threshold * 2
        return total >= self.threshold

    def summarize(self) -> str:
        """调用 LLM 生成摘要"""
        # 构造需要被摘要的文本
        if self._summary:
            # 已有旧摘要，摘要 = 旧摘要 + 上次摘要后的新对话
            recent = "\n".join(m.format() for m in self._buffer.get_history()[-self.threshold * 2:])
            to_summarize = f"已有摘要：{self._summary}\n\n新增对话：\n{recent}"
        else:
            to_summarize = self._buffer.get_context()

        prompt = f"""请将以下对话压缩为一段简洁的摘要（50-150字），保留关键信息：
- 用户身份和偏好
- 已经解决的问题和结论
- 重要的约定和承诺

对话：
{to_summarize}

简洁摘要："""

        self._summary = self.llm(prompt)
        return self._summary

    def get_context(self) -> str:
        """返回：摘要（如有） + 最近 k 轮完整对话"""
        parts = []

        if self._summary:
            parts.append(f"[对话摘要] {self._summary}")

        recent = self._buffer.get_history()[-self.k * 2:]
        if recent:
            if self._summary:
                parts.append("[最近对话]")
            parts.extend(m.format() for m in recent)

        return "\n".join(parts)

    def clear(self):
        self._summary = None
        self._buffer.clear()

    @property
    def round_count(self) -> int:
        return self._buffer.round_count


# ============================================================
# 4. HybridMemory —— 三层融合
# ============================================================

class HybridMemory(BaseMemory):
    """
    混合记忆 —— 组合三种记忆类型。

    get_context() 输出结构：
        [用户信息] name=张三; job=AI工程师     ← 实体记忆
        [对话摘要] 用户问了产品功能和价格...      ← 摘要记忆
        [最近对话]
        user: 最后一个问题
        assistant: 最后一个回答                  ← 短期记忆

    优点：
        实体记忆保证关键信息不丢失
        摘要记忆控制上下文长度
        短期记忆保留最近对话的完整性
    """

    def __init__(
        self,
        llm: Optional[Callable[[str], str]] = None,
        buffer_k: int = 10,
        summary_k: int = 4,
        summary_threshold: int = 6,
    ):
        self.buffer = CircularBufferMemory(k=buffer_k)
        self.entity = EntityMemory(
            llm_extract=(lambda text: self._llm_extract_entity(text)) if llm else None
        )
        self.summary = SummaryMemory(
            llm=llm, k=summary_k, threshold=summary_threshold
        ) if llm else None
        self._llm = llm

    def add_user_message(self, content: str):
        self.buffer.add_user_message(content)
        self.entity.add_user_message(content)
        if self.summary:
            self.summary.add_user_message(content)

    def add_ai_message(self, content: str):
        self.buffer.add_ai_message(content)
        self.entity.add_ai_message(content)
        if self.summary:
            self.summary.add_ai_message(content)

    def get_context(self) -> str:
        """组装三层记忆为完整的 Prompt 上下文"""
        parts = []

        # 第 1 层：实体记忆（始终在最前面）
        entity_ctx = self.entity.get_context()
        if entity_ctx:
            parts.append(entity_ctx)

        # 第 2 层：摘要记忆
        if self.summary and self.summary._summary:
            parts.append(self.summary.get_context())
        else:
            # 第 3 层：没有摘要时直接用短期记忆
            history = self.buffer.get_context()
            if history:
                parts.append("[对话历史]")
                parts.append(history)

        return "\n".join(parts)

    def try_summarize(self):
        """如果达到阈值，触发摘要（通常由外部在合适时机调用）"""
        if self.summary and self.summary.should_summarize():
            self.summary.summarize()

    def _llm_extract_entity(self, text: str) -> dict:
        """用 LLM 从文本中提取实体"""
        prompt = f"""从以下文本中提取用户的关键信息（name, city, job, hobby 等），以 JSON 格式返回：
文本：{text}

JSON（只返回 JSON，不要其他内容）："""
        try:
            result = self._llm(prompt)
            # 尝试从结果中解析 JSON
            import json
            # 找到 { } 包裹的部分
            match = re.search(r'\{.*\}', result, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return {}

    def get_stats(self) -> dict:
        """获取当前 Memory 统计信息"""
        return {
            "buffer_messages": self.buffer.message_count,
            "buffer_rounds": self.buffer.round_count,
            "estimated_tokens": self.buffer.token_estimate,
            "entities": dict(self.entity.get_all()),
            "entity_count": len(self.entity.get_all()),
            "has_summary": bool(self.summary and self.summary._summary),
            "summary_text": self.summary._summary if self.summary else None,
        }

    def clear(self):
        self.buffer.clear()
        self.entity.clear()
        if self.summary:
            self.summary.clear()
