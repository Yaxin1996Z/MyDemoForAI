"""
Memory 实战演示 —— 从手写 Memory 到 Agent 集成
===============================================

演示流程：
  1. 为什么需要 Memory？（无 Memory vs 有 Memory 对比）
  2. CircularBufferMemory 演示
  3. EntityMemory 演示
  4. SummaryMemory 演示
  5. HybridMemory + Agent 集成演示

用法：
  python memory_demo/run_demo.py
  python memory_demo/run_demo.py --part 1   # 只跑第 1 部分
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import os
import argparse

# 确保能找到项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from memory_demo.custom_memory import (
    CircularBufferMemory,
    EntityMemory,
    SummaryMemory,
    HybridMemory,
)


# ============================================================
# LLM 调用工具
# ============================================================

def create_llm(model: str = "deepseek-chat"):
    """创建 LLM 调用函数"""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )

    def call_llm(prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"[API Error] {e}"

    return call_llm


llm = create_llm()


# ============================================================
# 辅助函数
# ============================================================

def separator(title: str):
    """打印分隔标题"""
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def step(msg: str):
    """打印步骤"""
    print(f"\n  ▶ {msg}")


# ============================================================
# Part 1: 为什么需要 Memory
# ============================================================

def part1_why_memory():
    print('\n\U0001f4cc 场景：用户先自我介绍，然后问「我叫什么名字？」')
    print('   → 如果没有 Memory，LLM 会「失忆」')
    print('   → 有了 Memory，LLM 能「记住」')
    print('-' * 50)

    # ── 无 Memory ──
    step('无 Memory：每次都是独立的 API 调用')
    resp1 = llm('我叫张三，是一名 AI 工程师', '你是一个助手')
    print('  第一轮：用户说「我叫张三，是AI工程师」')
    print(f'  AI 回复：{resp1[:60]}...')

    resp2 = llm('我之前说我叫什么名字？', '你是一个助手')
    print('  第二轮：用户问「我之前说我叫什么名字？」')
    print(f'  AI 回复：{resp2[:60]}...')

    if '张三' in resp2:
        print('  ✅ AI 碰巧说对了（但不可靠）')
    else:
        print('  ❌ AI 不记得了 —— 这就是 LLM 的「失忆」问题')

    # ── 有 Memory ──
    step('有 Memory：把历史拼到 Prompt 里')
    mem = CircularBufferMemory(k=10)
    mem.add_user_message('我叫张三，是一名 AI 工程师')
    r1 = llm(mem.get_context() + '\n请回复用户。', '你是一个助手')
    mem.add_ai_message(r1)

    mem.add_user_message('我之前说我叫什么名字？')
    r2 = llm(
        f'{mem.get_context()}\n\n用户说：「我之前说我叫什么名字？」',
        '你是一个助手，根据对话历史回答用户的问题。'
    )
    mem.add_ai_message(r2)
    print('  第一轮：用户说「我叫张三，是AI工程师」')
    print('  第二轮：用户问「我之前说我叫什么名字？」')
    print(f'  AI 回复：{r2[:80]}...')
    print('  ✅ Memory 把历史拼到 Prompt → LLM 就能记住了')

    step('核心原理')
    print('  Memory 将历史消息格式化为文本拼到 Prompt 里：')
    print('  ┌─────────────────────────────────────────────')
    print('  │ system: 你是一个助手...')
    print('  │ user: 我叫张三，是一名 AI 工程师')
    print('  │ assistant: 你好张三！...')
    print('  │ user: 我之前说我叫什么名字？ ← LLM 看到了上面的对话')
    print('  └─────────────────────────────────────────────')


# ============================================================
# Part 2: CircularBufferMemory 演示
# ============================================================

def part2_circular_buffer():
    buffer = CircularBufferMemory(k=3)
    print(f'  配置：k={buffer.k}（保留最近 {buffer.k} 轮 = {buffer.k*2} 条消息）\n')
    print('  模拟 5 轮对话，观察消息数变化：')
    print('  ───────────────────────────────')

    for i in range(1, 6):
        buffer.add_user_message(f'第{i}条用户消息')
        buffer.add_ai_message(f'第{i}条AI回复')
        print(f'  第{i}轮后 → 消息数: {buffer.message_count:2d}  |  估计token: {buffer.token_estimate:3d}')

    print(f'\n  最终保留的消息（最新的 {buffer.k} 轮 = {buffer.message_count} 条）：')
    for m in buffer.get_history():
        print(f'    [{m.role:>9}] {m.content}')

    print('\n  \U0001f4a1 最早的两条消息（第1轮）已被自动丢弃')


# ============================================================
# Part 3: EntityMemory 演示
# ============================================================

def part3_entity():
    print('  演示从对话中自动提取用户信息\n')
    print('  ├─ 内置正则模式：name / city / job / hobby')
    print('  └─ 支持 LLM 补充提取\n')

    mem = EntityMemory()

    test_cases = [
        '你好，我叫张三',
        '我住在北京，是一名AI工程师',
        '平时我喜欢打羽毛球',
    ]

    for text in test_cases:
        step(f'用户说：「{text}」')
        mem.extract_from(text)
        entities = mem.get_all()
        print(f'    已提取实体: {entities}')

    print(f'\n  get_context() 输出:')
    print(f'    {mem.get_context()}')

    print('\n  \U0001f4a1 实体记忆把关键信息提取成结构化键值对，不会丢失')


# ============================================================
# Part 4: SummaryMemory 演示
# ============================================================

def part4_summary():
    print('  配置：threshold=4, k=2（超过 4 轮触发摘要，保留最近 2 轮）\n')

    summary_llm = lambda p: llm(p, '你是一个摘要助手，简洁准确地总结对话。')
    mem = SummaryMemory(llm=summary_llm, k=2, threshold=4)

    dialog = [
        ('user', '你好，我想了解一下你们的 AI 产品'),
        ('assistant', '你好！我们主要提供大模型 API 服务，支持文本生成、代码补全等功能。'),
        ('user', '价格怎么算的？有没有免费额度？'),
        ('assistant', '我们按 token 计费，百万输入 token 约 0.5 元。新用户赠送 500 万免费额度。'),
        ('user', '那你们支持多轮对话吗？比如客服场景'),
        ('assistant', '支持。可以传入历史消息实现多轮对话。'),
        ('user', '好的，明白了。再帮我看看文档链接'),
        ('assistant', '文档地址是 docs.example.com，里面有完整的 API 参考。'),
    ]

    for role, content in dialog:
        if role == 'user':
            mem.add_user_message(content)
        else:
            mem.add_ai_message(content)

    print(f'  原始对话：{len(dialog)} 轮（{mem.round_count} 轮）')
    yes_flag = '是 ✅' if mem.should_summarize() else '否 ❌'
    print(f'  是否需要压缩？{yes_flag}')

    step('触发摘要生成')
    summary = mem.summarize()
    print(f'  生成的摘要：\n    {summary}')

    step('get_context() 输出')
    print(f'  {mem.get_context()}')
    print('\n  \U0001f4a1 摘要保留了关键信息，最近 2 轮完整保留，历史被压缩')


# ============================================================
# Part 5: HybridMemory + Agent 集成
# ============================================================

def part5_hybrid_agent():
    print('  组合三层记忆：CircularBuffer + Entity + Summary\n')
    print('  Agent 工作流程：')
    print('    user_input → 存入 Memory → 构建 Context → LLM 调用 → 回存 Memory')
    print('  ─────────────────────────────────────────────\n')

    mem = HybridMemory(llm=llm, buffer_k=10, summary_k=3, summary_threshold=4)

    conversation = [
        '你好！我叫张三',
        '我是一名 AI 工程师，平时做 RAG 相关的工作',
        '帮我写一个 Python 的快速排序',
    ]

    for msg in conversation:
        step(f'用户说：「{msg}」')

        # 1. 存到 Memory（自动提取实体、管理缓冲区）
        mem.add_user_message(msg)

        # 2. 检查是否需要压缩
        mem.try_summarize()

        # 3. 构造带 Memory 的 Prompt
        context = mem.get_context()
        system = (
            '你是 AI 助手。你的回复需要利用对话历史和已知的用户信息。\n'
            '如果用户问了具体的技术问题，请认真回答。'
        )

        full_prompt = f'{context}\n\n用户说：{msg}'
        response = llm(full_prompt, system)
        mem.add_ai_message(response)

        print(f'  AI 回复：{response[:120]}...')

    # 打印统计
    stats = mem.get_stats()
    print(f'\n  \U0001f4ca Memory 最终状态：')
    print(f'     ├─ 消息数: {stats["buffer_messages"]}')
    print(f'     ├─ 估计 token: {stats["estimated_tokens"]}')
    print(f'     ├─ 实体: {stats["entities"]}')
    print(f'     ├─ 实体数: {stats["entity_count"]}')
    print(f'     └─ 已摘要: {stats["has_summary"]}')

    print('\n  \U0001f4a1 HybridMemory 同时具备短期记忆的完整性和长期记忆的压缩能力')


# ============================================================
# Main
# ============================================================

PARTS = {
    1: ('为什么需要 Memory？', part1_why_memory),
    2: ('CircularBufferMemory 演示', part2_circular_buffer),
    3: ('EntityMemory 演示', part3_entity),
    4: ('SummaryMemory 演示', part4_summary),
    5: ('HybridMemory + Agent 集成', part5_hybrid_agent),
}


def main():
    parser = argparse.ArgumentParser(description='Memory 实战演示')
    parser.add_argument('--part', type=int, choices=range(1, 6),
                        help='只运行指定部分 (1-5)')
    args = parser.parse_args()

    print('╔═════════════════════════════════════════════╗')
    print('║      \U0001f9e0  自定义 Memory 系统 \xb7 实战演示          ║')
    print('║      从手写实现到 Agent 集成                    ║')
    print('╚═════════════════════════════════════════════╝')

    if args.part:
        title, fn = PARTS[args.part]
        separator(f'Part {args.part}: {title}')
        fn()
    else:
        for i in range(1, 6):
            title, fn = PARTS[i]
            separator(f'Part {i}: {title}')
            fn()

    print('\n' + '=' * 60)
    print('  ✅ 演示全部完成！')
    print('  \U0001f4c1 源码: memory_demo/custom_memory.py')
    print('  \U0001f527 试试修改参数后重新运行，观察效果变化')
    print('=' * 60)


if __name__ == '__main__':
    main()
