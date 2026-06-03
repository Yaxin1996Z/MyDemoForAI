# MyDemoForAI 🤖

AI 学习项目 —— 从裸 API 到框架的完整学习路径。

## 目录结构

```
MyDemoForAI/
├── function_calling_demo/          # 裸 API Function Calling（无框架依赖）
│   └── fc_demo.py                 # 手写工具调用循环：JSON Schema → tool_calls → 执行 → 回送
├── langchain_demo/
│   ├── chain_demo.py              # Chain Demo：Prompt → LLM → Parser 流水线
│   ├── agent_demo.py              # Agent Demo：ReAct 循环 + Tool 调用
│   └── langgraph_demo.py          # LangGraph Demo：手写 StateGraph
├── .venv/                         # Python 虚拟环境
├── pyproject.toml                 # 项目依赖配置
└── README.md
```

## 快速开始

### 1. 激活虚拟环境

```bash
.venv\Scripts\activate
```

### 2. 运行 Function Calling Demo（裸 API）

```bash
python function_calling_demo\fc_demo.py
```

不用任何框架，直接用 API 实现完整的工具调用循环。演示 5 种场景：
- 单工具调用（计算器/天气）
- 无工具调用（纯对话）
- 条件判断 + 工具调用（查天气→判断是否发邮件）
- 并行工具调用（一轮调多个工具）

### 3. 运行 Chain Demo

```bash
python langchain_demo\chain_demo.py
```

演示最基本的 `PromptTemplate → LLM → OutputParser` 翻译流水线。

### 4. 运行 Agent Demo

```bash
python langchain_demo\agent_demo.py
```

演示 ReAct 循环：Agent 自动判断是否调用 `calculator` 和 `get_current_time` 工具。

### 5. 运行 LangGraph Demo

```bash
python langchain_demo\langgraph_demo.py
```

手写 StateGraph，理解 `create_agent` 内部如何工作：Node（LLM推理/工具执行）+ 条件路由（是否继续循环）+ 循环边（tools → llm）。

## 学习路径

```
裸 API Function Calling  ← 你现在在这里，理解底层协议
    ↓
LangChain @tool + Agent  ← 框架封装，提高效率
    ↓
LangGraph StateGraph     ← 白盒控制，灵活定制
```

## 环境要求

- Python >= 3.14
- DEEPSEEK_API_KEY 环境变量（已配置）

## 技术栈

| 技术 | 版本 |
|------|------|
| Python | 3.14.5 |
| LangChain | 1.3.2 |
| LangGraph | 1.2.2 |
| 模型 | DeepSeek Chat / deepseek-v4-flash |
