# MyDemoForAI 🤖

LangChain 学习项目 —— Chain 和 Agent 的实战 Demo。

## 目录结构

```
MyDemoForAI/
├── langchain_demo/
│   ├── chain_demo.py      # Chain Demo：Prompt → LLM → Parser 流水线
│   ├── agent_demo.py      # Agent Demo：ReAct 循环 + Tool 调用
│   └── langgraph_demo.py  # LangGraph Demo：手写 StateGraph，理解 Agent 内部机制
├── .venv/               # Python 虚拟环境
├── pyproject.toml       # 项目依赖配置
└── README.md
```

## 快速开始

### 1. 激活虚拟环境

```bash
.venv\Scripts\activate
```

### 2. 运行 Chain Demo

```bash
python langchain_demo\chain_demo.py
```

演示最基本的 `PromptTemplate → LLM → OutputParser` 翻译流水线。

### 3. 运行 Agent Demo

```bash
python langchain_demo\agent_demo.py
```

演示 ReAct 循环：Agent 自动判断是否调用 `calculator` 和 `get_current_time` 工具。

### 4. 运行 LangGraph Demo

```bash
python langchain_demo\langgraph_demo.py
```

手写 StateGraph，理解 `create_agent` 内部如何工作：Node（LLM推理/工具执行）+ 条件路由（是否继续循环）+ 循环边（tools → llm）。

## 环境要求

- Python >= 3.14
- DEEPSEEK_API_KEY 环境变量（已配置）

## 技术栈

| 技术 | 版本 |
|------|------|
| Python | 3.14.5 |
| LangChain | 1.3.2 |
| LangGraph | 1.2.2 |
| 模型 | DeepSeek Chat |
