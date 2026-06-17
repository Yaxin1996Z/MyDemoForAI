# MyDemoForAI 🤖

AI 学习项目 —— 从裸 API 到框架的完整学习路径。

## 目录结构

```
MyDemoForAI/
├── function_calling_demo/          # 裸 API Function Calling（无框架依赖）
│   ├── fc_demo.py                 # 手写工具调用循环：JSON Schema → tool_calls → 执行 → 回送
│   └── tool_router.py             # 工具路由分发
├── langchain_demo/
│   ├── chain_demo.py              # Chain Demo：Prompt → LLM → Parser 流水线
│   ├── agent_demo.py              # Agent Demo：ReAct 循环 + Tool 调用
│   └── langgraph_demo.py          # LangGraph Demo：手写 StateGraph
├── rag_demo/                      # RAG Demo：个人知识库问答
│   ├── rag_demo.py               # RAG 全流程：读取笔记 → 分块 → Chroma → 检索 → LLM 生成
│   └── rag_db/                    # Chroma 持久化向量库（自动生成）
├── memory_demo/                   # Agent Memory 系统
│   └── memory_demo.py            # CircularBuffer / EntityMemory / SummaryMemory / HybridMemory
├── mcp_demo/                      # MCP 协议实战
│   ├── mcp_server.py             # FastMCP Server（Tools + Resources + Prompts）
│   └── mcp_client.py             # MCP Client（自动发现 + 调用）
├── skill_demo/                    # Agent Skill 系统
│   └── skill_demo.py            # Skill 三层架构：Level 1/2/3 渐进式加载
├── crewai_demo/                   # CrewAI 多Agent编排
│   ├── handcraft_crew.py         # 手写多Agent协作
│   └── real_crew.py              # CrewAI 框架版
├── docker-demo/                   # Docker 容器化练习
│   ├── Dockerfile
│   └── app.py
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

### 5. 运行 RAG Demo

基于个人学习笔记的知识库问答。读取技能面板下的 Markdown 笔记 → 分块 → Chroma 向量库 → 检索 → LLM 生成带引用的回答。

**首次使用前**：复制 `rag_demo/config.example.py` 为 `rag_demo/config.py`，填入本地路径：

```bash
cp rag_demo/config.example.py rag_demo/config.py
# 然后编辑 config.py 中的 NOTES_DIR / DB_DIR / EMBED_MODEL_PATH
```

```bash
python rag_demo\rag_demo.py              # 运行（自动建库）
python rag_demo\rag_demo.py --rebuild    # 强制重建索引
```

### 6. 运行 LangGraph Demo

```bash
python langchain_demo\langgraph_demo.py
```

手写 StateGraph，理解 `create_agent` 内部如何工作：Node（LLM推理/工具执行）+ 条件路由（是否继续循环）+ 循环边（tools → llm）。

### 7. 运行 MCP Demo

```bash
python mcp_demo\mcp_server.py      # 启动 MCP Server（终端1）
python mcp_demo\mcp_client.py      # 启动 MCP Client（终端2）
```

演示 MCP 三大核心概念：Tools（可调用操作）、Resources（可读取数据）、Prompts（预置模板）。

### 8. 运行 Skill Demo

```bash
python skill_demo\skill_demo.py
```

演示 Agent Skill 三层架构：Level 1（技能元信息）→ Level 2（匹配展开指令）→ Level 3（编排工具调用）。

### 9. 运行 Memory Demo

```bash
python memory_demo\memory_demo.py
```

四种 Memory 实现：CircularBuffer / EntityMemory / SummaryMemory / HybridMemory。

### 10. 运行 CrewAI Demo

```bash
python crewai_demo\real_crew.py
```

多角色 Agent 协作：Planner / Researcher / Writer / Reviewer 四角色编排。

## 学习路径

```
裸 API Function Calling    ← 理解底层协议
    ↓
LangChain @tool + Agent    ← 框架封装
    ↓
LangGraph StateGraph       ← 白盒控制
    ↓
MCP 协议                   ← 工具标准化连接
    ↓
Agent Memory / Skill       ← 记忆与技能系统
    ↓
CrewAI 多Agent编排         ← 多角色协作
```

## 环境要求

- Python >= 3.12
- DEEPSEEK_API_KEY 环境变量（已配置）

## 技术栈

| 技术 | 版本 |
|------|------|
| Python | 3.14.5 |
| TypeScript / Node.js | 5.x / v24.15 |
| LangChain / LangGraph | 1.3.2 / 1.2.2 |
| FastMCP | 3.3.1 |
| 模型 | DeepSeek Chat / qwen3.5-plus / GLM-5 |
