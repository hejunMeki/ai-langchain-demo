# LangChain Agent Demo

基于 [LangChain](https://docs.langchain.com/oss/python/langchain/overview) 框架的学习示例，演示 **模型调用** + **工具调用 (Tool)** + **对话记忆 (Memory)** 的完整流程。

使用 DeepSeek 作为 LLM 后端（兼容 OpenAI 标准接口）。

## 项目结构

```
ai-langchain-demo/
├── .env                # 环境变量配置 (API Key、模型地址)
├── .env.example        # 环境变量模板
├── requirements.txt    # Python 依赖
├── demo_basic.py       # Agent Demo (模型 + 工具 + 记忆)
└── README.md           # 本文档
```

## 环境要求

- **Python >= 3.10**
- 需要 DeepSeek API Key（或其他兼容 OpenAI 接口的 API Key）

> **注意**: 你的系统中如果有多个 Python 版本，请确认 `python` 命令指向的是安装了依赖的那个版本。
> 可以使用完整路径运行，例如：
> ```powershell
> C:\Users\Hualin\AppData\Local\Programs\Python\Python312\python.exe demo_basic.py
> ```

## 快速开始

### 1. 配置环境变量

复制模板文件并填入你的 API Key：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
# DeepSeek 配置
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_API_BASE=https://api.deepseek.com
MODEL_NAME=deepseek-chat
```

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `OPENAI_API_KEY` | API Key（必填） | 无 |
| `OPENAI_API_BASE` | API 基础地址 | `https://api.openai.com/v1` |
| `MODEL_NAME` | 模型名称 | `deepseek-chat` |

其他兼容 OpenAI 接口的模型服务也可使用，只需修改对应的 `OPENAI_API_BASE` 和 `MODEL_NAME`。

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

安装的核心包：

| 包名 | 用途 |
|------|------|
| `langchain` | LangChain 核心框架，提供 Agent、Tool 等抽象 |
| `langchain-openai` | OpenAI 兼容模型接入（支持 DeepSeek 等） |
| `langgraph` | Agent 运行时引擎，提供记忆、检查点等功能 |
| `python-dotenv` | 从 `.env` 文件加载环境变量 |

### 3. 运行 Demo

```bash
python demo_basic.py
```

启动后进入交互式对话，输入 `quit` 或 `exit` 退出。

### 示例对话

```
============================================================
  LangChain 基础 Agent Demo (DeepSeek)
  输入 'quit' 或 'exit' 退出
============================================================

你: 北京今天天气怎么样？

AI: 北京今天晴天，气温 25°C，湿度 40%。

你: 帮我算一下 (15 + 27) * 3

AI: 计算结果为：(15 + 27) × 3 = 126

你: 什么是 LangChain？

AI: LangChain 是一个用于构建 LLM 应用的开源框架，支持 Agent、Tool、Memory 等核心概念。

你: 刚才的计算结果是多少？

AI: 刚才的计算结果是 (15 + 27) × 3 = 126
```

## Demo 功能说明

### 内置工具

| 工具 | 功能 | 示例输入 |
|------|------|----------|
| `get_weather` | 查询城市天气（模拟数据） | "北京天气怎么样" |
| `calculate` | 计算数学表达式 | "算一下 (100 + 200) * 3" |
| `search_knowledge` | 搜索知识库（模拟数据） | "什么是 LangChain" |

### 对话记忆

Demo 使用 `MemorySaver` 实现内存中的对话记忆，同一会话内 Agent 能记住之前的对话上下文。

## 核心架构

```
用户输入
  │
  ▼
┌─────────────────────────────────────┐
│           LangChain Agent           │
│  ┌───────────┐   ┌───────────────┐  │
│  │  DeepSeek │   │  System       │  │
│  │  LLM      │   │  Prompt       │  │
│  └─────┬─────┘   └───────────────┘  │
│        │                             │
│        ▼                             │
│  模型判断是否需要调用工具            │
│        │                             │
│   ┌────┴────┐                        │
│   │ 是      │ 否                     │
│   ▼         ▼                        │
│ 调用工具  直接回复                   │
│   │                                  │
│   ▼                                  │
│ 工具返回结果 → 模型组织回复          │
└─────────────────────────────────────┘
  │
  ▼
AI 回复
```

### 关键代码解读

**1. 模型初始化** — 使用 `ChatOpenAI` 连接 DeepSeek（OpenAI 兼容接口）：

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="deepseek-chat",
    api_key="sk-xxx",
    base_url="https://api.deepseek.com",
)
```

**2. 定义工具** — 用 `@tool` 装饰器，函数的 docstring 会作为工具描述供模型理解：

```python
from langchain_core.tools import tool

@tool
def calculate(expression: str) -> str:
    """计算数学表达式，例如: '2 + 3 * 4'"""
    return str(eval(expression))
```

**3. 创建 Agent** — 将模型、工具、提示词组合为智能体：

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

agent = create_agent(
    model=model,
    tools=[calculate, get_weather],
    system_prompt="你是一个智能助手",
    checkpointer=MemorySaver(),
)
```

**4. 调用 Agent** — 传入消息并获取回复：

```python
response = agent.invoke(
    {"messages": [{"role": "user", "content": "帮我算 1+1"}]},
    config={"configurable": {"thread_id": "session-1"}},
)
print(response["messages"][-1].content)
```

## 参考链接

- [LangChain 官方文档](https://docs.langchain.com/oss/python/langchain/overview)
- [LangChain 安装指南](https://docs.langchain.com/oss/python/langchain/install)
- [LangChain 快速入门](https://docs.langchain.com/oss/python/langchain/quickstart)
- [LangChain Tools 文档](https://docs.langchain.com/oss/python/langchain/tools)
- [DeepSeek API 文档](https://platform.deepseek.com/api-docs)
