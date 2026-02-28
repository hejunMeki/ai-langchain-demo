"""
Demo: LangChain 基础 Agent - 模型调用 + 本地工具

演示内容:
  1. 使用 ChatOpenAI 初始化模型 (兼容 DeepSeek 等 OpenAI 标准接口)
  2. 用 @tool 装饰器定义本地工具
  3. 用 create_agent 创建智能体
  4. 对话式调用 (带记忆)

运行:
  pip install -r requirements.txt
  python demo_basic.py
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv(dotenv_path=".env")


# ── 定义工具 ──────────────────────────────────────────────


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气情况"""
    weather_data = {
        "北京": "晴天, 25°C, 湿度 40%",
        "上海": "多云, 28°C, 湿度 65%",
        "深圳": "雷阵雨, 30°C, 湿度 80%",
        "杭州": "阴天, 22°C, 湿度 55%",
    }
    return weather_data.get(city, f"{city}: 暂无该城市的天气数据，请确认城市名称。")


@tool
def calculate(expression: str) -> str:
    """计算数学表达式，例如: '2 + 3 * 4'"""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "错误: 表达式包含不允许的字符"
    try:
        result = eval(expression)  # noqa: S307
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


@tool
def search_knowledge(query: str) -> str:
    """搜索知识库，可查询 langchain、python、mcp 等技术关键词"""
    knowledge = {
        "langchain": "LangChain 是一个用于构建 LLM 应用的开源框架，支持 Agent、Tool、Memory 等核心概念。",
        "python": "Python 是一种广泛使用的高级编程语言，以其简洁易读的语法著称。",
        "mcp": "MCP (Model Context Protocol) 是一个开放协议，标准化了应用程序向 LLM 提供工具和上下文的方式。",
    }
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return value
    return f"未找到关于 '{query}' 的知识条目。"


# ── 配置模型 ──────────────────────────────────────────────


def create_model():
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    model_name = os.getenv("MODEL_NAME", "deepseek-chat")

    if not api_key:
        raise ValueError("请在 .env 文件中设置 OPENAI_API_KEY")

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=api_base,
        temperature=0.7,
        max_tokens=2000,
    )


# ── 主程序 ────────────────────────────────────────────────

SYSTEM_PROMPT = """你是一个智能助手，可以帮助用户查天气、做数学计算、查询知识库。
请用中文回复，回答简洁明了。如果需要使用工具，请先使用工具获取信息再回复用户。"""


def main():
    print("=" * 60)
    print("  LangChain 基础 Agent Demo (DeepSeek)")
    print("  输入 'quit' 或 'exit' 退出")
    print("=" * 60)

    model = create_model()
    checkpointer = MemorySaver()

    agent = create_agent(
        model=model,
        tools=[get_weather, calculate, search_knowledge],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )

    config = {"configurable": {"thread_id": "demo-session-1"}}

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("再见!")
            break

        try:
            response = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
            )
            ai_message = response["messages"][-1]
            print(f"\nAI: {ai_message.content}")
        except Exception as e:
            print(f"\n错误: {e}")


if __name__ == "__main__":
    main()
