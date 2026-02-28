"""
Test: 股票分析 Agent - 通过 MCP Server 获取工具

MCP Server: https://htai-test.chinalin.com/mcp

注意: DeepSeek API 不支持 content 为数组格式的消息,
     需要用 DeepSeekChat 包装类将 list content 展平为字符串。
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver

load_dotenv(dotenv_path=".env")

MCP_SERVER_URL = "https://htai-test.chinalin.com/mcp"

SYSTEM_PROMPT = """你是一位专业的A股股票分析师。
你可以使用提供的工具查询股票的实时行情、估值数据、财务指标、技术指标等信息。
请基于工具返回的数据进行专业分析，用中文回答用户问题。
回答要简洁、有条理，必要时使用表格展示关键数据。

重要提示:
- 股票代码格式: sh+代码(上交所) 或 sz+代码(深交所), 例如 sh600519(贵州茅台), sz002594(比亚迪), sh600036(招商银行)
- 当用户提到股票名称时，你需要知道对应的市场代码来调用工具。"""


class DeepSeekChat(ChatOpenAI):
    """兼容 DeepSeek API 的 ChatOpenAI 包装类。

    DeepSeek 不支持 OpenAI 多模态格式 (content 为 list),
    此类在调用 API 前将所有消息的 list content 展平为纯字符串。
    """

    @staticmethod
    def _flatten_content(messages: list[BaseMessage]) -> list[BaseMessage]:
        result = []
        for msg in messages:
            if isinstance(msg.content, list):
                parts = []
                for block in msg.content:
                    if isinstance(block, dict):
                        parts.append(block.get("text", str(block)))
                    elif isinstance(block, str):
                        parts.append(block)
                    else:
                        parts.append(str(block))
                msg = msg.model_copy(update={"content": "\n".join(parts)})
            result.append(msg)
        return result

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return super()._generate(
            self._flatten_content(messages), stop=stop, run_manager=run_manager, **kwargs
        )

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return await super()._agenerate(
            self._flatten_content(messages), stop=stop, run_manager=run_manager, **kwargs
        )


def create_model():
    return DeepSeekChat(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
        temperature=0.3,
        max_tokens=4000,
    )


async def main():
    print("=" * 60)
    print("  Stock Analysis Agent - MCP Tools")
    print(f"  MCP Server: {MCP_SERVER_URL}")
    print("=" * 60)

    # 1. Connect to MCP Server and load tools
    print("\nConnecting to MCP Server...")
    client = MultiServerMCPClient(
        {
            "stock": {
                "transport": "http",
                "url": MCP_SERVER_URL,
            }
        }
    )

    tools = await client.get_tools()
    print(f"Loaded {len(tools)} tools:")
    for t in tools:
        desc = t.description[:50] if t.description else ""
        print(f"  - {t.name}: {desc}...")

    # 2. Create agent
    model = create_model()
    checkpointer = MemorySaver()

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )

    config = {"configurable": {"thread_id": "stock-1"}}

    # 3. Run test queries
    queries = [
        "比亚迪今天的行情怎么样？",
        "贵州茅台的PE估值是多少？在历史上处于什么水平？",
        "帮我分析一下招商银行的财务状况",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {query}")
        print("=" * 60)

        try:
            resp = await agent.ainvoke(
                {"messages": [{"role": "user", "content": query}]},
                config=config,
            )
            ai_msg = resp["messages"][-1]
            print(f"\nAI:\n{ai_msg.content}")
        except Exception as e:
            print(f"\nError: {e}")

    # 4. Interactive mode
    print(f"\n{'='*60}")
    print("Interactive mode (type 'quit' to exit)")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nyou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() in ("quit", "exit"):
            print("Bye!")
            break

        try:
            resp = await agent.ainvoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
            )
            print(f"\nAI:\n{resp['messages'][-1].content}")
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    asyncio.run(main())
