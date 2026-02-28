"""
LangChain 全功能综合测试

覆盖 LangChain 核心组件:
  1. Model Invoke    - 基础模型调用
  2. Streaming       - 流式输出 (逐 token)
  3. Batch           - 批量调用
  4. Tool Calling    - 手动工具调用循环 (bind_tools)
  5. Structured Output - 结构化输出 (Pydantic)
  6. Agent + Tools   - 智能体自动工具调用
  7. Short-term Memory - 对话记忆 (多轮上下文)
  8. MCP Integration - MCP Server 远程工具加载

使用 DeepSeek (OpenAI 兼容接口) 作为 LLM 后端。
运行: python test_all_features.py
"""

import asyncio
import os
import time
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv(dotenv_path=".env")


# ═══════════════════════════════════════════════════════════
# DeepSeek 兼容层
# ═══════════════════════════════════════════════════════════


class DeepSeekChat(ChatOpenAI):
    """兼容 DeepSeek API 的 ChatOpenAI 包装类。

    DeepSeek 不支持 OpenAI 多模态消息格式 (content 为 list),
    此类在调用前将 list content 展平为纯字符串。
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


def create_model(**kwargs):
    defaults = dict(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
        temperature=0.3,
        max_tokens=1000,
    )
    defaults.update(kwargs)
    return DeepSeekChat(**defaults)


# ═══════════════════════════════════════════════════════════
# 工具定义
# ═══════════════════════════════════════════════════════════


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气"""
    data = {
        "北京": "晴天, 25°C, 湿度40%",
        "上海": "多云, 28°C, 湿度65%",
        "深圳": "雷阵雨, 30°C, 湿度80%",
    }
    return data.get(city, f"{city}: 暂无数据")


@tool
def calculate(expression: str) -> str:
    """计算数学表达式, 例如 '2 + 3 * 4'"""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "错误: 含非法字符"
    try:
        return f"{expression} = {eval(expression)}"
    except Exception as e:
        return f"计算错误: {e}"


@tool
def get_current_date() -> str:
    """获取当前日期和时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ═══════════════════════════════════════════════════════════
# 结构化输出 Schema
# ═══════════════════════════════════════════════════════════


class SentimentResult(BaseModel):
    """文本情感分析结果"""
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="情感倾向")
    confidence: float = Field(description="置信度 0-1", ge=0, le=1)
    keywords: list[str] = Field(description="关键词列表, 每项1-3个字")
    summary: str = Field(description="一句话总结")


# ═══════════════════════════════════════════════════════════
# 测试用例
# ═══════════════════════════════════════════════════════════


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_1_invoke():
    """Test 1: 基础模型调用 (invoke)"""
    print_header("Test 1: Model Invoke - 基础模型调用")

    model = create_model()

    # 简单字符串调用
    resp = model.invoke("用一句话解释什么是 LangChain")
    print(f"[String input]  AI: {resp.content}")

    # Message 对象调用 (带对话历史)
    messages = [
        SystemMessage(content="你是翻译助手, 将中文翻译为英文"),
        HumanMessage(content="今天天气真好"),
    ]
    resp = model.invoke(messages)
    print(f"[Messages input] AI: {resp.content}")

    print("PASS")


def test_2_streaming():
    """Test 2: 流式输出 (streaming)"""
    print_header("Test 2: Streaming - 流式输出")

    model = create_model()
    print("AI: ", end="", flush=True)

    full_text = ""
    for chunk in model.stream("写一首关于编程的四行诗"):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            full_text += chunk.content
    print()

    assert len(full_text) > 0, "流式输出为空"
    print("PASS")


def test_3_batch():
    """Test 3: 批量调用 (batch)"""
    print_header("Test 3: Batch - 批量调用")

    model = create_model(max_tokens=50)
    questions = [
        "Python 的创建者是谁?",
        "HTTP 200 表示什么?",
        "JSON 的全称是什么?",
    ]

    start = time.time()
    responses = model.batch(questions, config={"max_concurrency": 3})
    elapsed = time.time() - start

    for q, r in zip(questions, responses):
        print(f"  Q: {q}")
        print(f"  A: {r.content[:80]}...")
        print()

    print(f"  Batch completed in {elapsed:.1f}s ({len(questions)} requests)")
    print("PASS")


def test_4_tool_calling():
    """Test 4: 手动工具调用循环 (bind_tools)"""
    print_header("Test 4: Tool Calling - 手动工具调用")

    model = create_model()
    model_with_tools = model.bind_tools([get_weather, calculate])

    messages = [HumanMessage(content="北京天气怎么样? 另外帮我算 (100+200)*3")]
    ai_msg = model_with_tools.invoke(messages)
    messages.append(ai_msg)

    print(f"  Model requested {len(ai_msg.tool_calls)} tool call(s):")
    for tc in ai_msg.tool_calls:
        print(f"    - {tc['name']}({tc['args']})")

    tool_map = {"get_weather": get_weather, "calculate": calculate}
    for tc in ai_msg.tool_calls:
        tool_result = tool_map[tc["name"]].invoke(tc)
        messages.append(tool_result)
        print(f"  Tool [{tc['name']}] returned: {tool_result.content}")

    final = model_with_tools.invoke(messages)
    print(f"\n  Final AI: {final.content}")
    print("PASS")


def test_5_structured_output():
    """Test 5: 结构化输出 (Pydantic model)"""
    print_header("Test 5: Structured Output - 结构化输出")

    model = create_model()

    # DeepSeek 不支持 response_format, 用 function_calling 方式
    structured_model = model.with_structured_output(SentimentResult, method="function_calling")
    result = structured_model.invoke(
        "分析这段评论的情感: '这个产品质量非常好，物流也很快，但是价格偏贵，总体还是值得推荐的'"
    )

    print(f"  Type:       {type(result).__name__}")
    print(f"  Sentiment:  {result.sentiment}")
    print(f"  Confidence: {result.confidence}")
    print(f"  Keywords:   {result.keywords}")
    print(f"  Summary:    {result.summary}")

    assert isinstance(result, SentimentResult), "返回类型错误"
    assert result.sentiment in ("positive", "negative", "neutral")
    print("PASS")


def test_6_agent_with_tools():
    """Test 6: Agent 智能体 (自动工具调用)"""
    print_header("Test 6: Agent + Tools - 智能体")

    model = create_model()
    agent = create_agent(
        model=model,
        tools=[get_weather, calculate, get_current_date],
        system_prompt="你是智能助手, 用中文简洁回答。需要时主动调用工具。",
    )

    queries = [
        "现在几点了?",
        "上海天气怎么样? 顺便算一下 15*24 是多少",
    ]
    for q in queries:
        resp = agent.invoke({"messages": [{"role": "user", "content": q}]})
        ai_msg = resp["messages"][-1]
        print(f"  Q: {q}")
        print(f"  A: {ai_msg.content[:120]}")
        print()

    print("PASS")


def test_7_memory():
    """Test 7: 对话记忆 (Short-term Memory)"""
    print_header("Test 7: Short-term Memory - 对话记忆")

    model = create_model()
    checkpointer = MemorySaver()

    agent = create_agent(
        model=model,
        tools=[calculate],
        system_prompt="你是智能助手, 用中文简洁回答。",
        checkpointer=checkpointer,
    )

    config = {"configurable": {"thread_id": "memory-test-1"}}

    conversations = [
        "我叫小明, 请记住我的名字",
        "帮我算 99 * 88",
        "我刚才说我叫什么? 刚才的计算结果是多少?",
    ]

    for msg in conversations:
        resp = agent.invoke(
            {"messages": [{"role": "user", "content": msg}]},
            config=config,
        )
        ai_msg = resp["messages"][-1]
        print(f"  User: {msg}")
        print(f"  AI:   {ai_msg.content[:120]}")
        print()

    # 验证: 最后一条回复应包含 "小明" (记住了名字)
    last_reply = resp["messages"][-1].content
    assert "小明" in last_reply or "名字" in last_reply, "Agent 未能记住用户名字"
    print("PASS")


async def test_8_mcp():
    """Test 8: MCP Server 远程工具集成"""
    print_header("Test 8: MCP Integration - 远程工具")

    from langchain_mcp_adapters.client import MultiServerMCPClient

    mcp_url = "https://htai-test.chinalin.com/mcp"
    print(f"  Connecting to {mcp_url}...")

    client = MultiServerMCPClient(
        {"stock": {"transport": "http", "url": mcp_url}}
    )
    tools = await client.get_tools()
    print(f"  Loaded {len(tools)} MCP tools")

    # 只展示前 5 个工具
    for t in tools[:5]:
        print(f"    - {t.name}")
    if len(tools) > 5:
        print(f"    ... and {len(tools)-5} more")

    model = create_model(max_tokens=2000)
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="你是A股分析师, 简洁回答。股票代码格式: sh+代码(上交所) 或 sz+代码(深交所)。",
    )

    resp = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "查一下贵州茅台的PE估值"}]}
    )
    ai_msg = resp["messages"][-1]
    print(f"\n  Q: 查一下贵州茅台的PE估值")
    print(f"  A: {ai_msg.content[:200]}...")
    print("PASS")


# ═══════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════


async def main():
    print("=" * 60)
    print("  LangChain All Features Test")
    print("  Model: DeepSeek (OpenAI-compatible)")
    print("=" * 60)

    tests = [
        # ("1. Model Invoke",       test_1_invoke),
        ("2. Streaming",          test_2_streaming),
        # ("3. Batch",              test_3_batch),
        # ("4. Tool Calling",       test_4_tool_calling),
        # ("5. Structured Output",  test_5_structured_output),
        # ("6. Agent + Tools",      test_6_agent_with_tools),
        # ("7. Short-term Memory",  test_7_memory),
        # ("8. MCP Integration",    test_8_mcp),
    ]

    results = []
    for name, fn in tests:
        try:
            if asyncio.iscoroutinefunction(fn):
                await fn()
            else:
                fn()
            results.append((name, True, ""))
        except Exception as e:
            print(f"\n  FAIL: {e}")
            results.append((name, False, str(e)))

    # 汇总
    print_header("Test Summary")
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    for name, ok, err in results:
        status = "PASS" if ok else f"FAIL ({err[:40]})"
        print(f"  {'[OK]' if ok else '[!!]'} {name}: {status}")
    print(f"\n  Result: {passed}/{total} passed")


if __name__ == "__main__":
    asyncio.run(main())
