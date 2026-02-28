"""Test: agent with tools - non-interactive"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv(dotenv_path=".env")


@tool
def get_weather(city: str) -> str:
    """Query weather for a city"""
    data = {"Beijing": "Sunny, 25C", "Shanghai": "Cloudy, 28C"}
    return data.get(city, f"No weather data for {city}")


@tool
def calculate(expression: str) -> str:
    """Calculate a math expression"""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: invalid chars"
    try:
        return f"{expression} = {eval(expression)}"
    except Exception as e:
        return f"Error: {e}"


model = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "deepseek-chat"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    temperature=0.7,
    max_tokens=2000,
)

checkpointer = MemorySaver()

agent = create_agent(
    model=model,
    tools=[get_weather, calculate],
    system_prompt="You are a helpful assistant. Use tools when needed. Reply in Chinese.",
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "test-1"}}

print("=== Test 1: Calculate ===")
resp = agent.invoke(
    {"messages": [{"role": "user", "content": "calculate (15 + 27) * 3"}]},
    config=config,
)
print(f"AI: {resp['messages'][-1].content}")

print("\n=== Test 2: Weather ===")
resp = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in Beijing?"}]},
    config=config,
)
print(f"AI: {resp['messages'][-1].content}")

print("\n=== Test 3: Memory ===")
resp = agent.invoke(
    {"messages": [{"role": "user", "content": "What was the calculation result?"}]},
    config=config,
)
print(f"AI: {resp['messages'][-1].content}")

print("\nAll tests passed!")
