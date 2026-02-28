"""
LangGraph 全功能综合测试

覆盖 LangGraph 核心组件:
  1. StateGraph 基础     - 节点/边/状态定义, 构建并运行图
  2. State + Reducer     - 自定义状态与归约器 (Annotated + operator.add)
  3. Conditional Edges   - 条件路由, 根据状态动态选择分支
  4. Command             - 在节点中同时更新状态 + 控制路由
  5. Persistence         - 检查点持久化, get_state / get_state_history / 时间旅行
  6. Interrupt (HITL)    - 人机交互中断, interrupt() + Command(resume=...)
  7. Streaming           - 流式输出 (updates / messages 模式)
  8. Send (Map-Reduce)   - 并行扇出, 动态创建子任务
  9. Subgraph            - 子图嵌套, 父图调用子图
  10. LLM-in-Graph       - 在图节点中集成 LLM, 实现智能工作流

使用 DeepSeek (OpenAI 兼容接口) 作为 LLM 后端。
运行: python test_all_langgraph_features.py
"""

import asyncio
import os
import operator
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, AnyMessage
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, Send, interrupt

load_dotenv(dotenv_path=".env")


# ═══════════════════════════════════════════════════════════
# DeepSeek 兼容层 (复用)
# ═══════════════════════════════════════════════════════════


class DeepSeekChat(ChatOpenAI):
    """DeepSeek 不支持 list content, 展平为字符串。"""

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
        max_tokens=500,
    )
    defaults.update(kwargs)
    return DeepSeekChat(**defaults)


# ═══════════════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════════════


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════
# Test 1: StateGraph 基础
# ═══════════════════════════════════════════════════════════

from typing_extensions import TypedDict


def test_1_state_graph_basics():
    """节点 → 边 → 编译 → 执行"""
    print_header("Test 1: StateGraph 基础 - 节点/边/编译")

    class SimpleState(TypedDict):
        value: str

    def node_a(state: SimpleState):
        return {"value": state["value"] + " → A"}

    def node_b(state: SimpleState):
        return {"value": state["value"] + " → B"}

    def node_c(state: SimpleState):
        return {"value": state["value"] + " → C"}

    builder = StateGraph(SimpleState)
    builder.add_node("a", node_a)
    builder.add_node("b", node_b)
    builder.add_node("c", node_c)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", "c")
    builder.add_edge("c", END)
    graph = builder.compile()

    result = graph.invoke({"value": "START"})
    print(f"  Flow: {result['value']}")
    assert result["value"] == "START → A → B → C"
    print("  PASS")


# ═══════════════════════════════════════════════════════════
# Test 2: State + Reducer
# ═══════════════════════════════════════════════════════════


def test_2_state_reducers():
    """Annotated reducer: operator.add 累加列表"""
    print_header("Test 2: State + Reducer - 状态归约器")

    class AccState(TypedDict):
        count: int
        log: Annotated[list[str], operator.add]

    def step_1(state: AccState):
        return {"count": state["count"] + 10, "log": ["step_1 done"]}

    def step_2(state: AccState):
        return {"count": state["count"] * 2, "log": ["step_2 done"]}

    builder = StateGraph(AccState)
    builder.add_node("step_1", step_1)
    builder.add_node("step_2", step_2)
    builder.add_edge(START, "step_1")
    builder.add_edge("step_1", "step_2")
    builder.add_edge("step_2", END)
    graph = builder.compile()

    result = graph.invoke({"count": 5, "log": ["init"]})
    print(f"  count: {result['count']}  (5 +10 → 15 *2 → 30)")
    print(f"  log:   {result['log']}")
    assert result["count"] == 30
    assert result["log"] == ["init", "step_1 done", "step_2 done"]
    print("  PASS")


# ═══════════════════════════════════════════════════════════
# Test 3: Conditional Edges
# ═══════════════════════════════════════════════════════════


def test_3_conditional_edges():
    """根据状态动态路由到不同分支"""
    print_header("Test 3: Conditional Edges - 条件路由")

    class RouteState(TypedDict):
        score: int
        result: str

    def evaluate(state: RouteState):
        return state

    def route_fn(state: RouteState) -> str:
        if state["score"] >= 90:
            return "excellent"
        elif state["score"] >= 60:
            return "pass"
        else:
            return "fail"

    def excellent(state: RouteState):
        return {"result": "优秀"}

    def pass_node(state: RouteState):
        return {"result": "及格"}

    def fail_node(state: RouteState):
        return {"result": "不及格"}

    builder = StateGraph(RouteState)
    builder.add_node("evaluate", evaluate)
    builder.add_node("excellent", excellent)
    builder.add_node("pass", pass_node)
    builder.add_node("fail", fail_node)
    builder.add_edge(START, "evaluate")
    builder.add_conditional_edges("evaluate", route_fn)
    builder.add_edge("excellent", END)
    builder.add_edge("pass", END)
    builder.add_edge("fail", END)
    graph = builder.compile()

    cases = [(95, "优秀"), (75, "及格"), (40, "不及格")]
    for score, expected in cases:
        r = graph.invoke({"score": score, "result": ""})
        status = "OK" if r["result"] == expected else "FAIL"
        print(f"  score={score} → {r['result']}  [{status}]")
        assert r["result"] == expected

    print("  PASS")


# ═══════════════════════════════════════════════════════════
# Test 4: Command (状态更新 + 路由)
# ═══════════════════════════════════════════════════════════


def test_4_command():
    """在节点中同时更新状态并控制跳转"""
    print_header("Test 4: Command - 状态更新+路由")

    class CmdState(TypedDict):
        value: int
        path: Annotated[list[str], operator.add]

    def router(state: CmdState) -> Command[Literal["double", "triple"]]:
        if state["value"] % 2 == 0:
            return Command(
                update={"path": ["router→double"]},
                goto="double",
            )
        else:
            return Command(
                update={"path": ["router→triple"]},
                goto="triple",
            )

    def double(state: CmdState):
        return {"value": state["value"] * 2, "path": ["doubled"]}

    def triple(state: CmdState):
        return {"value": state["value"] * 3, "path": ["tripled"]}

    builder = StateGraph(CmdState)
    builder.add_node("router", router)
    builder.add_node("double", double)
    builder.add_node("triple", triple)
    builder.add_edge(START, "router")
    builder.add_edge("double", END)
    builder.add_edge("triple", END)
    graph = builder.compile()

    r1 = graph.invoke({"value": 4, "path": []})
    print(f"  value=4 (偶数) → {r1['value']}, path={r1['path']}")
    assert r1["value"] == 8

    r2 = graph.invoke({"value": 3, "path": []})
    print(f"  value=3 (奇数) → {r2['value']}, path={r2['path']}")
    assert r2["value"] == 9

    print("  PASS")


# ═══════════════════════════════════════════════════════════
# Test 5: Persistence (检查点)
# ═══════════════════════════════════════════════════════════


def test_5_persistence():
    """检查点: 持久化状态, get_state, get_state_history, 时间旅行"""
    print_header("Test 5: Persistence - 检查点持久化")

    class CountState(TypedDict):
        counter: int
        log: Annotated[list[str], operator.add]

    def increment(state: CountState):
        new_val = state["counter"] + 1
        return {"counter": new_val, "log": [f"inc→{new_val}"]}

    def finalize(state: CountState):
        return {"log": [f"final={state['counter']}"]}

    builder = StateGraph(CountState)
    builder.add_node("increment", increment)
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "increment")
    builder.add_edge("increment", "finalize")
    builder.add_edge("finalize", END)

    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "persist-1"}}
    result = graph.invoke({"counter": 0, "log": []}, config)
    print(f"  Result: counter={result['counter']}, log={result['log']}")

    # get_state: 获取最新状态
    snapshot = graph.get_state(config)
    print(f"  get_state: counter={snapshot.values['counter']}, next={snapshot.next}")
    assert snapshot.values["counter"] == 1

    # get_state_history: 历史快照
    history = list(graph.get_state_history(config))
    print(f"  get_state_history: {len(history)} checkpoints")
    assert len(history) >= 3  # input + increment + finalize (+ start)

    # 时间旅行: 回溯到 increment 之前
    past_config = history[-2].config  # input checkpoint
    past_state = graph.get_state(past_config)
    print(f"  Time travel → counter={past_state.values['counter']}")

    print("  PASS")


# ═══════════════════════════════════════════════════════════
# Test 6: Interrupt (人机交互)
# ═══════════════════════════════════════════════════════════


def test_6_interrupt():
    """interrupt() 暂停, Command(resume=...) 恢复"""
    print_header("Test 6: Interrupt (HITL) - 人机交互")

    class ApprovalState(TypedDict):
        action: str
        status: str

    def request_approval(state: ApprovalState) -> Command[Literal["execute", "cancel"]]:
        decision = interrupt({
            "question": f"是否批准操作: {state['action']}?",
            "options": ["approve", "reject"],
        })
        if decision == "approve":
            return Command(goto="execute")
        else:
            return Command(goto="cancel")

    def execute(state: ApprovalState):
        return {"status": "approved_and_executed"}

    def cancel(state: ApprovalState):
        return {"status": "rejected"}

    builder = StateGraph(ApprovalState)
    builder.add_node("request_approval", request_approval)
    builder.add_node("execute", execute)
    builder.add_node("cancel", cancel)
    builder.add_edge(START, "request_approval")
    builder.add_edge("execute", END)
    builder.add_edge("cancel", END)

    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    # 场景1: 批准
    config1 = {"configurable": {"thread_id": "hitl-approve"}}
    r1 = graph.invoke({"action": "部署上线", "status": "pending"}, config1)
    print(f"  中断返回: {r1.get('__interrupt__', [{}])[0]}")

    r1_resume = graph.invoke(Command(resume="approve"), config1)
    print(f"  批准后: status={r1_resume['status']}")
    assert r1_resume["status"] == "approved_and_executed"

    # 场景2: 拒绝
    config2 = {"configurable": {"thread_id": "hitl-reject"}}
    graph.invoke({"action": "删除数据库", "status": "pending"}, config2)
    r2_resume = graph.invoke(Command(resume="reject"), config2)
    print(f"  拒绝后: status={r2_resume['status']}")
    assert r2_resume["status"] == "rejected"

    print("  PASS")


# ═══════════════════════════════════════════════════════════
# Test 7: Streaming (流式输出)
# ═══════════════════════════════════════════════════════════


def test_7_streaming():
    """stream_mode=updates 获取每步更新"""
    print_header("Test 7: Streaming - 流式更新")

    class PipeState(TypedDict):
        data: str
        steps: Annotated[list[str], operator.add]

    def clean(state: PipeState):
        return {"data": state["data"].strip().lower(), "steps": ["cleaned"]}

    def transform(state: PipeState):
        return {"data": state["data"].replace(" ", "_"), "steps": ["transformed"]}

    def output(state: PipeState):
        return {"data": f"[{state['data']}]", "steps": ["output"]}

    builder = StateGraph(PipeState)
    builder.add_node("clean", clean)
    builder.add_node("transform", transform)
    builder.add_node("output", output)
    builder.add_edge(START, "clean")
    builder.add_edge("clean", "transform")
    builder.add_edge("transform", "output")
    builder.add_edge("output", END)
    graph = builder.compile()

    print("  Stream updates:")
    for chunk in graph.stream(
        {"data": "  Hello World  ", "steps": []},
        stream_mode="updates",
    ):
        for node_name, update in chunk.items():
            print(f"    [{node_name}] data={update.get('data', '?')}")

    final = graph.invoke({"data": "  Hello World  ", "steps": []})
    print(f"  Final: data={final['data']}, steps={final['steps']}")
    assert final["data"] == "[hello_world]"
    print("  PASS")


# ═══════════════════════════════════════════════════════════
# Test 8: Send (Map-Reduce 并行)
# ═══════════════════════════════════════════════════════════


def test_8_send_map_reduce():
    """Send 动态扇出, 并行处理后汇总"""
    print_header("Test 8: Send (Map-Reduce) - 并行扇出")

    class OverallState(TypedDict):
        items: list[str]
        results: Annotated[list[str], operator.add]

    class WorkerState(TypedDict):
        item: str
        results: Annotated[list[str], operator.add]

    def fan_out(state: OverallState):
        return [Send("worker", {"item": item}) for item in state["items"]]

    def worker(state: WorkerState):
        processed = f"{state['item'].upper()}!"
        return {"results": [processed]}

    builder = StateGraph(OverallState)
    builder.add_node("worker", worker)
    builder.add_conditional_edges(START, fan_out)
    builder.add_edge("worker", END)
    graph = builder.compile()

    result = graph.invoke({"items": ["apple", "banana", "cherry"], "results": []})
    print(f"  Input:  {['apple', 'banana', 'cherry']}")
    print(f"  Output: {sorted(result['results'])}")
    assert len(result["results"]) == 3
    assert "APPLE!" in result["results"]
    print("  PASS")


# ═══════════════════════════════════════════════════════════
# Test 9: Subgraph (子图嵌套)
# ═══════════════════════════════════════════════════════════


def test_9_subgraph():
    """父图中嵌套调用子图"""
    print_header("Test 9: Subgraph - 子图嵌套")

    # --- 子图: 验证流水线 ---
    class ValidateState(TypedDict):
        text: str
        valid: bool

    def check_length(state: ValidateState):
        return {"valid": len(state["text"]) >= 3}

    def check_content(state: ValidateState):
        if not state["valid"]:
            return state
        has_bad = any(w in state["text"] for w in ["spam", "xxx"])
        return {"valid": not has_bad}

    sub_builder = StateGraph(ValidateState)
    sub_builder.add_node("check_length", check_length)
    sub_builder.add_node("check_content", check_content)
    sub_builder.add_edge(START, "check_length")
    sub_builder.add_edge("check_length", "check_content")
    sub_builder.add_edge("check_content", END)
    sub_graph = sub_builder.compile()

    # --- 父图 ---
    class MainState(TypedDict):
        text: str
        valid: bool
        output: str

    def preprocess(state: MainState):
        return {"text": state["text"].strip()}

    def validate(state: MainState):
        r = sub_graph.invoke({"text": state["text"], "valid": True})
        return {"valid": r["valid"]}

    def decide(state: MainState) -> str:
        return "accept" if state["valid"] else "reject"

    def accept(state: MainState):
        return {"output": f"Accepted: {state['text']}"}

    def reject(state: MainState):
        return {"output": f"Rejected: {state['text']}"}

    main_builder = StateGraph(MainState)
    main_builder.add_node("preprocess", preprocess)
    main_builder.add_node("validate", validate)
    main_builder.add_node("accept", accept)
    main_builder.add_node("reject", reject)
    main_builder.add_edge(START, "preprocess")
    main_builder.add_edge("preprocess", "validate")
    main_builder.add_conditional_edges("validate", decide)
    main_builder.add_edge("accept", END)
    main_builder.add_edge("reject", END)
    main_graph = main_builder.compile()

    cases = [
        ("Hello World", True),
        ("Hi", False),        # 长度 < 3
        ("spam is bad", False),  # 含违禁词
    ]
    for text, expect_valid in cases:
        r = main_graph.invoke({"text": text, "valid": False, "output": ""})
        ok = ("Accepted" in r["output"]) == expect_valid
        print(f"  '{text}' → {r['output']}  [{'OK' if ok else 'FAIL'}]")
        assert ok

    print("  PASS")


# ═══════════════════════════════════════════════════════════
# Test 10: LLM-in-Graph (LLM 集成到图节点)
# ═══════════════════════════════════════════════════════════


def test_10_llm_in_graph():
    """在 StateGraph 节点中调用 LLM, 实现多步推理工作流"""
    print_header("Test 10: LLM-in-Graph - LLM 智能工作流")

    class AnalysisState(TypedDict):
        topic: str
        outline: str
        article: str
        messages: Annotated[list[AnyMessage], add_messages]

    model = create_model(max_tokens=300)

    def generate_outline(state: AnalysisState):
        resp = model.invoke([
            SystemMessage(content="你是写作助手, 用中文回答, 简洁明了"),
            HumanMessage(content=f"为'{state['topic']}'生成一个3点提纲, 每点一行"),
        ])
        return {
            "outline": resp.content,
            "messages": [
                HumanMessage(content=f"生成提纲: {state['topic']}"),
                resp,
            ],
        }

    def write_article(state: AnalysisState):
        resp = model.invoke([
            SystemMessage(content="你是写作助手, 基于提纲写100字以内的短文, 中文"),
            HumanMessage(content=f"提纲:\n{state['outline']}\n\n请写短文:"),
        ])
        return {
            "article": resp.content,
            "messages": [
                HumanMessage(content="基于提纲写短文"),
                resp,
            ],
        }

    builder = StateGraph(AnalysisState)
    builder.add_node("generate_outline", generate_outline)
    builder.add_node("write_article", write_article)
    builder.add_edge(START, "generate_outline")
    builder.add_edge("generate_outline", "write_article")
    builder.add_edge("write_article", END)
    graph = builder.compile()

    result = graph.invoke({
        "topic": "Python编程的优势",
        "outline": "",
        "article": "",
        "messages": [],
    })

    print(f"  Topic: {result['topic']}")
    print(f"  Outline:\n    {result['outline'][:100]}...")
    print(f"  Article:\n    {result['article'][:120]}...")
    print(f"  Message count: {len(result['messages'])}")

    assert len(result["outline"]) > 0
    assert len(result["article"]) > 0
    assert len(result["messages"]) >= 4
    print("  PASS")


# ═══════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════


def main():
    print("=" * 60)
    print("  LangGraph All Features Test")
    print("  Framework: LangGraph (graph orchestration)")
    print("=" * 60)

    tests = [
        ("1.  StateGraph 基础",      test_1_state_graph_basics),
        ("2.  State + Reducer",      test_2_state_reducers),
        ("3.  Conditional Edges",    test_3_conditional_edges),
        ("4.  Command",              test_4_command),
        ("5.  Persistence",          test_5_persistence),
        ("6.  Interrupt (HITL)",     test_6_interrupt),
        ("7.  Streaming",            test_7_streaming),
        ("8.  Send (Map-Reduce)",    test_8_send_map_reduce),
        ("9.  Subgraph",             test_9_subgraph),
        ("10. LLM-in-Graph",        test_10_llm_in_graph),
    ]

    results = []
    for name, fn in tests:
        try:
            fn()
            results.append((name, True, ""))
        except Exception as e:
            print(f"\n  FAIL: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, str(e)))

    # 汇总
    print_header("Test Summary")
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    for name, ok, err in results:
        status = "PASS" if ok else f"FAIL ({err[:50]})"
        print(f"  {'[OK]' if ok else '[!!]'} {name}: {status}")
    print(f"\n  Result: {passed}/{total} passed")


if __name__ == "__main__":
    main()
