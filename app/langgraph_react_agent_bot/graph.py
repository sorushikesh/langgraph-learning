from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from agent import initialize_finance_agent


class AgentState(TypedDict):
    input: str
    final_answer: Optional[str]


def invoke_agent_node(state: AgentState) -> AgentState:
    """Invoke the finance agent with the current state."""
    agent = initialize_finance_agent()
    result = agent.invoke({"input": state["input"]})
    return {"input": state["input"], "final_answer": result["output"]}


def should_continue(state: AgentState) -> str:
    """Determine if the agent should continue or finish."""
    return END if state.get("final_answer") else "invoke_agent"


def build_graph():
    """Build the state graph for the agent."""
    graph = StateGraph(AgentState)

    graph.add_node("invoke_agent", invoke_agent_node)

    graph.set_entry_point("invoke_agent")
    graph.add_conditional_edges("invoke_agent", should_continue)
    return graph.compile()
