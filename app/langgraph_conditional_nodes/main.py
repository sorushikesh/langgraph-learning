from flask import Flask, request, jsonify
from langgraph.graph import StateGraph
from langgraph.constants import END
from langchain_core.messages import HumanMessage, AIMessage
from typing import Annotated, TypedDict
import re

app = Flask(__name__)


# -- Graph State --
class GraphState(TypedDict):
    history: Annotated[list[HumanMessage | AIMessage], ...]


# -- Greet Node --
def greet_node(state: GraphState) -> GraphState:
    """Initial greeting node that sets up the conversation."""
    msg = AIMessage(content="Hi! I'm a calculator. Send me a math expression.")
    return {"history": state["history"] + [msg]}


# -- User Input Node --
def user_input_node(state: GraphState, user_input: str) -> GraphState:
    """User input node that captures the user's message."""
    return {"history": state["history"] + [HumanMessage(content=user_input.strip())]}


# -- Router Node (just returns state) --
def router_node(state: GraphState) -> GraphState:
    """Router node that decides which node to go to based on user input."""
    return state


# -- Router Decision Function --
def router(state: GraphState) -> str:
    """Decides which node to route to based on user input."""
    user_input = state["history"][-1].content
    if re.fullmatch(r"[0-9+\-*/ ().%]+", user_input):
        return "calculate"
    return "llm"


# -- Calculator Node --
def calculate_node(state: GraphState) -> GraphState:
    """Node that evaluates the user's mathematical expression."""
    user_input = state["history"][-1].content
    try:
        result = eval(user_input, {"__builtins__": {}}, {})
        reply = AIMessage(content=f"Result: {result}")
    except Exception:
        reply = AIMessage(content="Sorry, I couldn't calculate that.")
    return {"history": state["history"] + [reply]}


# -- Fallback LLM Node --
def llm_node(state: GraphState) -> GraphState:
    """Fallback node that handles non-mathematical user input."""
    user_input = state["history"][-1].content
    reply = AIMessage(
        content=f"I'm a calculator assistant. You asked: '{user_input}', but I can only do math."
    )
    return {"history": state["history"] + [reply]}


# -- Build Graph --
def build_graph(user_input: str):
    """Builds the state graph for the calculator application."""
    graph = StateGraph(GraphState)

    graph.add_node("greet", greet_node)
    graph.add_node("user", lambda s: user_input_node(s, user_input))
    graph.add_node("router", router_node)
    graph.add_node("calculate", calculate_node)
    graph.add_node("llm", llm_node)

    graph.set_entry_point("greet")
    graph.add_edge("greet", "user")
    graph.add_edge("user", "router")
    graph.add_conditional_edges(
        "router",
        router,
        {
            "calculate": "calculate",
            "llm": "llm",
        },
    )
    graph.add_edge("calculate", END)
    graph.add_edge("llm", END)

    return graph.compile()


# -- API Endpoint --
@app.route("/chat", methods=["POST"])
def chat():
    """API endpoint for the calculator chat."""
    data = request.get_json()
    user_input = data.get("user_message", "").strip()
    if not user_input:
        return jsonify({"error": "Missing user_message"}), 400

    graph = build_graph(user_input)
    final_state = graph.invoke({"history": []})
    messages = [
        {"type": msg.type, "content": msg.content} for msg in final_state["history"]
    ]
    return jsonify({"conversation": messages})


# -- Run App --
if __name__ == "__main__":
    import os
    debug_mode = os.getenv("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=8080, debug=debug_mode)
