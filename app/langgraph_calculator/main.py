from flask import Flask, request, jsonify
from langgraph.graph import StateGraph
from langgraph.constants import END
from langchain_core.messages import HumanMessage, AIMessage
from typing import Annotated, TypedDict

app = Flask(__name__)


# -- Graph State --
class GraphState(TypedDict):
    history: Annotated[list[HumanMessage | AIMessage], ...]


def log_message(message: str):
    """Utility function to log messages."""
    print(f"[LOG] {message}")


# -- Nodes --
def greet_node(state: GraphState) -> GraphState:
    """Initial greeting node that sets up the conversation."""
    log_message("Starting calculator conversation.")
    msg = AIMessage(content="Hi! I'm a calculator. Send me a math expression.")
    return {"history": state["history"] + [msg]}


def user_input_node(state: GraphState, user_input: str) -> GraphState:
    """User input node that captures the user's message."""
    log_message(f"User input received: {user_input}")
    return {"history": state["history"] + [HumanMessage(content=user_input.strip())]}


def calculate_node(state: GraphState) -> GraphState:
    """Node that evaluates the user's mathematical expression."""
    log_message("Calculating the result of the user's input.")
    try:
        user_input = state["history"][-1].content
        result = eval(user_input, {"__builtins__": {}}, {})
        reply = AIMessage(content=f"Result: {result}")
    except Exception as e:
        reply = AIMessage(content=f"Error: {str(e)}")
    return {"history": state["history"] + [reply]}


# -- Build Graph --
def build_graph(user_input: str):

    graph = StateGraph(GraphState)
    graph.add_node("greet", greet_node)
    graph.add_node("user", lambda s: user_input_node(s, user_input))
    graph.add_node("calc", calculate_node)

    graph.set_entry_point("greet")
    graph.add_edge("greet", "user")
    graph.add_edge("user", "calc")
    graph.add_edge("calc", END)

    return graph.compile()


# -- API --
@app.route("/chat", methods=["POST"])
def chat():
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


# -- Run --
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
