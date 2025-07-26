from typing import Annotated, TypedDict

import langchain_anthropic
from flask import Flask, request, jsonify
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.constants import END
from langgraph.graph import StateGraph

from app.constants.config import ModelDetails
from app.constants.logger_config import setup_logger

app = Flask(__name__)
logger = setup_logger()


# -- Graph State Definition --
class GraphState(TypedDict):
    history: Annotated[list[HumanMessage | AIMessage], ...]


# -- Initialize LLM --
llm = langchain_anthropic.ChatAnthropic(
    model_name=ModelDetails.ANTHROPIC_MODEL_ID,
    max_tokens_to_sample=ModelDetails.MAX_TOKENS,
    temperature=ModelDetails.TEMPERATURE,
    timeout=None,
    stop=None,
)


# -- Node Definitions --
def greet_node(state: GraphState) -> GraphState:
    """Node to greet the user and initialize conversation history."""
    logger.info("Executing 'greet_node'")
    msg = AIMessage(content="Hi! I'm your assistant. How can I help you today?")
    return {"history": state["history"] + [msg]}


def user_response_node(state: GraphState, user_input: str) -> GraphState:
    """Node to process user responses."""
    if not isinstance(user_input, str) or not user_input.strip():
        raise ValueError("User input must be a non-empty string.")
    return {"history": state["history"] + [HumanMessage(content=user_input.strip())]}


def assistant_response_node(state: GraphState) -> GraphState:
    """Node to generate the assistant's response."""
    logger.info("Executing 'assistant_response_node'")
    try:
        response = llm.invoke(state["history"])
    except Exception as e:
        logger.exception("LLM invocation failed")
        response = AIMessage(
            content=f"An error occurred while generating response: {str(e)}"
        )
    return {"history": state["history"] + [response]}


# -- Graph Construction --
def build_graph(user_input: str):
    logger.info("Building graph pipeline...")
    graph = StateGraph(GraphState)

    graph.add_node("greet", greet_node)
    graph.add_node("user_responds", lambda state: user_response_node(state, user_input))
    graph.add_node("assistant_responds", assistant_response_node)

    graph.set_entry_point("greet")
    graph.add_edge("greet", "user_responds")
    graph.add_edge("user_responds", "assistant_responds")
    graph.add_edge("assistant_responds", END)

    return graph.compile()


# -- API Endpoint --
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("user_message")

    logger.info(f"Received /chat request with message: {user_message}")

    if not isinstance(user_message, str) or not user_message.strip():
        logger.warning("Invalid or missing 'user_message'")
        return jsonify({"error": "Invalid or missing 'user_message' in request"}), 400

    try:
        app_graph = build_graph(user_message.strip())
        final_state = app_graph.invoke({"history": []})
        logger.info("LangGraph execution completed.")
    except Exception as e:
        logger.exception("Internal server error during graph execution.")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

    conversation = [
        {"type": msg.type, "content": msg.content} for msg in final_state["history"]
    ]

    logger.info(f"Returning response: {conversation}")
    return jsonify({"conversation": conversation})


# -- Run Flask App --
if __name__ == "__main__":
    logger.info("Starting Flask LangGraph basics API on port 8080...")
    app.run(host="0.0.0.0", port=8080, debug=True)
