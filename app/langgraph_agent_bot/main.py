from typing import List

import langchain_anthropic
from flask import Flask, request, jsonify
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph
from langgraph.constants import END
from pydantic import BaseModel

from app.app.constants.config import ModelDetails

app = Flask(__name__)

# ---- Prompt for Summary ----
summary_prompt = PromptTemplate.from_template(
    """
    Progressively summarize the conversation.

    Summary so far:
    {summary}

    New lines of conversation:
    {new_lines}

    Updated Summary:
    """
)

# -- Initialize LLM --
llm = langchain_anthropic.ChatAnthropic(
    model_name=ModelDetails.ANTHROPIC_MODEL_ID,
    max_tokens_to_sample=ModelDetails.MAX_TOKENS,
    temperature=ModelDetails.TEMPERATURE,
    timeout=None,
    stop=None,
)

# ---- Summary Memory ----
memory = ConversationSummaryMemory(
    llm=llm, prompt=summary_prompt, memory_key="chat_history", return_messages=True
)


# -- Graph State Definition --
class GraphState(BaseModel):
    chat_history: List[BaseMessage]
    user_input: str


# ---- User Input Node ----
def user_input_node(state: GraphState) -> GraphState:
    """Node to handle user input and append it to chat history."""
    user_msg = HumanMessage(content=state.user_input)
    state.chat_history.append(user_msg)
    return state


# ---- Assistant Response Node ----
def assistant_node(state: GraphState) -> GraphState:
    """Node to generate assistant response based on user input."""
    summary = memory.load_memory_variables({})
    messages = summary["chat_history"] + [HumanMessage(content=state.user_input)]
    response = llm.invoke(messages)
    state.chat_history.append(response)

    # update memory
    memory.save_context({"input": state.user_input}, {"output": response.content})
    return state


# -- Graph Construction --
def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("user_input", user_input_node)
    graph.add_node("assistant", assistant_node)

    graph.set_entry_point("user_input")
    graph.add_edge("user_input", "assistant")
    graph.add_edge("assistant", END)

    return graph.compile()


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("user_message")
    if not user_input:
        return jsonify({"error": "No user_message provided"}), 400

    app_graph = build_graph()
    initial_state = GraphState(
        chat_history=memory.chat_memory.messages, user_input=user_input
    )
    final_state = app_graph.invoke(initial_state)

    return jsonify({"response": final_state["chat_history"][-1].content})


if __name__ == "__main__":
    app.run(port=8080, debug=True)
