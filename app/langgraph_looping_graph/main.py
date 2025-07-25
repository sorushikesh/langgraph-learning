import random
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph
from langgraph.constants import END

class GraphState(TypedDict):
    name: str
    number: list[int]
    counter: int

# Greet node
def greet_node(state: GraphState) -> GraphState:
    state["name"] = f"Hey there, {state['name']}"
    state["counter"] = 0
    state["number"] = []
    return state

# Random node
def random_node(state: GraphState) -> GraphState:
    state["number"].append(random.randint(0, 10))
    state["counter"] += 1
    return state

# Branching logic
def should_continue(state: GraphState) -> str:
    if state["counter"] < 5:
        print(f"Looping... iteration {state['counter']}")
        return "loop"
    else:
        print("Exiting loop")
        return "exit"

def build_graph():
    graph_builder = StateGraph(GraphState)

    graph_builder.add_node("greet", greet_node)
    graph_builder.add_node("random", random_node)

    graph_builder.set_entry_point("greet")
    graph_builder.add_edge("greet", "random")

    graph_builder.add_conditional_edges(
        "random",
        should_continue,
        {
            "loop": "random",
            "exit": END
        }
    )

    return graph_builder.compile()


graph = build_graph()
initial_state = {"name": "Rushikesh", "number": [], "counter": 0}
final_state = graph.invoke(initial_state)
print(final_state)
