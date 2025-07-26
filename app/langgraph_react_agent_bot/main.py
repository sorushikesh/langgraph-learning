from flask import Flask, request, jsonify
from graph import build_graph

app = Flask(__name__)
graph = build_graph()


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    user_input = data.get("input", "")
    result = graph.invoke({"input": user_input})
    return jsonify({"response": result["final_answer"]})


if __name__ == "__main__":
    app.run(debug=True, port=8080)
