from langchain_core.tools import tool
import json


@tool
def categorize_transaction(description: str) -> str:
    """Categorize a transaction based on its description."""
    desc = description.lower()
    if any(word in desc for word in ["uber", "taxi", "ola"]):
        return "Transport"
    if any(word in desc for word in ["zomato", "pizza", "coffee"]):
        return "Food"
    if any(word in desc for word in ["amazon", "flipkart"]):
        return "Shopping"
    return "Other"


@tool
def calculate_spend(input_json: str) -> str:
    """Calculate total spend per category from transaction JSON."""
    try:
        data = json.loads(input_json)
        transactions = data["transactions"]
        totals = {}
        for txn in transactions:
            category = categorize_transaction(txn["desc"])
            totals[category] = totals.get(category, 0) + txn["amount"]
        return json.dumps(totals)
    except Exception as e:
        return f"Error processing input: {e}"


@tool
def check_budget_violation(input_json: str) -> str:
    """Check for budget violations using total spend and user budget."""
    try:
        data = json.loads(input_json)
        budget = data["budget"]
        spent = data["spent"]
        violations = {
            k: spent[k] - budget[k] for k in spent if spent[k] > budget.get(k, 0)
        }
        if not violations:
            return "No budget violations."
        return "Violations:\n" + "\n".join(
            [f"{k}: ₹{v} over" for k, v in violations.items()]
        )
    except Exception as e:
        return f"Error checking budget: {e}"
