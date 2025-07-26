import langchain_anthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.agents import initialize_agent, AgentType, Tool

from app.constants.config import ModelDetails
from tools import categorize_transaction, calculate_spend, check_budget_violation


# -- LLM --
def get_llm() -> BaseChatModel:
    return langchain_anthropic.ChatAnthropic(
        model_name=ModelDetails.ANTHROPIC_MODEL_ID,
        max_tokens_to_sample=ModelDetails.MAX_TOKENS,
        temperature=ModelDetails.TEMPERATURE,
        timeout=None,
        stop=None,
    )


def initialize_finance_agent():
    """Initialize the finance agent with tools and LLM."""
    llm = get_llm()
    tools = [
        Tool.from_function(
            categorize_transaction,
            name="CategorizeTransaction",
            description="Categorizes a transaction based on its description.",
        ),
        Tool.from_function(
            calculate_spend,
            name="CalculateSpend",
            description="Calculates total spend from a list of transactions.",
        ),
        Tool.from_function(
            check_budget_violation,
            name="CheckBudgetViolation",
            description="Checks if spending exceeds the given budget.",
        ),
    ]

    return initialize_agent(
        tools=tools, llm=llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
