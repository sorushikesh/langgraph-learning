import os

from dotenv import load_dotenv

load_dotenv()


class ModelDetails:

    """
    Configuration for anthropic claude sonnet 4
    """
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1024))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.5))
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL_ID = os.getenv("ANTHROPIC_MODEL_ID")


class Settings:
    """
    Variables
    """
    WIKI_LANG = os.getenv("WIKI_LANG")
