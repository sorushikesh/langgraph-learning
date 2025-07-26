class PromptLoader:
    def __init__(self):
        self.user_prompt = self.get_user_prompt()

    def get_user_prompt(self) -> str:
        return (
            "You are a backpacker or digital nomad exploring Latin America. You're seeking travel tips, destination ideas, "
            "budget-friendly options, coworking spots, or information on culture, safety, and visas. Ask questions clearly "
            "so your guide can offer the most relevant advice."
        )

prompt_loader = PromptLoader()
