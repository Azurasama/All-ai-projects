from models.llm import LLMService

class ResponseGenerator:
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    def generate_response(self, strategy: str, conversation_history: str, current_intent: str) -> str:
        prompt = f"""
Conversation History:
{conversation_history}

Current Extracted Intent: {current_intent}

Based on the strategy provided in the system instruction, and the conversation history above, generate the next response for the Assistant. Do not include 'Assistant:' in your output. Just output the response text.
"""
        return self.llm.generate(prompt=prompt, system_instruction=strategy, json_mode=False)
