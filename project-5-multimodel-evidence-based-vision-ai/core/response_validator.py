from models.llm import LLMModel

class ResponseValidator:
    def __init__(self, llm_model: LLMModel):
        self.llm = llm_model
        
    def validate(self, generated_response: str, user_query: str, evidence_text: str) -> dict:
        """
        Validate that the generated response is supported by evidence and doesn't hallucinate.
        """
        prompt = f"""
        User Query: {user_query}
        
        Provided Evidence:
        {evidence_text}
        
        Generated AI Response:
        {generated_response}
        
        Evaluate the AI response based on the following criteria:
        1. "is_supported": Does the evidence actually support the claims made in the response? (Boolean)
        2. "has_hallucination": Does the response invent facts or details not present in the evidence? (Boolean)
        3. "addresses_query": Does the response answer the user's question? (Boolean)
        4. "correction_needed": If it failed any checks, what needs to be fixed? (String)
        """
        
        schema = {
            "type": "OBJECT",
            "properties": {
                "is_supported": {"type": "BOOLEAN"},
                "has_hallucination": {"type": "BOOLEAN"},
                "addresses_query": {"type": "BOOLEAN"},
                "correction_needed": {"type": "STRING"}
            },
            "required": ["is_supported", "has_hallucination", "addresses_query", "correction_needed"]
        }
        
        system_prompt = "You are a strict evidence validator. Your job is to prevent hallucination in AI systems."
        return self.llm.generate_structured(prompt, response_schema=schema, system_instruction=system_prompt)
