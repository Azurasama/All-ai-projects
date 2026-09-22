from medical.entity_extractor import MedicalEntityExtractor
from medical.context_manager import ContextManager
from retrieval.retriever import HybridRetriever
from models.llm import LLM
from utils.logging import get_logger

logger = get_logger(__name__)

class MedicalQAPipeline:
    def __init__(self):
        logger.info("Initializing QA Pipeline components...")
        self.ner = MedicalEntityExtractor()
        self.context = ContextManager()
        self.retriever = HybridRetriever()
        self.llm = LLM()
        logger.info("QA Pipeline initialized.")

    def process_query(self, user_query: str):
        # 1. Resolve context (e.g., "What are its symptoms?")
        resolved_query = self.context.resolve_query(user_query)
        logger.info(f"Original Query: {user_query} | Resolved: {resolved_query}")

        # 2. Extract entities and question type from resolved query
        entities = self.ner.extract_entities(resolved_query)
        question_type = self.ner.detect_question_type(resolved_query)
        
        # 3. Update conversation context
        self.context.update_context(user_query, entities, question_type)

        # 4. Retrieve evidence
        retrieved_docs = self.retriever.retrieve(resolved_query, entities, question_type)

        # 5. Generate Answer
        answer = self.llm.generate_answer(resolved_query, retrieved_docs)
        
        # 6. Update context with bot response
        self.context.add_bot_response(answer)
        
        return {
            "resolved_query": resolved_query,
            "entities": entities,
            "question_type": question_type,
            "evidence": retrieved_docs,
            "answer": answer
        }
