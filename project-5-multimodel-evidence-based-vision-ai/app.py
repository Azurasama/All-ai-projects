import streamlit as st
import uuid

# Set up page config first
st.set_page_config(page_title="Multimodal AI Assistant", layout="wide")

from models.vision_model import VisionModel
from models.llm import LLMModel
from models.ocr import OCRModel
from vision.image_processor import ImageProcessor
from core.conversation_manager import ConversationManager
from core.evidence_manager import EvidenceManager
from core.reasoning_engine import ReasoningEngine
from core.response_validator import ResponseValidator
from config.settings import GEMINI_API_KEY

# Initialize session state for singletons
if "initialized" not in st.session_state:
    if not GEMINI_API_KEY:
        st.error("Please set GEMINI_API_KEY in the .env file or environment variables.")
        st.stop()
        
    st.session_state.vision_model = VisionModel()
    st.session_state.llm_model = LLMModel()
    
    # We load EasyOCR lazily or show a spinner because it takes a moment
    with st.spinner("Initializing OCR engine..."):
        st.session_state.ocr_model = OCRModel()
        
    st.session_state.image_processor = ImageProcessor()
    st.session_state.conv_manager = ConversationManager()
    st.session_state.evidence_manager = EvidenceManager()
    
    st.session_state.reasoning = ReasoningEngine(st.session_state.llm_model)
    st.session_state.validator = ResponseValidator(st.session_state.llm_model)
    
    st.session_state.initialized = True
    st.session_state.chat_history_ui = [] # For Streamlit display (role, content, images, evidence_panel)

# Main UI
st.title("Multimodal AI Assistant")

col_main, col_side = st.columns([2, 1])

with col_main:
    # Display chat history
    for msg in st.session_state.chat_history_ui:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("images"):
                for img in msg["images"]:
                    st.image(img, width=300)

    # Chat input
    user_input = st.chat_input("Ask a question...")

with col_side:
    st.subheader("Upload Images")
    uploaded_files = st.file_uploader("Choose images...", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
    
    # Sidebar for evidence inspection
    st.subheader("Evidence & Reasoning")
    if st.session_state.evidence_manager.evidence_store:
        for img_id, ev in st.session_state.evidence_manager.evidence_store.items():
            with st.expander(f"Image: {img_id[:8]}"):
                st.write("**Summary:**", ev['visual_summary'])
                st.write("**Objects:**", ", ".join(ev['objects']))
                st.write("**OCR Text:**", ev['ocr_text'])
                st.write("**Confidence:**", ev['confidence'])
                st.write("**Unclear Flag:**", ev['is_unclear'])

# Process new input
if user_input:
    # 1. Process uploaded images (if any)
    new_image_ids = []
    display_images = []
    
    if uploaded_files:
        for f in uploaded_files:
            processed = st.session_state.image_processor.process_image(f)
            if not processed["success"]:
                st.error(processed["error"])
                continue
                
            img = processed["image"]
            img_id = str(uuid.uuid4())
            new_image_ids.append(img_id)
            display_images.append(img)
            
            # Extract Evidence
            with st.spinner(f"Extracting visual evidence..."):
                ocr_text = st.session_state.ocr_model.extract_text(img)
                visual_info = st.session_state.vision_model.extract_visual_information(img)
                
                st.session_state.evidence_manager.store_evidence(
                    image_id=img_id,
                    visual_info=visual_info,
                    ocr_text=ocr_text
                )
    
    # Show user message
    st.session_state.chat_history_ui.append({
        "role": "user",
        "content": user_input,
        "images": display_images
    })
    
    # Update backend conversation state
    st.session_state.conv_manager.add_user_message(user_input, new_image_ids)
    
    # Refresh to show user message immediately
    st.rerun()

# Handle Assistant response if last message was user
if st.session_state.chat_history_ui and st.session_state.chat_history_ui[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            user_query = st.session_state.chat_history_ui[-1]["content"]
            history_str = st.session_state.conv_manager.get_recent_history()
            all_known_img_ids = list(st.session_state.evidence_manager.evidence_store.keys())
            
            # 2. Reasoning Engine: Intent & Reference Resolution
            intent_analysis = st.session_state.reasoning.analyze_intent(
                user_query=user_query,
                chat_history=history_str,
                available_image_ids=all_known_img_ids
            )
            
            if intent_analysis.get("is_ambiguous"):
                clarification = intent_analysis.get("clarification_needed", "Could you please clarify your question?")
                st.markdown(clarification)
                st.session_state.chat_history_ui.append({"role": "assistant", "content": clarification})
                st.session_state.conv_manager.add_assistant_message(clarification)
                st.rerun()
                
            # Retrieve relevant evidence based on intent analysis
            ref_imgs = intent_analysis.get("referenced_images", [])
            evidence_str = st.session_state.evidence_manager.format_evidence_for_prompt(ref_imgs)
            
            # 3. Generate initial response
            initial_response = st.session_state.reasoning.generate_response(
                user_query=user_query,
                chat_history=history_str,
                evidence_text=evidence_str
            )
            
            # 4. Validate Response
            validation = st.session_state.validator.validate(
                generated_response=initial_response,
                user_query=user_query,
                evidence_text=evidence_str
            )
            
            final_response = initial_response
            
            if not validation.get("is_supported") or validation.get("has_hallucination"):
                # Regenerate with explicit correction
                correction_prompt = f"""
                The previous response failed validation: {validation.get('correction_needed')}
                Please rewrite the response to strictly adhere to the evidence.
                
                USER QUERY: {user_query}
                EVIDENCE: {evidence_str}
                """
                final_response = st.session_state.reasoning.llm.generate_response(correction_prompt)
            
            st.markdown(final_response)
            
            # Update state
            st.session_state.chat_history_ui.append({"role": "assistant", "content": final_response})
            st.session_state.conv_manager.add_assistant_message(final_response)
            
            # Optional: Show a tiny debug/metadata block
            with st.expander("Debug Validation Metrics"):
                st.json(validation)
                st.json(intent_analysis)
