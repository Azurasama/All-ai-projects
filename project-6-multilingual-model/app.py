import streamlit as st
import uuid
from chatbot.conversation_manager import ConversationManager
from dotenv import load_dotenv

# Load environment variables early
load_dotenv()

st.set_page_config(
    page_title="Multilingual AI Chatbot",
    page_icon="🌍",
    layout="wide"
)

# Initialize Conversation Manager in session state
if "conv_manager" not in st.session_state:
    try:
        st.session_state.conv_manager = ConversationManager()
    except Exception as e:
        st.error(f"Failed to initialize models. Did you set the correct API key in .env?\nError: {e}")
        st.stop()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Layout
st.title("🌍 Multilingual Context-Aware AI Chatbot")
st.markdown("Supports **English, Hindi, Marathi, and Spanish** with Cross-Lingual Context Retention.")

# Sidebar for controls and demonstration info
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("Start New Conversation"):
        st.session_state.conv_manager.clear_conversation(st.session_state.session_id)
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    st.header("🧪 Evaluation Test Cases")
    st.info("""
    **Try these scenarios:**
    1. **Language Switching**: Ask "What is a black hole?" -> Then "इसे हिंदी में समझाओ।" -> Then "And in Marathi?"
    2. **Context Retention**: Ask "Who is the CEO of Google?" -> "त्यांचे वय काय आहे?" (What is his age? - Marathi)
    3. **Code-Mixing**: "Machine learning kaise kaam karta hai in simple words?"
    """)
    
    st.markdown("---")
    st.header("🧠 Engine Info")
    st.markdown("- **LLM**: Groq (Llama-3.1-8B)")
    st.markdown("- **Embeddings**: Sentence-Transformers (MiniLM-L12-v2)")
    st.markdown("- **Detection**: Lingua")

# Main Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "meta" in message:
            # Display metadata as a small caption
            meta_str = f"🌐 Language: {message['meta']['detected_language']}"
            if message['meta'].get('semantic_context_used'):
                meta_str += " | 📚 Vector Context Retrieved"
            st.caption(meta_str)

if prompt := st.chat_input("Type your message here (EN, HI, MR, ES)..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Process message
    with st.spinner("Thinking..."):
        try:
            result = st.session_state.conv_manager.process_message(st.session_state.session_id, prompt)
            
            # Save user message to state
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "meta": result
            })
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(result["reply"])
                
            # Save assistant message to state
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["reply"]
            })
            
            st.rerun()
        except Exception as e:
            st.error(f"Error processing message: {str(e)}")
