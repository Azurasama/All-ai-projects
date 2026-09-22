import streamlit as st
from chatbot.conversation_manager import ConversationManager

st.set_page_config(page_title="Sentiment-Aware Support Chatbot", layout="wide")

# Initialize Chatbot Manager in Session State to keep context
if "chatbot_manager" not in st.session_state:
    st.session_state.chatbot_manager = ConversationManager()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🤝 Sentiment-Aware AI Support")
st.markdown("A customer support chatbot that understands what you want, and how you feel.")

# Sidebar for Debug Info
with st.sidebar:
    st.header("🔍 Debug & Analytics Panel")
    st.markdown("View real-time sentiment and extraction data.")
    
    debug_container = st.empty()

# Main Chat Interface
chat_container = st.container()

with chat_container:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

user_input = st.chat_input("Type your message here...")

if user_input:
    # 1. Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # 2. Process via chatbot manager (which updates its own context)
    with st.spinner("Analyzing intent and sentiment..."):
        response = st.session_state.chatbot_manager.process_message(user_input)
        
    # 3. Show assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # 4. Update Debug Panel
    debug_info = st.session_state.chatbot_manager.last_debug_info
    
    with debug_container.container():
        st.subheader("Last Message Analysis")
        
        st.write(f"**Sentiment:** {debug_info.get('sentiment', 'N/A').capitalize()} ({debug_info.get('sentiment_confidence', 0)} confidence)")
        if debug_info.get('emotion') and debug_info.get('emotion') != 'neutral':
            st.write(f"**Emotion:** {debug_info.get('emotion').capitalize()} ({debug_info.get('emotion_confidence', 0)} confidence)")
            
        st.write(f"**Intent:** {debug_info.get('intent', 'N/A')}")
        st.write(f"**Urgency:** {debug_info.get('urgency', 'N/A')}")
        st.write(f"**Sentiment Trend:** {debug_info.get('trend', 'N/A')}")
        
        aspects = debug_info.get('aspects', [])
        if aspects:
            st.write("**Aspects Detected:**")
            for asp in aspects:
                st.write(f"- {asp.get('aspect')}: {asp.get('sentiment')}")
                
        with st.expander("Response Validation & Strategy"):
            st.write(f"**Validation Passed:** {debug_info.get('validation_passed')}")
            if not debug_info.get('validation_passed'):
                st.write(f"**Reason:** {debug_info.get('validation_reason')}")
            st.text_area("Strategy Used", debug_info.get('strategy_used', ''), height=150, disabled=True)
