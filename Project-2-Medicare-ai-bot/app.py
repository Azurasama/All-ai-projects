import streamlit as st
import sys
from pathlib import Path
import json

# Setup paths
sys.path.append(str(Path(__file__).resolve().parent))
from chatbot.qa_pipeline import MedicalQAPipeline
from config.settings import TOP_K_SEMANTIC, RELEVANCE_THRESHOLD

# Configure page
st.set_page_config(page_title="Medical Q&A Assistant", page_icon="🏥", layout="wide")

@st.cache_resource
def load_pipeline():
    return MedicalQAPipeline()

pipeline = load_pipeline()

st.title("🏥 Medical Q&A Assistant")
st.markdown("""
> [!WARNING]
> **Disclaimer:** This chatbot provides medical information from the MedQuAD knowledge base and is **not** a substitute for a qualified healthcare professional. Do not use this tool for medical diagnosis, personalized treatment, or emergency assessment. If you are experiencing a medical emergency, please seek immediate professional help.
""")

# Sidebar settings and examples
with st.sidebar:
    st.header("Settings & Examples")
    st.write("This chatbot uses Retrieval-Augmented Generation (RAG) to answer questions based on the MedQuAD dataset.")
    
    st.subheader("Example Queries")
    st.markdown("""
    - What are the symptoms of asthma?
    - What causes diabetes?
    - How is Parkinson's disease diagnosed?
    - What are the treatments for hypertension?
    - What is the difference between Type 1 and Type 2 diabetes?
    """)
    
    if st.button("Clear Conversation History"):
        st.session_state.messages = []
        pipeline.context.history = []
        pipeline.context.current_topic = None
        pipeline.context.recent_entities = {}
        st.success("History cleared!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "evidence" in message and message["evidence"]:
            with st.expander("View retrieved sources & entities"):
                st.write("**Resolved Query:**", message.get("resolved_query"))
                st.write("**Detected Entities:**", message.get("entities"))
                st.write("**Question Type:**", message.get("question_type"))
                st.write("---")
                for i, ev in enumerate(message["evidence"]):
                    st.markdown(f"**Source {i+1}** (Relevance: {ev['final_score']:.2f})")
                    st.markdown(f"*Focus:* {ev['metadata'].get('focus')} | *Type:* {ev['metadata'].get('question_type')} | *Source:* {ev['metadata'].get('source')}")
                    st.markdown(f"*Question:* {ev['metadata'].get('question')}")
                    st.markdown(f"*Answer:* {ev['metadata'].get('answer')[:300]}...")

# React to user input
if prompt := st.chat_input("Ask a medical question..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    
    with st.spinner("Processing query..."):
        try:
            result = pipeline.process_query(prompt)
            answer = result["answer"]
            evidence = result["evidence"]
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(answer)
                with st.expander("View retrieved sources & entities"):
                    st.write("**Resolved Query:**", result["resolved_query"])
                    st.write("**Detected Entities:**", result["entities"])
                    st.write("**Question Type:**", result["question_type"])
                    st.write("---")
                    for i, ev in enumerate(evidence):
                        st.markdown(f"**Source {i+1}** (Relevance: {ev['final_score']:.2f})")
                        st.markdown(f"*Focus:* {ev['metadata'].get('focus')} | *Type:* {ev['metadata'].get('question_type')} | *Source:* {ev['metadata'].get('source')}")
                        st.markdown(f"*Question:* {ev['metadata'].get('question')}")
                        st.markdown(f"*Answer:* {ev['metadata'].get('answer')[:300]}...")
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "evidence": evidence,
                "resolved_query": result["resolved_query"],
                "entities": result["entities"],
                "question_type": result["question_type"]
            })
        except Exception as e:
            st.error(f"An error occurred: {e}")
