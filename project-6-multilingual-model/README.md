# Multilingual Context-Aware AI Chatbot

A sophisticated, modular conversational AI system that understands multiple languages, retains context across language switches, resolves cross-lingual coreferences, and handles code-mixed inputs (e.g., Hinglish).

## Features

- **Automatic Language Detection**: Detects English, Hindi, Marathi, and Spanish using `lingua-language-detector`.
- **Multilingual Conversation & Context**: Answers in the language of the latest query while retaining context from previous turns across different languages.
- **Code-Mixed Input**: Gracefully handles inputs containing multiple languages.
- **Cross-Lingual Semantic Retrieval**: Uses `sentence-transformers` (multilingual-e5 or paraphrase-multilingual) and ChromaDB to semantically match queries across languages (e.g. matching "What is Python?" with "Python kya hai?").
- **Open-Source LLM Integration**: Uses LangChain to orchestrate models. Optimized for inference via Groq (Llama-3), but easily extensible to local models or Hugging Face endpoints.

## System Architecture

1. **User Input** → Streamlit captures the message.
2. **Language Detection (Lingua)** → Accurately determines the source language or code-mix ratios.
3. **Semantic Embedding (SentenceTransformers)** → Generates a language-agnostic dense vector representation.
4. **Context Retrieval (ChromaDB)** → Pulls top-K semantically similar past turns.
5. **Prompt Assembly (LangChain)** → Assembles a prompt with recent history and semantic context.
6. **LLM Generation (Llama-3 via Groq)** → Performs cross-lingual reasoning and responds.

## Installation

1. Clone the repository and navigate to the project directory:
   ```bash
   cd multilingual-chatbot
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Environment Variables:
   Copy `.env.example` to `.env` and add your Groq API key (or other configured provider).
   ```bash
   cp .env.example .env
   ```
   *Note: Obtain a free API key from [Groq](https://console.groq.com).*

## Running the Application

Run the Streamlit application:
```bash
streamlit run app.py
```
The application will be available at `http://localhost:8501`. 

*Note: On the first run, the local embedding model (~400MB) will be downloaded from Hugging Face.*

## Evaluation Methodology

Open the UI and use the "Evaluation Test Cases" sidebar.
- Test **Context Retention**: Ask a question in English, then refer to the subject using a pronoun in Spanish or Hindi. The Vector Store and LangChain context window will resolve the entity.
- Test **Language Switching**: Switch languages mid-conversation without explicitly instructing the bot to translate.

## Limitations
- Heavily relies on the multilinguality of the underlying LLM. Llama-3 is good at Hindi/Spanish but might occasionally falter in Marathi generation compared to Qwen.
- Local embedding models run on CPU, which is fast enough for chat but could be optimized.
