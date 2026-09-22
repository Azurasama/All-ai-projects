import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

# Configure page
st.set_page_config(page_title="arXiv CS Domain Expert", page_icon="🔬", layout="wide")

# Add project root to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from config.settings import CS_CATEGORIES
from retrieval.hybrid_search import get_hybrid_searcher
from chatbot.rag_pipeline import RAGPipeline
from chatbot.context_manager import ContextManager
from chatbot.query_classifier import QueryClassifier
from chatbot.response_validator import ResponseValidator
from research.summarizer import Summarizer
from research.concept_extractor import ConceptExtractor
from research.comparison import PaperComparer
from visualization.concept_graph import ConceptGraphVisualizer
from visualization.research_timeline import TimelineVisualizer

# Initialize state
if 'context_mgr' not in st.session_state:
    st.session_state.context_mgr = ContextManager()
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = RAGPipeline()
if 'query_classifier' not in st.session_state:
    st.session_state.query_classifier = QueryClassifier()
if 'searcher' not in st.session_state:
    st.session_state.searcher = get_hybrid_searcher()
if 'selected_papers_for_comparison' not in st.session_state:
    st.session_state.selected_papers_for_comparison = []

# Title
st.title("🔬 arXiv Computer Science Research Assistant")
st.markdown("An expert system powered by local RAG over the arXiv dataset.")

# Sidebar examples
with st.sidebar:
    st.header("Example Queries")
    st.markdown("**Paper Search**")
    st.code("Find recent papers about retrieval augmented generation.")
    st.markdown("**Concept Explanation**")
    st.code("Explain self-attention.")
    st.markdown("**Follow-Up**")
    st.code("What dataset did they use?")
    if st.button("Clear Conversation Context"):
        st.session_state.context_mgr.clear()
        st.success("Context cleared!")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "💬 Chat", 
    "🔍 Paper Search", 
    "📄 Paper Analysis", 
    "⚖️ Compare Papers", 
    "🌐 Concept Explorer", 
    "📈 Research Trends"
])

# --- TAB 1: Chat ---
with tab1:
    st.header("Chat with the Research Assistant")
    
    # Display chat history
    for msg in st.session_state.context_mgr.conversation_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    query = st.chat_input("Ask a research question...")
    
    if query:
        # Add user msg
        st.session_state.context_mgr.add_user_message(query)
        with st.chat_message("user"):
            st.write(query)
            
        with st.chat_message("assistant"):
            with st.spinner("Classifying and researching..."):
                # Classify
                intent = st.session_state.query_classifier.classify(query)
                st.caption(f"Detected Intent: {intent}")
                
                # Execute RAG
                raw_answer, papers = st.session_state.rag_pipeline.answer_query(query, st.session_state.context_mgr)
                
                # Validate
                is_valid, final_answer = ResponseValidator.validate(raw_answer, papers)
                
                st.write(final_answer)
                st.session_state.context_mgr.add_assistant_message(final_answer)
                
                if papers:
                    with st.expander(f"Sources ({len(papers)})"):
                        for i, p in enumerate(papers):
                            meta = p.get('metadata', {})
                            score = p.get('score', 0)
                            st.markdown(f"**{i+1}. {meta.get('title', 'Unknown Title')}** (Score: {score})")
                            st.markdown(f"*Authors: {meta.get('authors', 'Unknown')} | Categories: {meta.get('categories', '')}*")
                            st.markdown(f"*ID: {p.get('id', '')}*")

# --- TAB 2: Paper Search ---
with tab2:
    st.header("Search Papers")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("Enter keywords (e.g., 'graph neural networks')")
    with col2:
        top_k_ui = st.number_input("Results", min_value=1, max_value=50, value=10)
        
    st.subheader("Filters")
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        category_options = list(CS_CATEGORIES.keys())
        selected_cats = st.multiselect("Category", category_options, format_func=lambda x: f"{x} - {CS_CATEGORIES[x]}")
    with f_col2:
        year_filter = st.text_input("Year (e.g., 2023)")
    with f_col3:
        author_filter = st.text_input("Author")
        
    if st.button("Search", type="primary"):
        with st.spinner("Searching..."):
            filters = {}
            if selected_cats: filters["categories"] = selected_cats
            if year_filter: filters["year"] = year_filter
            if author_filter: filters["author"] = author_filter
            
            results = st.session_state.searcher.search(search_query, top_k=top_k_ui, filters=filters)
            
            if results:
                st.success(f"Found {len(results)} papers.")
                for res in results:
                    meta = res['metadata']
                    with st.container():
                        st.markdown(f"### {meta.get('title')}")
                        st.markdown(f"**Authors:** {meta.get('authors')} | **Date:** {meta.get('update_date')} | **Score:** {res['score']}")
                        st.caption(f"**ID:** {res['id']} | **Categories:** {meta.get('categories')}")
                        with st.expander("Abstract"):
                            st.write(res['document'])
                        st.divider()
            else:
                st.warning("No papers found matching the criteria.")

# --- TAB 3: Paper Analysis ---
with tab3:
    st.header("Paper Analysis & Summarization")
    st.markdown("Search for a paper in Tab 2 and enter its ID here to analyze it.")
    
    analysis_id = st.text_input("Enter arXiv Paper ID (e.g., '1706.03762')")
    summary_level = st.selectbox("Summary Level", ["Standard Summary", "Quick Summary", "Detailed Summary", "Beginner Explanation"])
    
    if st.button("Analyze Paper") and analysis_id:
        with st.spinner("Fetching and analyzing..."):
            vector_store = st.session_state.searcher.vector_store
            paper = vector_store.get_paper_by_id(analysis_id)
            
            if paper:
                meta = paper['metadata']
                st.subheader(meta.get('title'))
                
                summarizer = Summarizer()
                summary = summarizer.summarize(paper, level=summary_level)
                
                st.markdown("### Summary")
                st.write(summary)
                
                st.markdown("### Add to Comparison")
                if st.button("Select for Comparison"):
                    if paper not in st.session_state.selected_papers_for_comparison:
                        st.session_state.selected_papers_for_comparison.append(paper)
                        st.success("Added to comparison list!")
            else:
                st.error("Paper not found in index.")

# --- TAB 4: Compare Papers ---
with tab4:
    st.header("Compare Research Papers")
    st.write(f"Currently selected papers: {len(st.session_state.selected_papers_for_comparison)}")
    
    for i, p in enumerate(st.session_state.selected_papers_for_comparison):
        st.markdown(f"- {p['metadata'].get('title')} ({p['id']})")
        
    if st.button("Clear Selection"):
        st.session_state.selected_papers_for_comparison = []
        st.rerun()
        
    if len(st.session_state.selected_papers_for_comparison) >= 2:
        if st.button("Generate Comparison", type="primary"):
            with st.spinner("Analyzing and comparing..."):
                comparer = PaperComparer()
                comp_result = comparer.compare_papers(st.session_state.selected_papers_for_comparison)
                st.markdown("### Comparison Results")
                st.write(comp_result)
    else:
        st.info("Select at least 2 papers from the Paper Analysis tab to compare them.")

# --- TAB 5: Concept Explorer ---
with tab5:
    st.header("Concept Explorer")
    st.markdown("Extract and visualize technical concepts from a research topic.")
    
    topic_query = st.text_input("Enter a research topic (e.g., 'Large Language Models')")
    
    if st.button("Visualize Concepts") and topic_query:
        with st.spinner("Retrieving papers and extracting concepts..."):
            # 1. Get papers
            papers = st.session_state.searcher.search(topic_query, top_k=5)
            
            if papers:
                # 2. Extract concepts
                extractor = ConceptExtractor()
                all_concepts = {"algorithms": [], "models": [], "datasets": [], "metrics": []}
                
                for p in papers:
                    extracted = extractor.extract_concepts(p)
                    for k in all_concepts.keys():
                        if k in extracted and isinstance(extracted[k], list):
                            all_concepts[k].extend(extracted[k])
                            
                # Deduplicate
                for k in all_concepts.keys():
                    all_concepts[k] = list(set([str(x) for x in all_concepts[k] if x]))
                    
                # 3. Render graph
                st.subheader(f"Concept Graph for: {topic_query}")
                html_graph = ConceptGraphVisualizer.generate_html(all_concepts, root_topic=topic_query)
                components.html(html_graph, height=550)
            else:
                st.warning("No papers found for this topic.")

# --- TAB 6: Research Trends ---
with tab6:
    st.header("Research Trends (Timeline)")
    st.markdown("Visualize the number of retrieved papers over time.")
    
    trend_query = st.text_input("Enter a topic to analyze trends (e.g., 'Transformers')", key="trend_q")
    
    if st.button("Show Timeline") and trend_query:
        with st.spinner("Retrieving history..."):
            # Retrieve a larger batch for timeline
            papers = st.session_state.searcher.search(trend_query, top_k=50)
            if papers:
                fig = TimelineVisualizer.create_timeline(papers)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Could not parse dates from the retrieved papers.")
            else:
                st.warning("No papers found.")
