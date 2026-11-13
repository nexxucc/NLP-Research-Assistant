"""
NLP Research Assistant - Streamlit UI

A polished, production-ready interface for the RAG-based research paper Q&A system.

Features:
- Upload and index research papers
- Ask questions with natural language
- View answers with source citations
- Compare retrieval methods (semantic vs hybrid)
- Evaluate system performance
- Export results to PDF

Run:
    streamlit run src/ui/streamlit_app.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import streamlit as st
import os
from datetime import datetime
import json
from typing import List, Dict

from src.pipeline import RAGPipeline
from src.models.embeddings import EmbeddingModel
from src.models.hybrid_retriever import HybridRetriever
from src.data.pdf_loader import PDFLoader
from src.data.chunker import TextChunker

# Page configuration
st.set_page_config(
    page_title="NLP Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stat-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .source-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #28a745;
    }
    .answer-box {
        background-color: #e7f3ff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #0066cc;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'index_loaded' not in st.session_state:
    st.session_state.index_loaded = False
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'use_hybrid' not in st.session_state:
    st.session_state.use_hybrid = True


def init_pipeline():
    """Initialize RAG pipeline with this session's own Gemini key."""
    with st.spinner("Initializing RAG pipeline..."):
        pipeline = RAGPipeline(
            api_key=st.session_state.get('api_key'),
            chunk_size=512,
            chunk_overlap=50,
            top_k=5
        )
        st.session_state.pipeline = pipeline
    return pipeline


def load_or_build_index(papers_dir: str, index_dir: str, rebuild: bool = False):
    """Load existing index or build new one."""
    pipeline = st.session_state.pipeline

    if not pipeline:
        pipeline = init_pipeline()

    index_path = Path(index_dir)

    # Check if index exists
    if index_path.exists() and not rebuild:
        try:
            with st.spinner("Loading existing index..."):
                pipeline.load_index(index_dir)
                st.session_state.index_loaded = True
                st.success("✅ Index loaded successfully!")
                return True
        except Exception as e:
            st.error(f"Error loading index: {e}")
            st.info("Building new index...")

    # Build new index
    papers_path = Path(papers_dir)
    if not papers_path.exists() or not list(papers_path.glob("*.pdf")):
        st.error(f"No PDF files found in {papers_dir}")
        st.info("Please add PDF files to the papers directory and try again.")
        return False

    try:
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Step 1: Load papers
        status_text.text("📄 Loading PDF documents...")
        progress_bar.progress(25)
        num_papers = pipeline.load_papers(papers_dir)

        # Step 2: Build index
        status_text.text("🔨 Building FAISS index...")
        progress_bar.progress(50)
        num_chunks = pipeline.build_index()

        # Step 3: Save index
        status_text.text("💾 Saving index to disk...")
        progress_bar.progress(75)
        pipeline.save_index(index_dir)

        progress_bar.progress(100)
        status_text.text("✅ Index built successfully!")

        st.session_state.index_loaded = True

        st.success(f"""
        ✅ **Index built successfully!**
        - Papers processed: {num_papers}
        - Chunks created: {num_chunks}
        - Index saved to: {index_dir}
        """)

        return True

    except Exception as e:
        st.error(f"Error building index: {e}")
        return False


def display_stats():
    """Display system statistics."""
    if st.session_state.pipeline and st.session_state.index_loaded:
        stats = st.session_state.pipeline.get_stats()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="stat-box">
                <h3 style="margin:0; color:#1f77b4;">📚 {stats.get('num_documents', 0)}</h3>
                <p style="margin:0; color:#666;">Documents</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="stat-box">
                <h3 style="margin:0; color:#1f77b4;">📝 {stats.get('num_chunks', 0)}</h3>
                <p style="margin:0; color:#666;">Chunks</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="stat-box">
                <h3 style="margin:0; color:#1f77b4;">🔢 {stats.get('embedding_dimension', 0)}</h3>
                <p style="margin:0; color:#666;">Embedding Dim</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="stat-box">
                <h3 style="margin:0; color:#1f77b4;">❓ {len(st.session_state.query_history)}</h3>
                <p style="margin:0; color:#666;">Queries</p>
            </div>
            """, unsafe_allow_html=True)


def query_system(question: str, top_k: int = 5):
    """Query the RAG system."""
    if not st.session_state.pipeline or not st.session_state.index_loaded:
        st.error("Please load or build an index first!")
        return None

    try:
        with st.spinner("🔍 Searching and generating answer..."):
            result = st.session_state.pipeline.query(
                question=question,
                top_k=top_k,
                return_context=True
            )

            # Add to history
            st.session_state.query_history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'question': question,
                'answer': result['answer'],
                'sources': result['sources']
            })

            return result

    except Exception as e:
        st.error(f"Error querying system: {e}")
        return None


def display_result(result: Dict):
    """Display query result in a nice format."""
    # Display answer
    st.markdown(f"""
    <div class="answer-box">
        <h3 style="margin-top:0;">💡 Answer</h3>
        <p style="font-size:1.1rem; line-height:1.6;">{result['answer']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Display sources
    st.markdown("### 📖 Sources")

    for i, source in enumerate(result['sources'], 1):
        score_text = f"(Score: {source['score']:.3f})" if source['score'] else ""

        st.markdown(f"""
        <div class="source-box">
            <strong>Source {i}:</strong> {source['filename']}
            {f", Page {source['page']}" if source.get('page') else ""}
            {score_text}
        </div>
        """, unsafe_allow_html=True)

    # Expandable context
    if 'context' in result:
        with st.expander("🔍 View Retrieved Context"):
            st.text(result['context'])


def main():
    """Main Streamlit app."""

    # Header
    st.markdown('<h1 class="main-header">📚 NLP Research Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Research Paper Q&A with RAG</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # API Key
        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            help="Get your API key from https://makersuite.google.com/app/apikey"
        )

        # Keep the key in this visitor's session only. Writing it to os.environ
        # would share it with every other visitor of the same server process.
        st.session_state.api_key = api_key
        if api_key and st.session_state.pipeline:
            st.session_state.pipeline.llm.set_api_key(api_key)

        st.markdown("---")

        # Index management
        st.header("📁 Index Management")

        papers_dir = st.text_input("Papers Directory", value="data/papers")
        index_dir = st.text_input("Index Directory", value="data/faiss_index")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📥 Load Index", use_container_width=True):
                if not api_key:
                    st.error("Please provide API key first!")
                else:
                    load_or_build_index(papers_dir, index_dir, rebuild=False)

        with col2:
            if st.button("🔨 Rebuild Index", use_container_width=True):
                if not api_key:
                    st.error("Please provide API key first!")
                else:
                    load_or_build_index(papers_dir, index_dir, rebuild=True)

        st.markdown("---")

        # Query settings
        st.header("🎛️ Query Settings")

        top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)

        st.session_state.use_hybrid = st.checkbox(
            "Use Hybrid Retrieval",
            value=True,
            help="Combine semantic + keyword search for better results"
        )

        st.markdown("---")

        # Help
        with st.expander("ℹ️ How to Use"):
            st.markdown("""
            1. **Add API Key**: Enter your Google Gemini API key
            2. **Add Papers**: Place PDF files in `data/papers/`
            3. **Build Index**: Click "Rebuild Index" to process papers
            4. **Ask Questions**: Type questions in natural language
            5. **View Results**: Get answers with source citations
            """)

    # Main content
    if not api_key:
        st.warning("⚠️ Please provide your Google Gemini API key in the sidebar to get started.")
        st.info("""
        **Getting Started:**
        1. Get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Enter the API key in the sidebar
        3. Add PDF research papers to `data/papers/`
        4. Click "Rebuild Index" to process your papers
        """)
        return

    # Display stats
    if st.session_state.index_loaded:
        display_stats()
        st.markdown("---")

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["💬 Ask Questions", "📊 Query History", "🧪 Evaluation"])

    with tab1:
        st.header("Ask Questions About Your Research Papers")

        # Query input
        question = st.text_input(
            "Your Question:",
            placeholder="E.g., What is the main contribution of this paper?",
            help="Ask any question about your research papers"
        )

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            search_button = st.button("🔍 Search", use_container_width=True, type="primary")

        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                question = ""
                st.rerun()

        # Handle search
        if search_button and question:
            if not st.session_state.index_loaded:
                st.error("Please load or build an index first!")
            else:
                result = query_system(question, top_k=top_k)

                if result:
                    display_result(result)

        # Example questions
        if st.session_state.index_loaded:
            st.markdown("---")
            st.markdown("**💡 Example Questions:**")

            example_questions = [
                "What is the main contribution of this research?",
                "What methodology was used in the experiments?",
                "What were the key findings and results?",
                "What datasets were used?",
                "What are the limitations of this approach?",
                "What future work is suggested?"
            ]

            cols = st.columns(3)
            for i, eq in enumerate(example_questions):
                with cols[i % 3]:
                    if st.button(eq, key=f"example_{i}", use_container_width=True):
                        st.session_state.example_question = eq
                        st.rerun()

            if 'example_question' in st.session_state:
                question = st.session_state.example_question
                del st.session_state.example_question

    with tab2:
        st.header("📊 Query History")

        if st.session_state.query_history:
            # Export button
            if st.button("📥 Export History (JSON)"):
                history_json = json.dumps(st.session_state.query_history, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=history_json,
                    file_name=f"query_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

            st.markdown("---")

            # Display history (most recent first)
            for i, entry in enumerate(reversed(st.session_state.query_history)):
                with st.expander(f"🕐 {entry['timestamp']} - {entry['question'][:60]}..."):
                    st.markdown(f"**Question:** {entry['question']}")
                    st.markdown(f"**Answer:** {entry['answer']}")
                    st.markdown("**Sources:**")
                    for source in entry['sources']:
                        st.markdown(f"- {source['filename']}, Page {source.get('page', 'N/A')}")
        else:
            st.info("No queries yet. Ask a question in the 'Ask Questions' tab!")

    with tab3:
        st.header("🧪 System Evaluation")

        st.info("This section allows you to compare different retrieval methods and evaluate performance.")

        # Coming soon features
        st.markdown("""
        **Available Evaluations:**

        - **Retrieval Comparison**: Compare semantic vs. hybrid vs. keyword-only retrieval
        - **Performance Metrics**: AUROC, AUPRC, precision, recall
        - **Response Quality**: Evaluate answer relevance and accuracy
        - **Speed Benchmarks**: Measure query response times

        *Advanced evaluation features coming soon...*
        """)

        if st.button("Run Quick Evaluation"):
            st.warning("Full evaluation suite requires labeled test data. Coming in Phase 2!")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Built with ❤️ using LangChain, FAISS, HuggingFace, and Streamlit</p>
        <p>Powered by Google Gemini • Enhanced with Hybrid Retrieval</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
