"""
RAG Pipeline Module

Complete pipeline for Retrieval-Augmented Generation system.
Orchestrates PDF loading, chunking, embedding, retrieval, and answer generation.
"""

from typing import List, Dict, Optional
import logging
from pathlib import Path

from src.data.pdf_loader import PDFLoader
from src.data.chunker import TextChunker
from src.models.embeddings import EmbeddingModel
from src.models.retriever import FAISSRetriever
from src.models.llm import LLMHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline for research paper Q&A.

    This class orchestrates all components:
    - PDF loading and text extraction
    - Intelligent text chunking
    - Embedding generation
    - Vector similarity search
    - LLM-based answer generation

    Example:
        >>> pipeline = RAGPipeline()
        >>> pipeline.load_papers("data/papers")
        >>> answer = pipeline.query("What is the main contribution?")
        >>> print(answer['answer'])
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_custom_embeddings: bool = False,
        custom_embedding_path: str = None,
        llm_model: Optional[str] = None,
        api_key: Optional[str] = None,
        use_custom_llm: bool = False,
        custom_llm_path: str = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 5
    ):
        """
        Initialize RAG pipeline.

        Args:
            embedding_model: HuggingFace embedding model name
            llm_model: Gemini model name (defaults to the GEMINI_MODEL env var)
            api_key: Gemini API key for this pipeline (falls back to GOOGLE_API_KEY)
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            top_k: Number of chunks to retrieve
        """
        logger.info("Initializing RAG Pipeline...")

        # Configuration
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        # Initialize components
        self.pdf_loader = PDFLoader()
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Use custom or generic embeddings
        if use_custom_embeddings and custom_embedding_path:
            logger.info(f"Using custom embeddings from: {custom_embedding_path}")
            self.embedder = EmbeddingModel(
                use_custom_model=True,
                custom_model_path=custom_embedding_path
            )
        else:
            self.embedder = EmbeddingModel(model_name=embedding_model)

        # Use custom or API-based LLM
        if use_custom_llm and custom_llm_path:
            logger.info(f"Using custom LLM from: {custom_llm_path}")
            self.llm = LLMHandler(
                api_key=api_key,
                use_custom_model=True,
                custom_model_path=custom_llm_path
            )
        else:
            self.llm = LLMHandler(model_name=llm_model, api_key=api_key)

        # State
        self.retriever: Optional[FAISSRetriever] = None
        self.documents: List[Dict] = []
        self.chunks: List[Dict] = []

        logger.info("✓ Pipeline initialized successfully")

    def load_papers(self, papers_dir: str) -> int:
        """
        Load research papers from directory.

        Args:
            papers_dir: Path to directory containing PDFs

        Returns:
            Number of papers loaded
        """
        logger.info(f"Loading papers from {papers_dir}...")

        self.documents = self.pdf_loader.load_directory(papers_dir)

        if not self.documents:
            logger.warning(f"No PDF files found in {papers_dir}")
            return 0

        logger.info(f"✓ Loaded {len(self.documents)} papers")
        return len(self.documents)

    def build_index(self) -> int:
        """
        Build FAISS index from loaded papers.

        Returns:
            Number of chunks indexed
        """
        if not self.documents:
            raise ValueError("No documents loaded. Call load_papers() first.")

        logger.info("Building FAISS index...")

        # Step 1: Chunk documents
        logger.info("  1/3 Chunking documents...")
        self.chunks = self.chunker.chunk_documents(self.documents)
        logger.info(f"      Created {len(self.chunks)} chunks")

        # Step 2: Generate embeddings
        logger.info("  2/3 Generating embeddings...")
        embeddings = self.embedder.encode_chunks(self.chunks, show_progress=True)
        logger.info(f"      Generated {len(embeddings)} embeddings")

        # Step 3: Build FAISS index
        logger.info("  3/3 Building FAISS index...")
        self.retriever = FAISSRetriever(
            embedding_dimension=embeddings.shape[1],
            index_type="Flat"
        )
        self.retriever.add_documents(self.chunks, embeddings)

        logger.info(f"✓ Index built with {len(self.chunks)} chunks")
        return len(self.chunks)

    def save_index(self, save_dir: str):
        """
        Save FAISS index to disk.

        Args:
            save_dir: Directory to save index
        """
        if self.retriever is None:
            raise ValueError("No index to save. Call build_index() first.")

        logger.info(f"Saving index to {save_dir}...")
        self.retriever.save(save_dir)
        logger.info(f"✓ Index saved successfully")

    def load_index(self, index_dir: str):
        """
        Load existing FAISS index from disk.

        Args:
            index_dir: Directory containing saved index
        """
        logger.info(f"Loading index from {index_dir}...")
        self.retriever = FAISSRetriever.load(index_dir)

        # Extract chunks from loaded retriever
        self.chunks = self.retriever.chunks

        logger.info(f"✓ Index loaded with {len(self.chunks)} chunks")

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_context: bool = False
    ) -> Dict:
        """
        Query the RAG system.

        Args:
            question: User question
            top_k: Number of chunks to retrieve (default: self.top_k)
            return_context: Include retrieved context in response

        Returns:
            Dictionary with answer, sources, and optionally context

        Example:
            >>> result = pipeline.query("What sensors were used?")
            >>> print(result['answer'])
            >>> for source in result['sources']:
            ...     print(f"  - {source['filename']}, Page {source['page']}")
        """
        if self.retriever is None:
            raise ValueError("No index loaded. Call build_index() or load_index() first.")

        k = top_k or self.top_k

        # Step 1: Retrieve relevant chunks
        logger.info(f"Searching for: '{question}'")
        query_embedding = self.embedder.encode(question)
        retrieved_chunks = self.retriever.search(query_embedding, k=k)

        logger.info(f"✓ Retrieved {len(retrieved_chunks)} relevant chunks")

        # Step 2: Generate answer using LLM
        logger.info("Generating answer...")
        answer_result = self.llm.answer_question(
            question=question,
            context_chunks=retrieved_chunks,
            include_sources=True
        )

        logger.info("✓ Answer generated")

        # Build response
        response = {
            'question': question,
            'answer': answer_result['answer'],
            'sources': answer_result['sources'],
            'num_chunks_used': len(retrieved_chunks)
        }

        if return_context:
            response['context'] = answer_result['context_used']
            response['retrieved_chunks'] = retrieved_chunks

        return response

    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        stats = {
            'num_documents': len(self.documents),
            'num_chunks': len(self.chunks),
            'embedding_model': self.embedder.model_name,
            'embedding_dimension': self.embedder.get_embedding_dimension(),
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
        }

        if self.retriever:
            stats.update(self.retriever.get_stats())

        return stats


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = RAGPipeline()

    # Option 1: Build new index
    print("\n=== Building New Index ===")
    pipeline.load_papers("data/papers")
    pipeline.build_index()
    pipeline.save_index("data/faiss_index")

    # Option 2: Load existing index
    # pipeline.load_index("data/faiss_index")

    # Query the system
    print("\n=== Querying System ===")
    questions = [
        "What is the main contribution of this research?",
        "What sensors were used in the experiment?",
        "What were the results?"
    ]

    for question in questions:
        print(f"\nQ: {question}")
        result = pipeline.query(question)
        print(f"A: {result['answer']}\n")
        print("Sources:")
        for source in result['sources']:
            print(f"  - {source['filename']}, Page {source['page']}")
        print("-" * 80)

    # Show statistics
    print("\n=== Pipeline Statistics ===")
    stats = pipeline.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
