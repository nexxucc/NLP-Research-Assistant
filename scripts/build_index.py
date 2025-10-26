"""
Build FAISS Index from Research Papers

This script processes PDF research papers and builds a searchable FAISS index.

Usage:
    python scripts/build_index.py [--papers-dir data/papers] [--output-dir data/faiss_index]

Features:
    - Loads PDFs from directory
    - Chunks text intelligently
    - Generates embeddings
    - Builds and saves FAISS index
    - Progress tracking and logging
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import logging
from datetime import datetime
from tqdm import tqdm

from src.data.pdf_loader import PDFLoader
from src.data.chunker import TextChunker
from src.models.embeddings import EmbeddingModel
from src.models.retriever import FAISSRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('build_index.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main(papers_dir: str, output_dir: str, chunk_size: int = 512, chunk_overlap: int = 50):
    """
    Build FAISS index from research papers.

    Args:
        papers_dir: Directory containing PDF files
        output_dir: Directory to save FAISS index
        chunk_size: Size of text chunks in characters
        chunk_overlap: Overlap between chunks
    """
    start_time = datetime.now()
    logger.info("=" * 80)
    logger.info("Starting FAISS Index Build Process")
    logger.info("=" * 80)

    # Validate directories
    papers_path = Path(papers_dir)
    if not papers_path.exists():
        logger.error(f"Papers directory not found: {papers_dir}")
        logger.info("Creating directory...")
        papers_path.mkdir(parents=True, exist_ok=True)
        logger.warning(f"Please add PDF files to {papers_dir} and run again")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Load PDFs
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Loading PDF Documents")
    logger.info("=" * 80)

    pdf_loader = PDFLoader()
    documents = pdf_loader.load_directory(papers_dir)

    if not documents:
        logger.error("No PDF documents found!")
        logger.warning(f"Please add PDF files to {papers_dir}")
        return

    logger.info(f"✓ Successfully loaded {len(documents)} documents")
    for doc in documents:
        logger.info(f"  - {doc['metadata']['filename']} ({doc['metadata']['page_count']} pages)")

    # Step 2: Chunk documents
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Chunking Documents")
    logger.info("=" * 80)
    logger.info(f"Chunk size: {chunk_size} characters")
    logger.info(f"Chunk overlap: {chunk_overlap} characters")

    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_documents(documents)

    logger.info(f"✓ Created {len(chunks)} chunks from {len(documents)} documents")
    logger.info(f"  Average chunks per document: {len(chunks) / len(documents):.1f}")

    # Step 3: Generate embeddings
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Generating Embeddings")
    logger.info("=" * 80)

    embedder = EmbeddingModel()
    logger.info(f"Using model: {embedder.model_name}")
    logger.info(f"Embedding dimension: {embedder.get_embedding_dimension()}")

    embeddings = embedder.encode_chunks(chunks, show_progress=True)

    logger.info(f"✓ Generated {len(embeddings)} embeddings")
    logger.info(f"  Embedding shape: {embeddings.shape}")

    # Step 4: Build FAISS index
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Building FAISS Index")
    logger.info("=" * 80)

    retriever = FAISSRetriever(
        embedding_dimension=embeddings.shape[1],
        index_type="Flat"  # Use Flat for exact search
    )

    retriever.add_documents(chunks, embeddings)

    stats = retriever.get_stats()
    logger.info(f"✓ FAISS index built successfully")
    logger.info(f"  Index type: {stats['index_type']}")
    logger.info(f"  Total documents: {stats['num_documents']}")
    logger.info(f"  Embedding dimension: {stats['embedding_dimension']}")

    # Step 5: Save index
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Saving Index to Disk")
    logger.info("=" * 80)

    retriever.save(output_dir)

    logger.info(f"✓ Index saved to: {output_dir}")
    logger.info(f"  Files created:")
    for file in output_path.iterdir():
        logger.info(f"    - {file.name} ({file.stat().st_size / 1024:.1f} KB)")

    # Summary
    elapsed_time = datetime.now() - start_time
    logger.info("\n" + "=" * 80)
    logger.info("BUILD COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Total time: {elapsed_time}")
    logger.info(f"Documents processed: {len(documents)}")
    logger.info(f"Chunks created: {len(chunks)}")
    logger.info(f"Embeddings generated: {len(embeddings)}")
    logger.info(f"Index location: {output_dir}")
    logger.info("\n✓ Your FAISS index is ready to use!")
    logger.info(f"✓ Run the Streamlit app: streamlit run src/ui/streamlit_app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build FAISS index from research papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build index from default papers directory
    python scripts/build_index.py

    # Specify custom directories
    python scripts/build_index.py --papers-dir my_papers --output-dir my_index

    # Adjust chunk size
    python scripts/build_index.py --chunk-size 1024 --chunk-overlap 100
        """
    )

    parser.add_argument(
        '--papers-dir',
        type=str,
        default='data/papers',
        help='Directory containing PDF files (default: data/papers)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/faiss_index',
        help='Directory to save FAISS index (default: data/faiss_index)'
    )

    parser.add_argument(
        '--chunk-size',
        type=int,
        default=512,
        help='Size of text chunks in characters (default: 512)'
    )

    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=50,
        help='Overlap between chunks in characters (default: 50)'
    )

    args = parser.parse_args()

    main(
        papers_dir=args.papers_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
