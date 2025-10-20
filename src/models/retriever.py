"""
Retrieval Module

Implements FAISS-based vector store for semantic search.
Handles indexing, searching, and persistence of embeddings.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import faiss
import pickle
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAISSRetriever:
    """
    FAISS-based vector store for semantic retrieval.

    FAISS (Facebook AI Similarity Search) enables efficient similarity search
    over large collections of vectors (embeddings). It uses approximate nearest
    neighbor search for speed.

    How it works:
    1. Store all chunk embeddings in FAISS index
    2. Convert query to embedding
    3. Find k most similar chunk embeddings
    4. Return corresponding chunks
    """

    def __init__(
        self,
        embedding_dimension: int,
        index_type: str = "Flat"
    ):
        """
        Initialize FAISS retriever.

        Args:
            embedding_dimension: Dimension of embeddings (e.g., 384 for MiniLM)
            index_type: Type of FAISS index
                - "Flat": Exact search, slower but perfect recall
                - "IVF": Approximate search, faster for large datasets

        Example:
            >>> retriever = FAISSRetriever(embedding_dimension=384)
            >>> retriever.add_documents(chunks, embeddings)
            >>> results = retriever.search(query_embedding, k=5)
        """
        self.embedding_dimension = embedding_dimension
        self.index_type = index_type

        # Initialize FAISS index
        if index_type == "Flat":
            # L2 distance (Euclidean) - works well with normalized embeddings
            self.index = faiss.IndexFlatL2(embedding_dimension)
        elif index_type == "IVF":
            # For larger datasets (>10k chunks), use IVF for speed
            quantizer = faiss.IndexFlatL2(embedding_dimension)
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dimension, 100)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        # Store document chunks for retrieval
        self.chunks = []
        self.is_trained = False

        logger.info(f"Initialized FAISS {index_type} index with dimension {embedding_dimension}")

    def add_documents(
        self,
        chunks: List[Dict],
        embeddings: np.ndarray
    ):
        """
        Add documents (chunks) and their embeddings to the index.

        Args:
            chunks: List of chunk dictionaries with text and metadata
            embeddings: numpy array of embeddings (num_chunks, embedding_dim)

        Example:
            >>> chunks = [{'text': 'chunk1', 'metadata': {}}, ...]
            >>> embeddings = embedder.encode_chunks(chunks)
            >>> retriever.add_documents(chunks, embeddings)
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        if embeddings.shape[1] != self.embedding_dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} doesn't match "
                f"index dimension {self.embedding_dimension}"
            )

        # Convert to float32 (FAISS requirement)
        embeddings = embeddings.astype('float32')

        # Train index if needed (for IVF)
        if self.index_type == "IVF" and not self.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)
            self.is_trained = True

        # Add embeddings to index
        self.index.add(embeddings)

        # Store chunks
        self.chunks.extend(chunks)

        logger.info(
            f"Added {len(chunks)} documents. "
            f"Total documents in index: {self.index.ntotal}"
        )

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        return_scores: bool = True
    ) -> List[Dict]:
        """
        Search for k most similar documents to query.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            return_scores: Include similarity scores in results

        Returns:
            List of dicts containing:
                - chunk: The chunk dict
                - score: Similarity score (if return_scores=True)
                - rank: Rank in results (0 = most similar)

        Example:
            >>> query_embedding = embedder.encode("What is machine learning?")
            >>> results = retriever.search(query_embedding, k=5)
            >>> for result in results:
            ...     print(f"Score: {result['score']:.3f}")
            ...     print(f"Text: {result['chunk']['text'][:100]}")
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty. Add documents before searching.")
            return []

        # Ensure query is 2D array of shape (1, dim)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype('float32')

        # Search
        k = min(k, self.index.ntotal)  # Can't retrieve more than we have
        distances, indices = self.index.search(query_embedding, k)

        # Build results
        results = []
        for rank, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.chunks):  # Safety check
                result = {
                    'chunk': self.chunks[idx],
                    'rank': rank
                }

                if return_scores:
                    # Convert L2 distance to similarity score
                    # Lower distance = higher similarity
                    # Normalized embeddings: similarity ≈ 1 - (distance²/4)
                    similarity = max(0, 1 - (distance / 4))
                    result['score'] = float(similarity)

                results.append(result)

        logger.info(f"Retrieved {len(results)} results for query")
        return results

    def save(self, save_dir: str):
        """
        Save index and chunks to disk.

        Args:
            save_dir: Directory to save index and chunks

        Files created:
            - index.faiss: The FAISS index
            - chunks.pkl: Pickled chunks
            - config.pkl: Configuration
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = save_path / "index.faiss"
        faiss.write_index(self.index, str(index_path))

        # Save chunks
        chunks_path = save_path / "chunks.pkl"
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)

        # Save config
        config = {
            'embedding_dimension': self.embedding_dimension,
            'index_type': self.index_type,
            'num_documents': self.index.ntotal,
            'is_trained': self.is_trained
        }
        config_path = save_path / "config.pkl"
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)

        logger.info(f"Saved retriever to {save_dir}")

    @classmethod
    def load(cls, load_dir: str) -> 'FAISSRetriever':
        """
        Load index and chunks from disk.

        Args:
            load_dir: Directory containing saved index

        Returns:
            Loaded FAISSRetriever instance
        """
        load_path = Path(load_dir)

        if not load_path.exists():
            raise FileNotFoundError(f"Directory not found: {load_dir}")

        # Load config
        config_path = load_path / "config.pkl"
        with open(config_path, 'rb') as f:
            config = pickle.load(f)

        # Create retriever instance
        retriever = cls(
            embedding_dimension=config['embedding_dimension'],
            index_type=config['index_type']
        )

        # Load FAISS index
        index_path = load_path / "index.faiss"
        retriever.index = faiss.read_index(str(index_path))
        retriever.is_trained = config['is_trained']

        # Load chunks
        chunks_path = load_path / "chunks.pkl"
        with open(chunks_path, 'rb') as f:
            retriever.chunks = pickle.load(f)

        logger.info(
            f"Loaded retriever from {load_dir} "
            f"({config['num_documents']} documents)"
        )

        return retriever

    def get_stats(self) -> Dict:
        """Get statistics about the index."""
        return {
            'num_documents': self.index.ntotal,
            'embedding_dimension': self.embedding_dimension,
            'index_type': self.index_type,
            'is_trained': self.is_trained
        }


if __name__ == "__main__":
    # Example usage
    from embeddings import EmbeddingModel

    # Create sample data
    sample_chunks = [
        {'text': 'Machine learning is a subset of AI', 'metadata': {'source': 'doc1'}},
        {'text': 'Deep learning uses neural networks', 'metadata': {'source': 'doc1'}},
        {'text': 'Python is a programming language', 'metadata': {'source': 'doc2'}},
    ]

    # Create embeddings
    embedder = EmbeddingModel()
    embeddings = embedder.encode_chunks(sample_chunks)

    # Create retriever and add documents
    retriever = FAISSRetriever(embedding_dimension=embeddings.shape[1])
    retriever.add_documents(sample_chunks, embeddings)

    # Search
    query = "What is neural network learning?"
    query_embedding = embedder.encode(query)
    results = retriever.search(query_embedding, k=2)

    print(f"\nQuery: {query}\n")
    for result in results:
        print(f"Score: {result['score']:.3f}")
        print(f"Text: {result['chunk']['text']}")
        print()
