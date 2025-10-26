"""
Hybrid Retrieval Module

Combines dense (semantic) and sparse (keyword) search with re-ranking.
Provides better retrieval quality than semantic search alone.

Architecture:
    1. Dense retrieval: FAISS vector similarity (semantic meaning)
    2. Sparse retrieval: BM25 keyword matching (exact terms)
    3. Fusion: Combine scores from both methods
    4. Re-ranking: Cross-encoder for final ranking

Why hybrid works better:
    - Semantic search: finds similar concepts ("AI" ~ "machine learning")
    - Keyword search: finds exact terms (important for technical terms)
    - Together: Best of both worlds!
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import logging

from src.models.retriever import FAISSRetriever
from src.models.embeddings import EmbeddingModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retrieval combining FAISS (semantic) + BM25 (keyword) + Cross-Encoder (re-ranking).

    Example:
        >>> retriever = HybridRetriever(embedding_dim=384)
        >>> retriever.add_documents(chunks, embeddings)
        >>> results = retriever.search("What is RAG?", k=5)
    """

    def __init__(
        self,
        embedding_dimension: int,
        faiss_index_type: str = "Flat",
        use_reranking: bool = True,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        alpha: float = 0.5
    ):
        """
        Initialize hybrid retriever.

        Args:
            embedding_dimension: Dimension of embeddings
            faiss_index_type: Type of FAISS index ("Flat" or "IVF")
            use_reranking: Whether to use cross-encoder re-ranking
            reranker_model: Cross-encoder model for re-ranking
            alpha: Weight for fusion (0.0 = only BM25, 1.0 = only FAISS, 0.5 = balanced)
        """
        self.embedding_dimension = embedding_dimension
        self.alpha = alpha
        self.use_reranking = use_reranking

        # Initialize FAISS retriever (semantic search)
        self.faiss_retriever = FAISSRetriever(
            embedding_dimension=embedding_dimension,
            index_type=faiss_index_type
        )

        # BM25 will be initialized when documents are added
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[Dict] = []

        # Initialize cross-encoder for re-ranking
        if use_reranking:
            logger.info(f"Loading cross-encoder: {reranker_model}")
            self.reranker = CrossEncoder(reranker_model)
            logger.info("✓ Cross-encoder loaded")
        else:
            self.reranker = None

        logger.info(
            f"Initialized Hybrid Retriever (FAISS + BM25, "
            f"re-ranking={'ON' if use_reranking else 'OFF'}, alpha={alpha})"
        )

    def add_documents(self, chunks: List[Dict], embeddings: np.ndarray):
        """
        Add documents to both FAISS and BM25 indices.

        Args:
            chunks: List of chunk dictionaries
            embeddings: Numpy array of embeddings
        """
        # Add to FAISS
        self.faiss_retriever.add_documents(chunks, embeddings)

        # Store chunks
        self.chunks = chunks

        # Build BM25 index
        logger.info("Building BM25 index...")
        texts = [chunk['text'] for chunk in chunks]

        # Tokenize for BM25 (simple whitespace splitting)
        tokenized_texts = [text.lower().split() for text in texts]

        self.bm25 = BM25Okapi(tokenized_texts)

        logger.info(f"✓ BM25 index built with {len(texts)} documents")

    def _search_faiss(
        self,
        query_embedding: np.ndarray,
        k: int
    ) -> List[Tuple[int, float]]:
        """
        Search using FAISS (semantic similarity).

        Returns:
            List of (chunk_index, similarity_score) tuples
        """
        results = self.faiss_retriever.search(query_embedding, k=k * 2)  # Get 2x for fusion

        # Extract indices and scores
        return [(
            self.chunks.index(result['chunk']),
            result['score']
        ) for result in results]

    def _search_bm25(
        self,
        query: str,
        k: int
    ) -> List[Tuple[int, float]]:
        """
        Search using BM25 (keyword matching).

        Returns:
            List of (chunk_index, bm25_score) tuples
        """
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call add_documents() first.")

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(tokenized_query)

        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:k * 2]  # Get 2x for fusion

        return [(idx, scores[idx]) for idx in top_indices]

    def _fuse_scores(
        self,
        faiss_results: List[Tuple[int, float]],
        bm25_results: List[Tuple[int, float]],
        k: int
    ) -> List[int]:
        """
        Fuse FAISS and BM25 scores using weighted average.

        Args:
            faiss_results: Results from FAISS search
            bm25_results: Results from BM25 search
            k: Number of results to return

        Returns:
            List of chunk indices sorted by fused score
        """
        # Normalize scores to [0, 1] range
        def normalize_scores(results):
            if not results:
                return {}

            scores = [score for _, score in results]
            min_score = min(scores)
            max_score = max(scores)

            if max_score == min_score:
                return {idx: 1.0 for idx, _ in results}

            return {
                idx: (score - min_score) / (max_score - min_score)
                for idx, score in results
            }

        # Normalize both score sets
        faiss_norm = normalize_scores(faiss_results)
        bm25_norm = normalize_scores(bm25_results)

        # Combine scores
        all_indices = set(faiss_norm.keys()) | set(bm25_norm.keys())

        fused_scores = {}
        for idx in all_indices:
            faiss_score = faiss_norm.get(idx, 0.0)
            bm25_score = bm25_norm.get(idx, 0.0)

            # Weighted fusion
            fused_scores[idx] = self.alpha * faiss_score + (1 - self.alpha) * bm25_score

        # Sort by fused score and return top k indices
        sorted_indices = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        return [idx for idx, _ in sorted_indices[:k]]

    def _rerank(
        self,
        query: str,
        candidate_indices: List[int],
        k: int
    ) -> List[Dict]:
        """
        Re-rank candidates using cross-encoder.

        Args:
            query: Query text
            candidate_indices: Indices of candidate chunks
            k: Number of final results

        Returns:
            List of re-ranked chunk dictionaries with scores
        """
        if self.reranker is None:
            # No re-ranking, return candidates as-is
            return [{
                'chunk': self.chunks[idx],
                'rank': rank,
                'score': None
            } for rank, idx in enumerate(candidate_indices[:k])]

        # Prepare pairs for cross-encoder
        pairs = [[query, self.chunks[idx]['text']] for idx in candidate_indices]

        # Get re-ranking scores
        rerank_scores = self.reranker.predict(pairs)

        # Sort by re-ranking score
        ranked = sorted(
            zip(candidate_indices, rerank_scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Build results
        results = []
        for rank, (idx, score) in enumerate(ranked[:k]):
            results.append({
                'chunk': self.chunks[idx],
                'rank': rank,
                'score': float(score)
            })

        return results

    def search(
        self,
        query: str,
        query_embedding: np.ndarray,
        k: int = 5,
        return_scores: bool = True
    ) -> List[Dict]:
        """
        Hybrid search combining FAISS, BM25, and re-ranking.

        Args:
            query: Query text (for BM25 and re-ranking)
            query_embedding: Query embedding (for FAISS)
            k: Number of results to return
            return_scores: Include scores in results

        Returns:
            List of ranked results

        Example:
            >>> embedder = EmbeddingModel()
            >>> query_emb = embedder.encode("What is RAG?")
            >>> results = hybrid_retriever.search("What is RAG?", query_emb, k=5)
        """
        logger.info(f"Hybrid search for: '{query}'")

        # Step 1: Search with FAISS (semantic)
        logger.info("  1/3 FAISS search...")
        faiss_results = self._search_faiss(query_embedding, k)

        # Step 2: Search with BM25 (keyword)
        logger.info("  2/3 BM25 search...")
        bm25_results = self._search_bm25(query, k)

        # Step 3: Fuse scores
        logger.info("  3/3 Score fusion...")
        fused_indices = self._fuse_scores(faiss_results, bm25_results, k * 3)

        # Step 4: Re-rank (optional)
        if self.use_reranking:
            logger.info("  4/4 Re-ranking...")
            final_results = self._rerank(query, fused_indices, k)
        else:
            final_results = [{
                'chunk': self.chunks[idx],
                'rank': rank,
                'score': None
            } for rank, idx in enumerate(fused_indices[:k])]

        logger.info(f"✓ Retrieved {len(final_results)} results")

        return final_results

    def get_stats(self) -> Dict:
        """Get retriever statistics."""
        stats = {
            'num_documents': len(self.chunks),
            'embedding_dimension': self.embedding_dimension,
            'bm25_enabled': self.bm25 is not None,
            'reranking_enabled': self.reranker is not None,
            'fusion_alpha': self.alpha
        }

        stats.update(self.faiss_retriever.get_stats())

        return stats

    def save(self, save_dir: str):
        """Save hybrid retriever (FAISS index + chunk metadata)."""
        self.faiss_retriever.save(save_dir)
        logger.info(f"✓ Hybrid retriever saved to {save_dir}")
        logger.info("  Note: BM25 index will be rebuilt on load")

    @classmethod
    def load(
        cls,
        load_dir: str,
        use_reranking: bool = True,
        alpha: float = 0.5
    ) -> 'HybridRetriever':
        """Load hybrid retriever from disk."""
        # Load FAISS retriever
        faiss_retriever = FAISSRetriever.load(load_dir)

        # Create hybrid retriever
        retriever = cls(
            embedding_dimension=faiss_retriever.embedding_dimension,
            faiss_index_type=faiss_retriever.index_type,
            use_reranking=use_reranking,
            alpha=alpha
        )

        # Set FAISS retriever
        retriever.faiss_retriever = faiss_retriever
        retriever.chunks = faiss_retriever.chunks

        # Rebuild BM25 index
        logger.info("Rebuilding BM25 index...")
        texts = [chunk['text'] for chunk in retriever.chunks]
        tokenized_texts = [text.lower().split() for text in texts]
        retriever.bm25 = BM25Okapi(tokenized_texts)

        logger.info(f"✓ Loaded hybrid retriever from {load_dir}")

        return retriever


if __name__ == "__main__":
    # Example usage
    from src.models.embeddings import EmbeddingModel

    # Create sample chunks
    chunks = [
        {'text': 'Machine learning is a subset of AI', 'metadata': {}},
        {'text': 'Deep learning uses neural networks', 'metadata': {}},
        {'text': 'Python is a programming language', 'metadata': {}},
    ]

    # Create embeddings
    embedder = EmbeddingModel()
    embeddings = embedder.encode_chunks(chunks)

    # Create hybrid retriever
    hybrid = HybridRetriever(
        embedding_dimension=embeddings.shape[1],
        use_reranking=True
    )

    hybrid.add_documents(chunks, embeddings)

    # Search
    query = "What is neural network learning?"
    query_embedding = embedder.encode(query)

    results = hybrid.search(query, query_embedding, k=2)

    print(f"\nQuery: {query}\n")
    for result in results:
        print(f"Rank {result['rank'] + 1}: {result['chunk']['text']}")
        print(f"  Score: {result['score']:.3f}\n")
