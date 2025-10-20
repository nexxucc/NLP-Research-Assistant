"""
Embedding Model Module

Handles text embedding generation using sentence-transformers.
Supports both generic and custom fine-tuned embedding models.
"""

from typing import List, Union, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Wrapper for sentence-transformer embedding models.

    Converts text into dense vector representations (embeddings) that capture
    semantic meaning. Similar texts have similar embeddings, enabling semantic search.

    Why embeddings:
    - Traditional keyword search misses semantic similarity
    - "machine learning" and "artificial intelligence" are different words but similar concepts
    - Embeddings capture this semantic relationship as vector proximity
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_custom_model: bool = False,
        custom_model_path: Optional[str] = None,
        device: str = None
    ):
        """
        Initialize embedding model.

        Args:
            model_name: HuggingFace model name for generic embeddings
            use_custom_model: Whether to use a custom fine-tuned model
            custom_model_path: Path to custom model (if use_custom_model=True)
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)

        Models:
            - all-MiniLM-L6-v2: Fast, 384 dimensions, good general purpose (default)
            - all-mpnet-base-v2: Better quality, 768 dimensions, slower
            - Custom fine-tuned: Domain-specific, optimized for research papers
        """
        self.use_custom_model = use_custom_model
        self.model_name = model_name
        self.custom_model_path = custom_model_path

        # Load model
        if use_custom_model and custom_model_path:
            if not Path(custom_model_path).exists():
                logger.warning(
                    f"Custom model not found at {custom_model_path}. "
                    f"Falling back to generic model: {model_name}"
                )
                self.model = SentenceTransformer(model_name, device=device)
                self.use_custom_model = False
            else:
                logger.info(f"Loading custom embedding model from {custom_model_path}")
                self.model = SentenceTransformer(custom_model_path, device=device)
        else:
            logger.info(f"Loading generic embedding model: {model_name}")
            self.model = SentenceTransformer(model_name, device=device)

        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dimension}")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        Convert text(s) into embeddings.

        Args:
            texts: Single text string or list of texts
            batch_size: Number of texts to process at once
            show_progress: Show progress bar for large batches
            normalize_embeddings: L2 normalize embeddings (recommended for similarity)

        Returns:
            numpy array of shape (num_texts, embedding_dim)

        Example:
            >>> embedder = EmbeddingModel()
            >>> embeddings = embedder.encode(["Hello world", "Machine learning"])
            >>> print(embeddings.shape)  # (2, 384)
            >>> similarity = np.dot(embeddings[0], embeddings[1])  # Cosine similarity
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True
        )

        return embeddings

    def encode_chunks(
        self,
        chunks: List[dict],
        text_key: str = 'text',
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode a list of chunk dictionaries.

        Args:
            chunks: List of dicts with text and metadata
            text_key: Key in dict that contains the text
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            numpy array of embeddings

        Example:
            >>> chunks = [
            ...     {'text': 'First chunk', 'metadata': {}},
            ...     {'text': 'Second chunk', 'metadata': {}}
            ... ]
            >>> embeddings = embedder.encode_chunks(chunks)
        """
        texts = [chunk[text_key] for chunk in chunks]

        logger.info(f"Encoding {len(texts)} chunks...")
        embeddings = self.encode(
            texts,
            batch_size=batch_size,
            show_progress=show_progress
        )

        logger.info(f"Generated embeddings of shape {embeddings.shape}")
        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self.embedding_dimension

    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between -1 and 1 (1 = identical, -1 = opposite)
        """
        # Assuming embeddings are already normalized
        return float(np.dot(embedding1, embedding2))

    def save_model(self, path: str):
        """
        Save the current model to disk.

        Useful for saving custom fine-tuned models.
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        logger.info(f"Model saved to {path}")


if __name__ == "__main__":
    # Example usage
    embedder = EmbeddingModel()

    # Test embedding generation
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "The weather is nice today."
    ]

    embeddings = embedder.encode(texts)
    print(f"Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")

    # Test similarity
    sim_1_2 = embedder.similarity(embeddings[0], embeddings[1])
    sim_1_3 = embedder.similarity(embeddings[0], embeddings[2])

    print(f"\nSimilarity between text 1 and 2 (related): {sim_1_2:.3f}")
    print(f"Similarity between text 1 and 3 (unrelated): {sim_1_3:.3f}")
