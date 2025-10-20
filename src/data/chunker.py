"""
Text Chunking Module

Implements intelligent text chunking strategies for research papers.
Preserves semantic coherence while maintaining optimal chunk sizes for embeddings.
"""

from typing import List, Dict, Optional
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextChunker:
    """
    Intelligent text chunking for research papers.

    Creates overlapping chunks that preserve context and maintain semantic coherence.
    Chunks are sized optimally for embedding models (typically 256-512 tokens).

    Why chunking matters:
    - Embedding models have max input lengths (e.g., 512 tokens)
    - Smaller chunks = more precise retrieval
    - Overlap ensures context isn't lost at boundaries
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize chunker with size parameters.

        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
            separators: List of text separators to split on (ordered by priority)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Default separators: prioritize semantic boundaries
        # Order matters: try to split on larger semantic units first
        self.separators = separators or [
            "\n\n",      # Paragraph breaks (highest priority)
            "\n",        # Line breaks
            ". ",        # Sentences
            "! ",        # Sentences
            "? ",        # Sentences
            "; ",        # Clauses
            ", ",        # Phrases
            " ",         # Words (last resort)
        ]

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[Dict[str, any]]:
        """
        Chunk a text document into overlapping segments.

        Args:
            text: The text to chunk
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of chunk dictionaries containing:
                - text: The chunk text
                - metadata: Original metadata plus chunk_id
                - start_char: Starting character position
                - end_char: Ending character position

        Example:
            >>> chunker = TextChunker(chunk_size=512, chunk_overlap=50)
            >>> chunks = chunker.chunk_text(paper_text, metadata={'source': 'paper.pdf'})
            >>> print(f"Created {len(chunks)} chunks")
            >>> print(chunks[0]['text'][:100])
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []

        chunks = []
        metadata = metadata or {}

        # Split text using recursive splitting
        text_chunks = self._recursive_split(text, self.chunk_size)

        # Create overlapping chunks
        overlapping_chunks = self._create_overlapping_chunks(
            text_chunks,
            self.chunk_size,
            self.chunk_overlap
        )

        # Build chunk dictionaries with metadata
        current_pos = 0
        for i, chunk_text in enumerate(overlapping_chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_id'] = i
            chunk_metadata['total_chunks'] = len(overlapping_chunks)

            chunks.append({
                'text': chunk_text,
                'metadata': chunk_metadata,
                'start_char': current_pos,
                'end_char': current_pos + len(chunk_text),
                'chunk_size': len(chunk_text)
            })

            # Update position (accounting for overlap)
            current_pos += len(chunk_text) - self.chunk_overlap

        logger.info(f"Created {len(chunks)} chunks from text of length {len(text)}")
        return chunks

    def chunk_documents(
        self,
        documents: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """
        Chunk multiple documents (e.g., from PDFLoader).

        Args:
            documents: List of document dicts with 'text' and 'metadata'

        Returns:
            List of all chunks from all documents

        Example:
            >>> loader = PDFLoader()
            >>> docs = loader.load_directory("data/papers")
            >>> chunker = TextChunker()
            >>> all_chunks = chunker.chunk_documents(docs)
        """
        all_chunks = []

        for doc in documents:
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})

            # Add page-level chunking for better citation
            if 'pages' in doc:
                for page in doc['pages']:
                    page_metadata = metadata.copy()
                    page_metadata['page_num'] = page['page_num']

                    chunks = self.chunk_text(page['text'], page_metadata)
                    all_chunks.extend(chunks)
            else:
                chunks = self.chunk_text(text, metadata)
                all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} total chunks from {len(documents)} documents")
        return all_chunks

    def _recursive_split(
        self,
        text: str,
        target_size: int,
        separator_index: int = 0
    ) -> List[str]:
        """
        Recursively split text using separators in priority order.

        This tries to split on paragraph boundaries first, then sentences,
        then words, ensuring semantic coherence.
        """
        if len(text) <= target_size:
            return [text]

        if separator_index >= len(self.separators):
            # Last resort: hard split at character boundary
            return self._hard_split(text, target_size)

        separator = self.separators[separator_index]
        splits = text.split(separator)

        # If no splits or too many tiny splits, try next separator
        if len(splits) <= 1 or all(len(s) < 50 for s in splits):
            return self._recursive_split(text, target_size, separator_index + 1)

        # Merge splits to approximate target size
        chunks = []
        current_chunk = []
        current_size = 0

        for split in splits:
            split_size = len(split) + len(separator)

            if current_size + split_size <= target_size:
                current_chunk.append(split)
                current_size += split_size
            else:
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                current_chunk = [split]
                current_size = split_size

        if current_chunk:
            chunks.append(separator.join(current_chunk))

        return chunks

    def _hard_split(self, text: str, chunk_size: int) -> List[str]:
        """Split text at exact character boundaries (last resort)."""
        return [
            text[i:i + chunk_size]
            for i in range(0, len(text), chunk_size)
        ]

    def _create_overlapping_chunks(
        self,
        chunks: List[str],
        chunk_size: int,
        overlap: int
    ) -> List[str]:
        """
        Create overlapping chunks from non-overlapping chunks.

        Overlap helps preserve context at chunk boundaries, which is crucial
        for retrieval - a relevant passage split across chunks won't be lost.
        """
        if not chunks:
            return []

        if len(chunks) == 1:
            return chunks

        overlapping = []

        for i in range(len(chunks)):
            # Combine current chunk with overlap from next
            chunk = chunks[i]

            # Add overlap from next chunk if it exists
            if i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                overlap_text = next_chunk[:overlap]
                chunk = chunk + " " + overlap_text

            overlapping.append(chunk)

        return overlapping


if __name__ == "__main__":
    # Example usage
    chunker = TextChunker(chunk_size=512, chunk_overlap=50)

    # Test with sample text
    sample_text = """
    This is a sample research paper abstract. It contains multiple sentences
    and demonstrates how the chunking system works.

    The chunking system splits text intelligently. It preserves semantic boundaries.
    This ensures that related content stays together in the same chunk.

    Overlapping chunks help maintain context. This is important for retrieval systems.
    """ * 10  # Repeat to create longer text

    chunks = chunker.chunk_text(sample_text)
    print(f"Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i}:")
        print(f"Size: {chunk['chunk_size']} chars")
        print(f"Text preview: {chunk['text'][:100]}...")
