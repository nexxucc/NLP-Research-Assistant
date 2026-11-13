"""
LLM Module

Handles integration with Large Language Models for question answering and generation.
Supports Google Gemini API (and can be extended for fine-tuned models).
"""

from typing import List, Dict, Optional
import os
import logging
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# "gemini-pro" was retired by Google; the model is now configurable via GEMINI_MODEL.
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

class LLMHandler:
    """
    Handler for Large Language Model interactions.

    Manages prompting, context injection, and response generation for RAG.
    Supports both API-based models (Gemini) and custom fine-tuned models.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        use_custom_model: bool = False,
        custom_model_path: Optional[str] = None
    ):
        """
        Initialize LLM handler.

        Args:
            model_name: Model identifier (defaults to GEMINI_MODEL, else "gemini-2.5-flash")
            api_key: Gemini API key for this handler. Falls back to GOOGLE_API_KEY.
                Each handler holds its own client, so different users never share a key.
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens in response
            use_custom_model: Whether to use custom fine-tuned model
            custom_model_path: Path to fine-tuned model (for Phase 4)

        Temperature guide:
            - 0.0-0.3: Factual, focused (good for Q&A)
            - 0.4-0.7: Balanced
            - 0.8-1.0: Creative (good for summarization)
        """
        model_name = model_name or DEFAULT_GEMINI_MODEL
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_custom_model = use_custom_model
        self.custom_model_path = custom_model_path

        if use_custom_model:
            logger.info(f"Custom model support will be added in Phase 4")
            logger.info("Falling back to Gemini for now")

        # Configure generation parameters
        self.generation_config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        self._api_key = None
        self.client = None
        self.set_api_key(api_key or os.getenv('GOOGLE_API_KEY'))

        logger.info(f"Initialized LLM: {model_name}")

    def set_api_key(self, api_key: Optional[str]) -> None:
        """Point this handler at a different Gemini API key (no-op if unchanged)."""
        if not api_key:
            raise ValueError(
                "No Gemini API key provided. Pass api_key or set GOOGLE_API_KEY."
            )
        if api_key != self._api_key:
            self.client = genai.Client(api_key=api_key)
            self._api_key = api_key

    def generate(self, prompt: str) -> str:
        """Send one prompt to Gemini with this handler's own key and settings."""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self.generation_config,
        )
        return response.text or ""

    def answer_question(
        self,
        question: str,
        context_chunks: List[Dict],
        include_sources: bool = True
    ) -> Dict[str, any]:
        """
        Answer a question using retrieved context chunks.

        This is the core RAG functionality:
        1. Take question and relevant chunks (from retriever)
        2. Build prompt with context
        3. Get LLM response
        4. Add source citations

        Args:
            question: User's question
            context_chunks: List of retrieved chunks (from FAISS search)
            include_sources: Whether to include source citations

        Returns:
            Dict containing:
                - answer: The generated answer
                - sources: List of source citations (if include_sources=True)
                - context_used: The context provided to LLM

        Example:
            >>> llm = LLMHandler()
            >>> chunks = retriever.search(query_embedding, k=5)
            >>> result = llm.answer_question("What is RAG?", chunks)
            >>> print(result['answer'])
            >>> print("Sources:", result['sources'])
        """
        # Build context from chunks
        context = self._build_context(context_chunks)

        # Create prompt
        prompt = self._create_qa_prompt(question, context)

        # Generate answer
        try:
            answer = self.generate(prompt)

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            answer = f"I apologize, but I encountered an error generating the answer: {str(e)}"

        # Extract sources
        sources = []
        if include_sources:
            sources = self._extract_sources(context_chunks)

        return {
            'answer': answer,
            'sources': sources,
            'context_used': context,
            'num_chunks_used': len(context_chunks)
        }

    def summarize_document(
        self,
        text: str,
        max_length: int = 500,
        focus: Optional[str] = None
    ) -> str:
        """
        Generate a summary of a document or section.

        Args:
            text: Text to summarize
            max_length: Target summary length in words
            focus: Optional focus area (e.g., "methodology", "findings")

        Returns:
            Generated summary

        Example:
            >>> summary = llm.summarize_document(
            ...     paper_text,
            ...     max_length=300,
            ...     focus="key findings"
            ... )
        """
        prompt = self._create_summary_prompt(text, max_length, focus)

        try:
            return self.generate(prompt)

        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return f"Error generating summary: {str(e)}"

    def _build_context(self, chunks: List[Dict], max_chunks: int = 5) -> str:
        """
        Build context string from retrieved chunks.

        Combines chunks with metadata for better prompting.
        """
        context_parts = []

        for i, chunk_data in enumerate(chunks[:max_chunks], 1):
            chunk = chunk_data.get('chunk', chunk_data)
            text = chunk.get('text', '')
            metadata = chunk.get('metadata', {})

            # Build context with source info
            source_info = f"[Source {i}"
            if 'filename' in metadata:
                source_info += f": {metadata['filename']}"
            if 'page_num' in metadata:
                source_info += f", Page {metadata['page_num']}"
            source_info += "]"

            context_parts.append(f"{source_info}\n{text}")

        return "\n\n".join(context_parts)

    def _create_qa_prompt(self, question: str, context: str) -> str:
        """
        Create a prompt for question answering with context.

        Prompt engineering is crucial for good RAG:
        - Clear instructions
        - Context before question
        - Request for citations
        - Handling of insufficient context
        """
        prompt = f"""You are a helpful research assistant. Answer the question based ONLY on the provided context from research papers.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer the question using information from the context above
2. If the context doesn't contain enough information to fully answer, say so clearly
3. Do not make up information or use knowledge beyond the provided context
4. Be specific and cite which source(s) support your answer
5. If relevant, mention specific details like methodology, findings, or data from the papers

ANSWER:"""
        return prompt

    def _create_summary_prompt(
        self,
        text: str,
        max_length: int,
        focus: Optional[str] = None
    ) -> str:
        """Create a prompt for document summarization."""
        focus_instruction = f" Focus especially on: {focus}." if focus else ""

        prompt = f"""Summarize the following research paper content in approximately {max_length} words.{focus_instruction}

Your summary should:
1. Capture the main ideas and key findings
2. Be clear and concise
3. Use technical terms appropriately
4. Highlight the most important contributions

TEXT TO SUMMARIZE:
{text}

SUMMARY:"""
        return prompt

    def _extract_sources(self, chunks: List[Dict]) -> List[Dict]:
        """
        Extract source information from chunks for citations.

        Returns:
            List of source dicts with filename, page, etc.
        """
        sources = []
        seen = set()  # Avoid duplicates

        for chunk_data in chunks:
            chunk = chunk_data.get('chunk', chunk_data)
            metadata = chunk.get('metadata', {})

            source_id = (
                metadata.get('filename', 'Unknown'),
                metadata.get('page_num', None)
            )

            if source_id not in seen:
                sources.append({
                    'filename': metadata.get('filename', 'Unknown'),
                    'page': metadata.get('page_num'),
                    'score': chunk_data.get('score', None)
                })
                seen.add(source_id)

        return sources

    def batch_summarize(
        self,
        texts: List[str],
        max_length: int = 300
    ) -> List[str]:
        """
        Summarize multiple texts (useful for multi-document summarization).

        Args:
            texts: List of texts to summarize
            max_length: Target length for each summary

        Returns:
            List of summaries
        """
        summaries = []

        for i, text in enumerate(texts, 1):
            logger.info(f"Summarizing document {i}/{len(texts)}")
            summary = self.summarize_document(text, max_length)
            summaries.append(summary)

        return summaries


if __name__ == "__main__":
    # Example usage
    llm = LLMHandler()

    # Test Q&A
    sample_chunks = [
        {
            'chunk': {
                'text': 'RAG (Retrieval-Augmented Generation) combines retrieval with generation. '
                        'It retrieves relevant documents and uses them to generate better answers.',
                'metadata': {'filename': 'rag_paper.pdf', 'page_num': 1}
            },
            'score': 0.95
        }
    ]

    result = llm.answer_question("What is RAG?", sample_chunks)
    print("Answer:", result['answer'])
    print("\nSources:", result['sources'])
