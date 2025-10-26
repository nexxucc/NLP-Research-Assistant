"""
Training Data Generator

Automatically generates training data from research papers for:
1. Embedding fine-tuning (sentence pairs)
2. LLM fine-tuning (Q&A pairs)

This eliminates the need for manual labeling.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from typing import List, Dict, Tuple
import random
import json
import logging
from tqdm import tqdm

from src.data.pdf_loader import PDFLoader
from src.data.chunker import TextChunker
from src.models.llm import LLMHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingDataGenerator:
    """
    Generates training data from research papers using LLM.

    Methods:
    - generate_embedding_pairs(): Creates positive/negative pairs for embedding training
    - generate_qa_pairs(): Creates question-answer pairs for LLM fine-tuning
    """

    def __init__(self, llm_handler: LLMHandler = None):
        """
        Initialize data generator.

        Args:
            llm_handler: LLM for generating synthetic questions
        """
        self.llm = llm_handler or LLMHandler(temperature=0.7)
        self.pdf_loader = PDFLoader()
        self.chunker = TextChunker()

    def generate_embedding_pairs(
        self,
        papers_dir: str,
        num_pairs_per_paper: int = 50,
        output_file: str = "data/training/embedding_pairs.json"
    ) -> List[Dict]:
        """
        Generate training pairs for embedding fine-tuning using TITLE/ABSTRACT-based approach.

        NEW APPROACH (query-based, no API needed):
        - Query: Title + Abstract from paper
        - Positive: Chunks from the SAME paper (relevant to the query)
        - Negative: Chunks from DIFFERENT papers

        This matches real RAG usage: user query → retrieve relevant documents

        Args:
            papers_dir: Directory with PDF papers
            num_pairs_per_paper: How many pairs per paper
            output_file: Where to save pairs

        Returns:
            List of training pair dictionaries with format:
            {'query': title/abstract, 'positive': chunk, 'negative': chunk}
        """
        logger.info("=" * 80)
        logger.info("GENERATING TITLE/ABSTRACT-BASED TRAINING PAIRS")
        logger.info("=" * 80)

        # Load papers
        logger.info(f"\n1. Loading papers from {papers_dir}...")
        documents = self.pdf_loader.load_directory(papers_dir)

        if not documents:
            raise ValueError(f"No papers found in {papers_dir}")

        logger.info(f"   Loaded {len(documents)} papers")

        # Chunk all papers and extract title/abstract
        logger.info("\n2. Extracting titles/abstracts and chunking papers...")
        all_chunks_by_paper = []
        paper_queries = []  # Store title/abstract for each paper

        for doc in documents:
            # Extract title/abstract as "query"
            # Use first page or first 500 chars as abstract proxy
            first_page_text = doc['pages'][0]['text'] if doc.get('pages') and len(doc['pages']) > 0 else doc['text'][:500]

            # Get title from metadata if available
            title = doc['metadata'].get('title', '')

            # Create query: combine title + first page (contains abstract)
            if title and title.strip():
                query = f"{title}\n\n{first_page_text[:800]}"
            else:
                # No title in metadata, use first 300 chars as query
                query = first_page_text[:300]

            paper_queries.append(query.strip())

            # Chunk the full document
            chunks = self.chunker.chunk_text(
                doc['text'],
                metadata=doc['metadata']
            )
            all_chunks_by_paper.append(chunks)

        total_chunks = sum(len(chunks) for chunks in all_chunks_by_paper)
        logger.info(f"   Created {total_chunks} total chunks")
        logger.info(f"   Extracted {len(paper_queries)} paper queries (title/abstract)")

        # Generate query-document pairs
        logger.info(f"\n3. Generating query-document pairs ({num_pairs_per_paper} per paper)...")
        pairs = []

        for paper_idx, chunks in enumerate(tqdm(all_chunks_by_paper, desc="Papers")):
            paper_name = documents[paper_idx]['metadata']['filename']
            query = paper_queries[paper_idx]

            # Skip if query is too short (corrupted PDF)
            if len(query) < 50:
                continue

            # Skip first 2 chunks (likely contain the abstract itself)
            # We want body chunks that are RELATED to but DIFFERENT from the abstract
            body_chunks = chunks[2:] if len(chunks) > 2 else chunks

            if not body_chunks:
                continue

            pairs_for_paper = 0

            # Create pairs: (query, positive_chunk_from_same_paper, negative_chunk_from_other_paper)
            for _ in range(min(num_pairs_per_paper, len(body_chunks))):
                # Positive: random chunk from this paper's body
                positive_chunk = random.choice(body_chunks)

                # Negative: random chunk from a different paper
                other_papers = [idx for idx in range(len(all_chunks_by_paper)) if idx != paper_idx]
                if not other_papers:
                    continue

                negative_paper_idx = random.choice(other_papers)
                negative_chunks = all_chunks_by_paper[negative_paper_idx]

                if not negative_chunks:
                    continue

                negative_chunk = random.choice(negative_chunks)

                # Create training pair
                pair = {
                    'query': query,
                    'positive': positive_chunk['text'],
                    'negative': negative_chunk['text'],
                    'source_paper': paper_name
                }

                pairs.append(pair)
                pairs_for_paper += 1

                if pairs_for_paper >= num_pairs_per_paper:
                    break

        logger.info(f"\n   Generated {len(pairs)} query-document training pairs")

        # Save pairs
        logger.info(f"\n4. Saving pairs to {output_file}...")
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(pairs, f, indent=2)

        logger.info(f"   Saved successfully!")

        logger.info("\n" + "=" * 80)
        logger.info(f"✓ TITLE/ABSTRACT-BASED PAIRS GENERATED: {len(pairs)} pairs")
        logger.info("=" * 80)
        logger.info("\nTraining approach: Query (title/abstract) → Document (body chunks)")
        logger.info("This matches real RAG usage and should give POSITIVE improvements!")

        return pairs

    def generate_qa_pairs(
        self,
        papers_dir: str,
        questions_per_chunk: int = 2,
        output_file: str = "data/training/qa_pairs.json"
    ) -> List[Dict]:
        """
        Generate Q&A pairs for LLM fine-tuning.

        Uses LLM to generate questions from chunks, then creates
        question-context-answer triplets.

        Args:
            papers_dir: Directory with PDF papers
            questions_per_chunk: Questions to generate per chunk
            output_file: Where to save Q&A pairs

        Returns:
            List of Q&A pair dictionaries
        """
        logger.info("=" * 80)
        logger.info("GENERATING Q&A TRAINING PAIRS")
        logger.info("=" * 80)

        # Load and chunk papers
        logger.info(f"\n1. Loading papers from {papers_dir}...")
        documents = self.pdf_loader.load_directory(papers_dir)
        logger.info(f"   Loaded {len(documents)} papers")

        logger.info("\n2. Chunking papers...")
        all_chunks = self.chunker.chunk_documents(documents)
        logger.info(f"   Created {len(all_chunks)} chunks")

        # Sample chunks (don't generate for ALL chunks, too expensive)
        max_chunks = min(100, len(all_chunks))
        sampled_chunks = random.sample(all_chunks, max_chunks)

        logger.info(f"\n3. Generating questions from {max_chunks} chunks...")
        logger.info(f"   (Generating {questions_per_chunk} questions per chunk)")

        qa_pairs = []

        for chunk in tqdm(sampled_chunks, desc="Chunks"):
            # Generate questions using LLM
            questions = self._generate_questions_from_chunk(
                chunk['text'],
                num_questions=questions_per_chunk
            )

            for question in questions:
                qa_pair = {
                    'question': question,
                    'context': chunk['text'],
                    'answer': self._generate_answer(question, chunk['text']),
                    'source': chunk['metadata'].get('filename', 'unknown'),
                    'page': chunk['metadata'].get('page_num', None)
                }

                qa_pairs.append(qa_pair)

        logger.info(f"\n   Generated {len(qa_pairs)} Q&A pairs")

        # Save pairs
        logger.info(f"\n4. Saving Q&A pairs to {output_file}...")
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(qa_pairs, f, indent=2)

        logger.info(f"   Saved successfully!")

        logger.info("\n" + "=" * 80)
        logger.info(f"✓ Q&A PAIRS GENERATED: {len(qa_pairs)} pairs")
        logger.info("=" * 80)

        return qa_pairs

    def _generate_questions_from_chunk(
        self,
        text: str,
        num_questions: int = 2
    ) -> List[str]:
        """
        Generate questions from a text chunk using LLM.

        Args:
            text: Text chunk
            num_questions: Number of questions to generate

        Returns:
            List of questions
        """
        prompt = f"""Generate {num_questions} specific, answerable questions based on the following research paper excerpt.
Questions should:
1. Be directly answerable from the text
2. Cover different aspects (methodology, findings, technical details)
3. Be clear and specific

TEXT:
{text}

Generate exactly {num_questions} questions, one per line, without numbering:"""

        try:
            response = self.llm.model.generate_content(prompt)
            questions = [q.strip() for q in response.text.strip().split('\n') if q.strip()]

            # Take only requested number
            return questions[:num_questions]

        except Exception as e:
            logger.warning(f"Error generating questions: {e}")
            return []

    def _generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer to question given context.

        Args:
            question: Question
            context: Context text

        Returns:
            Generated answer
        """
        prompt = f"""Answer the following question based ONLY on the provided context.
Be concise and specific.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

        try:
            response = self.llm.model.generate_content(prompt)
            return response.text.strip()

        except Exception as e:
            logger.warning(f"Error generating answer: {e}")
            return "Unable to generate answer"


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Check API key
    if not os.getenv('GOOGLE_API_KEY'):
        print("ERROR: Please set GOOGLE_API_KEY in .env file")
        exit(1)

    # Initialize generator
    generator = TrainingDataGenerator()

    # Generate embedding pairs
    print("\n" + "=" * 80)
    print("STEP 1: Generate Embedding Training Pairs")
    print("=" * 80)

    embedding_pairs = generator.generate_embedding_pairs(
        papers_dir="data/papers",
        num_pairs_per_paper=50,
        output_file="data/training/embedding_pairs.json"
    )

    print(f"\n✓ Generated {len(embedding_pairs)} embedding training pairs")
    print(f"  Saved to: data/training/embedding_pairs.json")

    # Generate Q&A pairs
    print("\n" + "=" * 80)
    print("STEP 2: Generate Q&A Training Pairs")
    print("=" * 80)

    qa_pairs = generator.generate_qa_pairs(
        papers_dir="data/papers",
        questions_per_chunk=2,
        output_file="data/training/qa_pairs.json"
    )

    print(f"\n✓ Generated {len(qa_pairs)} Q&A training pairs")
    print(f"  Saved to: data/training/qa_pairs.json")

    print("\n" + "=" * 80)
    print("✓ TRAINING DATA GENERATION COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Check generated data in data/training/")
    print("  2. Run embedding fine-tuning: notebooks/03_embedding_finetuning.ipynb")
    print("  3. Run LLM fine-tuning: notebooks/04_llm_finetuning.ipynb")
