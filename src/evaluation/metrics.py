"""
Evaluation Metrics for RAG Systems

Provides metrics for evaluating retrieval and generation quality.

Metrics included:
- Retrieval: Precision, Recall, NDCG, MRR
- Generation: ROUGE, BLEU, semantic similarity
- End-to-end: Answer relevance, source accuracy
"""

from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics import precision_score, recall_score, ndcg_score
from rouge_score import rouge_scorer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrievalMetrics:
    """
    Metrics for evaluating retrieval quality.

    Measures how well the system retrieves relevant chunks.
    """

    @staticmethod
    def precision_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
        """
        Calculate Precision@K.

        Precision = (# relevant retrieved in top-k) / k

        Args:
            retrieved: List of retrieved document IDs (ordered by rank)
            relevant: List of relevant document IDs
            k: Number of top results to consider

        Returns:
            Precision@K score (0 to 1)

        Example:
            >>> retrieved = [1, 3, 5, 7, 9]  # System retrieved these
            >>> relevant = [1, 2, 3, 4]       # These are actually relevant
            >>> precision_at_k(retrieved, relevant, k=3)
            0.667  # 2 out of 3 top results were relevant
        """
        retrieved_k = retrieved[:k]
        relevant_set = set(relevant)

        num_relevant_retrieved = len([doc for doc in retrieved_k if doc in relevant_set])

        return num_relevant_retrieved / k if k > 0 else 0.0

    @staticmethod
    def recall_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
        """
        Calculate Recall@K.

        Recall = (# relevant retrieved in top-k) / (# total relevant)

        Returns:
            Recall@K score (0 to 1)
        """
        retrieved_k = retrieved[:k]
        relevant_set = set(relevant)

        num_relevant_retrieved = len([doc for doc in retrieved_k if doc in relevant_set])

        return num_relevant_retrieved / len(relevant) if len(relevant) > 0 else 0.0

    @staticmethod
    def mean_reciprocal_rank(retrieved_lists: List[List[int]], relevant_lists: List[List[int]]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).

        MRR = average of (1 / rank of first relevant result)

        Args:
            retrieved_lists: List of retrieval results for each query
            relevant_lists: List of relevant documents for each query

        Returns:
            MRR score (0 to 1)

        Example:
            >>> retrieved = [[1, 3, 5], [2, 4, 6]]  # 2 queries
            >>> relevant = [[3, 4], [2, 5]]          # Relevant for each
            >>> mean_reciprocal_rank(retrieved, relevant)
            0.75  # Query 1: relevant at rank 2 (1/2), Query 2: rank 1 (1/1)
        """
        reciprocal_ranks = []

        for retrieved, relevant in zip(retrieved_lists, relevant_lists):
            relevant_set = set(relevant)

            for rank, doc_id in enumerate(retrieved, start=1):
                if doc_id in relevant_set:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)  # No relevant found

        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    @staticmethod
    def average_precision(retrieved: List[int], relevant: List[int]) -> float:
        """
        Calculate Average Precision (AP).

        AP = average of precision values at each relevant document position.

        Returns:
            AP score (0 to 1)
        """
        relevant_set = set(relevant)
        precisions = []
        num_relevant_seen = 0

        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant_set:
                num_relevant_seen += 1
                precision_at_rank = num_relevant_seen / rank
                precisions.append(precision_at_rank)

        return np.mean(precisions) if precisions else 0.0


class GenerationMetrics:
    """
    Metrics for evaluating generated answers.

    Measures quality of LLM-generated responses.
    """

    def __init__(self):
        """Initialize ROUGE scorer."""
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )

    def rouge_scores(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores.

        ROUGE measures overlap between generated and reference text.

        Args:
            prediction: Generated answer
            reference: Ground truth answer

        Returns:
            Dictionary with rouge1, rouge2, rougeL F1 scores

        Example:
            >>> metrics = GenerationMetrics()
            >>> pred = "The drone uses Intel RealSense cameras"
            >>> ref = "Intel RealSense cameras were used on the drone"
            >>> scores = metrics.rouge_scores(pred, ref)
            >>> print(scores['rouge1'])  # Word overlap
            0.75
        """
        scores = self.rouge_scorer.score(reference, prediction)

        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }

    def semantic_similarity(
        self,
        embedder,
        prediction: str,
        reference: str
    ) -> float:
        """
        Calculate semantic similarity using embeddings.

        Args:
            embedder: EmbeddingModel instance
            prediction: Generated answer
            reference: Ground truth answer

        Returns:
            Cosine similarity (0 to 1)
        """
        pred_emb = embedder.encode(prediction)
        ref_emb = embedder.encode(reference)

        return float(np.dot(pred_emb, ref_emb))


class RAGEvaluator:
    """
    End-to-end RAG system evaluator.

    Combines retrieval and generation metrics.
    """

    def __init__(self):
        """Initialize evaluator."""
        self.retrieval_metrics = RetrievalMetrics()
        self.generation_metrics = GenerationMetrics()

    def evaluate_retrieval(
        self,
        queries: List[str],
        retrieved_results: List[List[int]],
        relevant_docs: List[List[int]],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance across multiple queries.

        Args:
            queries: List of query strings
            retrieved_results: Retrieved document IDs for each query
            relevant_docs: Ground truth relevant docs for each query
            k_values: K values to evaluate

        Returns:
            Dictionary of metric_name -> score
        """
        results = {}

        # Precision and Recall at different K
        for k in k_values:
            precisions = []
            recalls = []

            for retrieved, relevant in zip(retrieved_results, relevant_docs):
                p = self.retrieval_metrics.precision_at_k(retrieved, relevant, k)
                r = self.retrieval_metrics.recall_at_k(retrieved, relevant, k)

                precisions.append(p)
                recalls.append(r)

            results[f'precision@{k}'] = np.mean(precisions)
            results[f'recall@{k}'] = np.mean(recalls)

            # F1 score
            p_mean = results[f'precision@{k}']
            r_mean = results[f'recall@{k}']

            if p_mean + r_mean > 0:
                results[f'f1@{k}'] = 2 * (p_mean * r_mean) / (p_mean + r_mean)
            else:
                results[f'f1@{k}'] = 0.0

        # MRR
        results['MRR'] = self.retrieval_metrics.mean_reciprocal_rank(
            retrieved_results,
            relevant_docs
        )

        # MAP (Mean Average Precision)
        aps = [
            self.retrieval_metrics.average_precision(retrieved, relevant)
            for retrieved, relevant in zip(retrieved_results, relevant_docs)
        ]
        results['MAP'] = np.mean(aps)

        return results

    def evaluate_generation(
        self,
        predictions: List[str],
        references: List[str],
        embedder=None
    ) -> Dict[str, float]:
        """
        Evaluate generation quality.

        Args:
            predictions: Generated answers
            references: Ground truth answers
            embedder: Optional EmbeddingModel for semantic similarity

        Returns:
            Dictionary of metric_name -> score
        """
        results = {}

        # ROUGE scores
        rouge_scores = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': []
        }

        for pred, ref in zip(predictions, references):
            scores = self.generation_metrics.rouge_scores(pred, ref)

            for key in rouge_scores:
                rouge_scores[key].append(scores[key])

        # Average ROUGE scores
        for key, scores in rouge_scores.items():
            results[key] = np.mean(scores)

        # Semantic similarity (if embedder provided)
        if embedder:
            similarities = [
                self.generation_metrics.semantic_similarity(embedder, pred, ref)
                for pred, ref in zip(predictions, references)
            ]
            results['semantic_similarity'] = np.mean(similarities)

        return results

    def generate_report(
        self,
        retrieval_results: Dict[str, float],
        generation_results: Dict[str, float]
    ) -> str:
        """
        Generate human-readable evaluation report.

        Args:
            retrieval_results: Results from evaluate_retrieval()
            generation_results: Results from evaluate_generation()

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("RAG SYSTEM EVALUATION REPORT")
        report.append("=" * 80)

        report.append("\n📊 RETRIEVAL METRICS")
        report.append("-" * 80)

        for metric, score in sorted(retrieval_results.items()):
            report.append(f"{metric:.<40} {score:.4f}")

        report.append("\n📝 GENERATION METRICS")
        report.append("-" * 80)

        for metric, score in sorted(generation_results.items()):
            report.append(f"{metric:.<40} {score:.4f}")

        report.append("\n" + "=" * 80)

        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    # Test retrieval metrics
    evaluator = RAGEvaluator()

    # Simulated data
    queries = ["query1", "query2"]
    retrieved = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    relevant = [[1, 3, 7], [6, 8, 11]]

    retrieval_results = evaluator.evaluate_retrieval(
        queries,
        retrieved,
        relevant,
        k_values=[1, 3, 5]
    )

    # Test generation metrics
    predictions = [
        "The drone uses Intel RealSense cameras.",
        "The system achieved high accuracy."
    ]

    references = [
        "Intel RealSense cameras were used on the drone.",
        "High accuracy was achieved by the system."
    ]

    generation_results = evaluator.evaluate_generation(predictions, references)

    # Print report
    report = evaluator.generate_report(retrieval_results, generation_results)
    print(report)
