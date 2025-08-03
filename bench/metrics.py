"""Evaluation metrics for information retrieval."""

import numpy as np


def ndcg_at_k(relevance_scores: list[float], k: int = 10) -> float:
    """Calculate Normalized Discounted Cumulative Gain at k.

    Args:
        relevance_scores: List of relevance scores in ranked order.
        k: Number of top results to consider.

    Returns:
        NDCG@k score.
    """
    if not relevance_scores:
        return 0.0

    # Truncate to k
    scores = relevance_scores[:k]

    # Calculate DCG
    dcg = scores[0] if scores else 0.0
    for i, score in enumerate(scores[1:], start=2):
        dcg += score / np.log2(i)

    # Calculate IDCG (ideal DCG)
    ideal_scores = sorted(relevance_scores, reverse=True)[:k]
    idcg = ideal_scores[0] if ideal_scores else 0.0
    for i, score in enumerate(ideal_scores[1:], start=2):
        idcg += score / np.log2(i)

    # Return normalized DCG
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(relevance_scores: list[float], k: int = 10) -> float:
    """Calculate Mean Reciprocal Rank at k.

    Args:
        relevance_scores: List of relevance scores in ranked order.
        k: Number of top results to consider.

    Returns:
        MRR@k score.
    """
    if not relevance_scores:
        return 0.0

    # Find the rank of the first relevant item (score > 0)
    for i, score in enumerate(relevance_scores[:k], start=1):
        if score > 0:
            return 1.0 / i

    return 0.0


def map_at_k(relevance_scores: list[float], k: int = 10) -> float:
    """Calculate Mean Average Precision at k.

    Args:
        relevance_scores: List of relevance scores in ranked order.
        k: Number of top results to consider.

    Returns:
        MAP@k score.
    """
    if not relevance_scores:
        return 0.0

    scores = relevance_scores[:k]
    relevant_count = 0
    precision_sum = 0.0

    for i, score in enumerate(scores, start=1):
        if score > 0:
            relevant_count += 1
            precision_sum += relevant_count / i

    total_relevant = sum(1 for score in relevance_scores if score > 0)

    return precision_sum / total_relevant if total_relevant > 0 else 0.0


def hits_at_k(relevance_scores: list[float], k: int = 10) -> float:
    """Calculate Hits at k (whether at least one relevant item is in top k).

    Args:
        relevance_scores: List of relevance scores in ranked order.
        k: Number of top results to consider.

    Returns:
        Hits@k score (0.0 or 1.0).
    """
    if not relevance_scores:
        return 0.0

    # Check if any of the top k items is relevant
    for score in relevance_scores[:k]:
        if score > 0:
            return 1.0

    return 0.0


def evaluate_retrieval_metrics(
    ranked_results: list[tuple[str, float]], relevant_docs: set[str], k: int = 10
) -> dict[str, float]:
    """Evaluate retrieval metrics for a single query.

    Args:
        ranked_results: List of (doc_id, similarity_score) tuples in ranked order.
        relevant_docs: Set of relevant document IDs.
        k: Number of top results to consider.

    Returns:
        Dictionary containing metric scores.
    """
    # Convert to relevance scores (1 if relevant, 0 if not)
    relevance_scores = [1.0 if doc_id in relevant_docs else 0.0 for doc_id, _ in ranked_results]

    return {
        f"ndcg@{k}": ndcg_at_k(relevance_scores, k),
        f"mrr@{k}": mrr_at_k(relevance_scores, k),
        f"map@{k}": map_at_k(relevance_scores, k),
        f"hits@{k}": hits_at_k(relevance_scores, k),
    }


def aggregate_metrics(metric_scores: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate metrics across multiple queries.

    Args:
        metric_scores: List of metric dictionaries for each query.

    Returns:
        Dictionary with averaged metrics.
    """
    if not metric_scores:
        return {}

    # Get all metric names
    metric_names = metric_scores[0].keys()

    # Calculate averages
    aggregated = {}
    for metric in metric_names:
        scores = [metrics[metric] for metrics in metric_scores]
        aggregated[metric] = np.mean(scores)

    return aggregated
