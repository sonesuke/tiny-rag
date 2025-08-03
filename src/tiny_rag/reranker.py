"""Reranker module using japanese-reranker-xsmall-v2 model."""

from sentence_transformers import CrossEncoder


class Reranker:
    """Reranker using japanese-reranker-xsmall-v2 for improved retrieval accuracy."""

    def __init__(self) -> None:
        """Initialize the reranker model."""
        self.model_name = "hotchpotch/japanese-reranker-xsmall-v2"

        # Initialize CrossEncoder with CPU device
        self._model = CrossEncoder(self.model_name, device="cpu")

    def score(self, query: str, documents: str | list[str]) -> list[float]:
        """Score query-document pair(s).

        Args:
            query: The query text.
            documents: Single document string or list of document strings.

        Returns:
            List of relevance scores (0.0-1.0). Single document returns list with one element.
        """
        # Convert single document to list for uniform processing
        if isinstance(documents, str):
            documents = [documents]

        # Handle empty list case
        if not documents:
            return []

        # Create query-document pairs for batch prediction
        pairs = [(query, doc) for doc in documents]

        # Use batch prediction for efficiency
        scores = self._model.predict(pairs)  # type: ignore[misc]

        return [float(score) for score in scores]

    def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> tuple[list[str], list[float]]:
        """Rerank documents by relevance to the query.

        Args:
            query: The query text.
            documents: List of document texts to rerank.
            top_k: Number of top results to return. If None, returns all.

        Returns:
            Tuple of (reranked_documents, scores) sorted by relevance (highest first).
        """
        if not documents:
            return [], []

        # Score all documents
        scores = self.score(query, documents)

        # Sort by score (descending)
        doc_score_pairs = list(zip(documents, scores, strict=True))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Apply top_k if specified
        if top_k is not None:
            doc_score_pairs = doc_score_pairs[:top_k]

        # Separate documents and scores
        reranked_docs = [doc for doc, _ in doc_score_pairs]
        reranked_scores = [score for _, score in doc_score_pairs]

        return reranked_docs, reranked_scores
