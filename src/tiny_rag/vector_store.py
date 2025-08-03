"""In-memory vector store for similarity search."""

import numpy as np

from .types import EmbeddingVector, SearchResult


class VectorStore:
    """Simple in-memory vector store with cosine similarity search."""

    def __init__(self, dimensions: int) -> None:
        """Initialize the vector store.

        Args:
            dimensions: Dimensionality of embeddings to store.
        """
        self.dimensions = dimensions
        self._embeddings: list[EmbeddingVector] = []
        self._documents: list[str] = []

    @property
    def size(self) -> int:
        """Get the number of documents in the store."""
        return len(self._documents)

    def add(self, document: str, embedding: EmbeddingVector) -> int:
        """Add a document and its embedding to the store.

        Args:
            document: The document text.
            embedding: The document embedding vector.

        Returns:
            The document ID (index).

        Raises:
            ValueError: If embedding dimensions don't match.
        """
        if embedding.shape != (self.dimensions,):
            msg = f"Embedding dimension {embedding.shape} does not match store dimension ({self.dimensions},)"
            raise ValueError(msg)

        doc_id = len(self._documents)
        self._documents.append(document)
        self._embeddings.append(embedding)
        return doc_id

    def search(self, query_embedding: EmbeddingVector, top_k: int = 5) -> list[SearchResult]:
        """Search for similar documents using cosine similarity.

        Args:
            query_embedding: The query embedding vector.
            top_k: Number of top results to return.

        Returns:
            List of SearchResult objects sorted by similarity (descending).

        Raises:
            ValueError: If query embedding dimensions don't match.
        """
        if query_embedding.shape != (self.dimensions,):
            msg = (
                f"Query embedding dimension {query_embedding.shape} does not match store dimension ({self.dimensions},)"
            )
            raise ValueError(msg)

        if not self._embeddings:
            return []

        # Calculate cosine similarities
        embeddings_matrix = np.stack(self._embeddings)
        similarities = np.dot(embeddings_matrix, query_embedding)

        # Get top-k indices sorted by similarity (descending)
        top_indices: np.ndarray = np.argsort(similarities)[::-1][:top_k]

        # Create results
        results: list[SearchResult] = []
        for idx in top_indices:
            idx_int = int(idx)  # Convert numpy int to Python int for type safety
            result = SearchResult(
                doc_id=idx_int,
                document=self._documents[idx_int],
                similarity=float(similarities[idx]),
            )
            results.append(result)

        return results
