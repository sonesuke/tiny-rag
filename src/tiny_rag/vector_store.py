"""FAISS-based vector store for similarity search."""

import faiss
import numpy as np

from .types import EmbeddingVector, SearchResult


class VectorStore:
    """FAISS-based vector store with cosine similarity search."""

    def __init__(self, dimensions: int) -> None:
        """Initialize the FAISS vector store.

        Args:
            dimensions: Dimensionality of embeddings to store.
        """
        self.dimensions = dimensions
        self._documents: list[str] = []
        
        # Create FAISS index with cosine similarity (normalized inner product)
        # Using IndexFlatIP for exact search with inner product
        self._index = faiss.IndexFlatIP(dimensions)
        
        # We need to normalize vectors for cosine similarity
        # cosine_similarity = normalized_inner_product

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
        
        # Normalize embedding for cosine similarity
        normalized_embedding = embedding / np.linalg.norm(embedding)
        
        # Add to FAISS index (needs to be 2D array)
        self._index.add(normalized_embedding.reshape(1, -1))
        
        return doc_id

    def add_batch(self, documents: list[str], embeddings: list[EmbeddingVector]) -> list[int]:
        """Add multiple documents and their embeddings to the store.

        Args:
            documents: List of document texts.
            embeddings: List of document embedding vectors.

        Returns:
            List of document IDs.

        Raises:
            ValueError: If embedding dimensions don't match or lists have different lengths.
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        doc_ids = []
        start_id = len(self._documents)
        
        # Prepare embeddings matrix
        embeddings_matrix = np.zeros((len(embeddings), self.dimensions), dtype=np.float32)
        
        for i, embedding in enumerate(embeddings):
            if embedding.shape != (self.dimensions,):
                msg = f"Embedding dimension {embedding.shape} does not match store dimension ({self.dimensions},)"
                raise ValueError(msg)
            
            # Normalize for cosine similarity
            embeddings_matrix[i] = embedding / np.linalg.norm(embedding)
            doc_ids.append(start_id + i)
        
        # Add all documents
        self._documents.extend(documents)
        
        # Add all embeddings to FAISS at once
        self._index.add(embeddings_matrix)
        
        return doc_ids

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

        if self.size == 0:
            return []

        # Normalize query embedding for cosine similarity
        normalized_query = query_embedding / np.linalg.norm(query_embedding)
        
        # Search (needs to be 2D array)
        k = min(top_k, self.size)
        distances, indices = self._index.search(normalized_query.reshape(1, -1), k)
        
        # Create results
        results: list[SearchResult] = []
        for i in range(k):
            idx = int(indices[0][i])
            similarity = float(distances[0][i])
            
            result = SearchResult(
                doc_id=idx,
                document=self._documents[idx],
                similarity=similarity,
            )
            results.append(result)

        return results