"""Main RAG implementation."""

from .embeddings import DEFAULT_DIMENSIONS, EmbeddingModel
from .reranker import Reranker
from .types import QueryResult
from .vector_store import VectorStore


class TinyRAG:
    """A lightweight RAG system for Japanese text."""

    def __init__(self, dimensions: int = DEFAULT_DIMENSIONS) -> None:
        """Initialize TinyRAG.

        Args:
            dimensions: Embedding dimensions (32, 64, 128, 256, 512, or 1024).
        """
        self.dimensions = dimensions
        self._embedding_model = EmbeddingModel(dimensions)
        self._vector_store = VectorStore(dimensions)
        self._reranker = Reranker()

    @property
    def document_count(self) -> int:
        """Get the number of documents in the RAG system."""
        return self._vector_store.size

    def add_documents(self, documents: list[str]) -> None:
        """Add documents to the RAG system.

        Args:
            documents: List of document texts to add.
        """
        if not documents:
            return

        # Filter out empty documents
        valid_documents = [doc for doc in documents if doc.strip()]
        if not valid_documents:
            return

        # Generate embeddings for all documents
        embeddings = self._embedding_model.embed(valid_documents)

        # Add to vector store using batch method for better FAISS performance
        self._vector_store.add_batch(valid_documents, list(embeddings))

    def query(self, query: str, top_k: int = 5) -> list[QueryResult]:
        """Query the RAG system for relevant documents.

        Args:
            query: The query text.
            top_k: Number of top results to return.

        Returns:
            List of QueryResult objects sorted by relevance.
        """
        if not query.strip():
            return []

        # Generate query embedding
        query_embedding = self._embedding_model.embed(query)[0]  # Get first (and only) embedding

        # Retrieve more candidates than final top_k for reranking
        retrieve_k = min(top_k * 3, self._vector_store.size)

        # Search vector store
        search_results = self._vector_store.search(query_embedding, retrieve_k)

        # Convert to QueryResult objects
        query_results: list[QueryResult] = []
        for result in search_results:
            query_result = QueryResult(doc_id=result.doc_id, document=result.document, similarity=result.similarity)
            query_results.append(query_result)

        # Apply reranking
        if query_results:
            documents = [r.document for r in query_results]
            reranked_docs, reranked_scores = self._reranker.rerank(query, documents, top_k)

            # Create new QueryResult objects with reranked scores
            # Need to find original doc_ids for reranked documents
            doc_to_result = {r.document: r for r in query_results}
            reranked_results: list[QueryResult] = []

            for doc, score in zip(reranked_docs, reranked_scores, strict=True):
                original_result = doc_to_result[doc]
                reranked_result = QueryResult(
                    doc_id=original_result.doc_id,
                    document=doc,
                    similarity=score,  # Use reranker score instead of vector similarity
                )
                reranked_results.append(reranked_result)

            return reranked_results

        return query_results[:top_k]
