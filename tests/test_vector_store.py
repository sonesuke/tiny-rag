"""Tests for the vector store module."""

import numpy as np
import pytest

from tiny_rag.vector_store import VectorStore


class TestVectorStore:
    """Test suite for VectorStore."""

    def test_init_empty_store(self) -> None:
        """Test that vector store initializes empty."""
        store = VectorStore(dimensions=1024)
        assert store.size == 0
        assert store.dimensions == 1024

    def test_add_single_document(self) -> None:
        """Test adding a single document."""
        store = VectorStore(dimensions=128)
        embedding = np.random.rand(128).astype(np.float32)

        doc_id = store.add("これはテストドキュメントです", embedding)

        assert doc_id == 0
        assert store.size == 1

    def test_add_multiple_documents(self) -> None:
        """Test adding multiple documents."""
        store = VectorStore(dimensions=128)

        for i in range(5):
            embedding = np.random.rand(128).astype(np.float32)
            doc_id = store.add(f"ドキュメント{i}", embedding)
            assert doc_id == i

        assert store.size == 5

    def test_search_returns_top_k(self) -> None:
        """Test that search returns top-k results."""
        store = VectorStore(dimensions=128)

        # Add 10 documents
        for i in range(10):
            embedding = np.random.rand(128).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding).astype(np.float32)  # Normalize
            store.add(f"ドキュメント{i}", embedding)

        # Search with a query
        query_embedding = np.random.rand(128).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding).astype(np.float32)  # Normalize

        results = store.search(query_embedding, top_k=3)

        assert len(results) == 3
        # Check results are sorted by similarity (descending)
        for i in range(len(results) - 1):
            assert results[i].similarity >= results[i + 1].similarity

    def test_search_empty_store(self) -> None:
        """Test searching in empty store returns empty results."""
        store = VectorStore(dimensions=128)
        query_embedding = np.random.rand(128).astype(np.float32)

        results = store.search(query_embedding, top_k=5)

        assert len(results) == 0

    def test_search_with_fewer_docs_than_k(self) -> None:
        """Test search when store has fewer documents than k."""
        store = VectorStore(dimensions=128)

        # Add only 3 documents
        for i in range(3):
            embedding = np.random.rand(128).astype(np.float32)
            store.add(f"ドキュメント{i}", embedding)

        query_embedding = np.random.rand(128).astype(np.float32)
        results = store.search(query_embedding, top_k=5)

        assert len(results) == 3

    def test_search_result_structure(self) -> None:
        """Test that search results have correct structure."""
        store = VectorStore(dimensions=128)

        # Add a document
        embedding = np.ones(128, dtype=np.float32)
        embedding = embedding / np.linalg.norm(embedding).astype(np.float32)  # Normalize
        store.add("テストドキュメント", embedding)

        # Search with identical embedding
        results = store.search(embedding, top_k=1)

        assert len(results) == 1
        result = results[0]
        assert hasattr(result, "doc_id")
        assert hasattr(result, "document")
        assert hasattr(result, "similarity")
        assert result.doc_id == 0
        assert result.document == "テストドキュメント"
        assert abs(result.similarity - 1.0) < 1e-5  # Perfect match

    def test_dimension_mismatch_error(self) -> None:
        """Test that dimension mismatch raises error."""
        store = VectorStore(dimensions=128)

        # Try to add with wrong dimensions
        wrong_embedding = np.random.rand(256).astype(np.float32)
        with pytest.raises(ValueError, match="dimension"):
            store.add("テスト", wrong_embedding)

        # Try to search with wrong dimensions
        wrong_query = np.random.rand(256).astype(np.float32)
        with pytest.raises(ValueError, match="dimension"):
            store.search(wrong_query, top_k=1)

    def test_cosine_similarity_calculation(self) -> None:
        """Test that similarity is computed correctly (cosine similarity)."""
        store = VectorStore(dimensions=2)

        # Add perpendicular vectors
        doc1_embedding = np.array([1.0, 0.0], dtype=np.float32)
        doc2_embedding = np.array([0.0, 1.0], dtype=np.float32)
        doc3_embedding = np.array([0.707, 0.707], dtype=np.float32)  # 45 degrees

        store.add("X軸", doc1_embedding)
        store.add("Y軸", doc2_embedding)
        store.add("45度", doc3_embedding)

        # Query with X-axis vector
        results = store.search(doc1_embedding, top_k=3)

        # Check similarities
        assert abs(results[0].similarity - 1.0) < 1e-5  # Same as query
        assert abs(results[1].similarity - 0.707) < 1e-3  # 45 degrees
        assert abs(results[2].similarity - 0.0) < 1e-5  # Perpendicular
