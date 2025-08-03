"""Tests for the embeddings module."""

import numpy as np

from tiny_rag.embeddings import EmbeddingModel


class TestEmbeddingModel:
    """Test suite for EmbeddingModel."""

    def test_init_default_dimensions(self) -> None:
        """Test that model initializes with default 1024 dimensions."""
        model = EmbeddingModel()
        assert model.dimensions == 1024

    def test_init_custom_dimensions(self) -> None:
        """Test that model can be initialized with custom dimensions."""
        model = EmbeddingModel(dimensions=128)
        assert model.dimensions == 128

    def test_embed_single_text(self) -> None:
        """Test embedding a single text returns correct shape."""
        model = EmbeddingModel()
        text = "これはテストテキストです"
        embeddings = model.embed(text)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (1, 1024)
        assert embeddings.dtype == np.float32

    def test_embed_batch(self) -> None:
        """Test batch embedding returns correct shape."""
        model = EmbeddingModel()
        texts = ["最初のテキスト", "2番目のテキスト", "3番目のテキスト"]
        embeddings = model.embed(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 1024)
        assert embeddings.dtype == np.float32

    def test_embed_empty_text(self) -> None:
        """Test handling of empty text."""
        model = EmbeddingModel()
        embeddings = model.embed("")

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (1, 1024)

    def test_embed_batch_empty_list(self) -> None:
        """Test handling of empty batch."""
        model = EmbeddingModel()
        embeddings = model.embed([])

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (0, 1024)

    def test_dimension_reduction(self) -> None:
        """Test embedding with dimension reduction."""
        model = EmbeddingModel(dimensions=128)
        text = "次元削減のテスト"
        embeddings = model.embed(text)

        assert embeddings.shape == (1, 128)

    def test_embeddings_are_normalized(self) -> None:
        """Test that embeddings are L2 normalized."""
        model = EmbeddingModel()
        text = "正規化のテスト"
        embeddings = model.embed(text)

        # Check L2 norm is approximately 1
        norm = np.linalg.norm(embeddings[0])
        assert abs(norm - 1.0) < 1e-5

    def test_different_texts_produce_different_embeddings(self) -> None:
        """Test that different texts produce different embeddings."""
        model = EmbeddingModel()
        embeddings1 = model.embed("ラーメンが好きです")
        embeddings2 = model.embed("寿司が好きです")

        # Embeddings should be different
        similarity = np.dot(embeddings1[0], embeddings2[0])
        assert similarity < 0.99  # Not identical
        assert similarity > 0.0  # But somewhat related (both about food)
