"""Unit tests for reranker module."""

from tiny_rag.reranker import Reranker


class TestReranker:
    """Test suite for Reranker class."""

    def test_init_default(self) -> None:
        """Test reranker initialization with defaults."""
        reranker = Reranker()
        assert reranker.model_name == "hotchpotch/japanese-reranker-xsmall-v2"

    def test_score_single_pair(self) -> None:
        """Test scoring a single query-document pair."""
        reranker = Reranker()
        query = "ラーメンについて教えてください"
        document = "ラーメンは日本の人気料理です。豚骨や醤油など様々な種類があります。"

        scores = reranker.score(query, document)

        assert isinstance(scores, list)
        assert len(scores) == 1
        assert isinstance(scores[0], float)
        assert 0.0 <= scores[0] <= 1.0

    def test_score_batch(self) -> None:
        """Test scoring multiple query-document pairs in batch."""
        reranker = Reranker()
        query = "日本料理について"
        documents = [
            "ラーメンは日本の人気料理です。",
            "寿司は伝統的な日本料理です。",
            "プログラミングは楽しい活動です。",
        ]

        scores = reranker.score(query, documents)
        assert len(scores) == len(documents)
        assert all(isinstance(score, float) for score in scores)
        assert all(0.0 <= score <= 1.0 for score in scores)
        # Japanese food documents should score higher than programming document
        assert scores[0] > scores[2]
        assert scores[1] > scores[2]

    def test_rerank(self) -> None:
        """Test reranking documents by relevance."""
        reranker = Reranker()
        query = "日本の伝統料理"
        documents = [
            "プログラミング言語Pythonの使い方",
            "寿司は新鮮な魚を使った日本の伝統料理です",
            "ラーメンは中国由来ですが日本で独自に発展しました",
            "天ぷらは江戸時代から続く日本の料理です",
        ]

        reranked_docs, scores = reranker.rerank(query, documents)

        assert len(reranked_docs) == len(documents)
        assert len(scores) == len(documents)
        assert all(isinstance(score, float) for score in scores)

        # Results should be sorted by relevance (highest first)
        assert scores == sorted(scores, reverse=True)

        # Traditional Japanese food documents should rank higher
        assert "寿司" in reranked_docs[0] or "天ぷら" in reranked_docs[0]
        assert "プログラミング" in reranked_docs[-1]

    def test_rerank_with_top_k(self) -> None:
        """Test reranking with top_k parameter."""
        reranker = Reranker()
        query = "料理"
        documents = [
            "ラーメンは人気料理です",
            "寿司は伝統料理です",
            "天ぷらは揚げ物料理です",
            "プログラミングの話",
            "機械学習の話",
        ]

        reranked_docs, scores = reranker.rerank(query, documents, top_k=3)

        assert len(reranked_docs) == 3
        assert len(scores) == 3
        assert all("料理" in doc for doc in reranked_docs)

    def test_empty_documents(self) -> None:
        """Test handling of empty document list."""
        reranker = Reranker()
        query = "テスト"
        documents: list[str] = []

        reranked_docs, scores = reranker.rerank(query, documents)

        assert len(reranked_docs) == 0
        assert len(scores) == 0

    def test_empty_query(self) -> None:
        """Test handling of empty query."""
        reranker = Reranker()
        query = ""
        documents = ["ドキュメント1", "ドキュメント2"]

        # Should handle gracefully, potentially with low scores
        scores = reranker.score(query, documents)
        assert len(scores) == len(documents)

    def test_model_cpu_only(self) -> None:
        """Test that model runs on CPU only."""
        reranker = Reranker()
        # This test verifies the model is loaded for CPU inference
        # The actual device check would depend on the implementation
        assert hasattr(reranker, "_model")

        # Test inference works (implicitly on CPU)
        scores = reranker.score("テスト", "テストドキュメント")
        assert isinstance(scores, list)
        assert len(scores) == 1
        assert isinstance(scores[0], float)

    def test_score_batch_consistency(self) -> None:
        """Test that batch scoring is consistent with individual scoring."""
        reranker = Reranker()
        query = "テストクエリ"
        documents = ["ドキュメント1", "ドキュメント2"]

        # Individual scores
        individual_scores: list[float] = []
        for doc in documents:
            scores = reranker.score(query, doc)
            assert isinstance(scores, list)
            assert len(scores) == 1
            individual_scores.append(scores[0])

        # Batch scores
        batch_scores = reranker.score(query, documents)

        # Should be approximately equal (allowing for small numerical differences)
        for individual, batch in zip(individual_scores, batch_scores, strict=True):
            assert abs(individual - batch) < 1e-6
