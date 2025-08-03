"""Integration tests for TinyRAG."""

from tiny_rag import TinyRAG


class TestTinyRAG:
    """Test suite for TinyRAG integration."""

    def test_init_default(self) -> None:
        """Test TinyRAG initialization with defaults."""
        rag = TinyRAG()
        assert rag.dimensions == 1024
        assert rag.document_count == 0

    def test_init_custom_dimensions(self) -> None:
        """Test TinyRAG initialization with custom dimensions."""
        rag = TinyRAG(dimensions=128)
        assert rag.dimensions == 128

    def test_add_single_document(self) -> None:
        """Test adding a single document."""
        rag = TinyRAG()
        rag.add_documents(["これは最初のドキュメントです"])
        assert rag.document_count == 1

    def test_add_multiple_documents(self) -> None:
        """Test adding multiple documents."""
        rag = TinyRAG()
        documents = ["ラーメンは日本の人気料理です", "寿司は伝統的な日本料理です", "天ぷらは揚げ物料理です"]
        rag.add_documents(documents)
        assert rag.document_count == 3

    def test_add_documents_incrementally(self) -> None:
        """Test adding documents in multiple calls."""
        rag = TinyRAG()
        rag.add_documents(["ドキュメント1"])
        rag.add_documents(["ドキュメント2", "ドキュメント3"])
        assert rag.document_count == 3

    def test_query_returns_results(self) -> None:
        """Test that query returns relevant results."""
        rag = TinyRAG()
        documents = [
            "ラーメンは日本の人気料理です。豚骨や醤油など様々な種類があります。",
            "寿司は伝統的な日本料理です。新鮮な魚を使います。",
            "天ぷらは野菜や海老を揚げた料理です。",
            "カレーライスは日本で人気のある洋食です。",
            "うどんは太い麺を使った料理です。",
        ]
        rag.add_documents(documents)

        results = rag.query("ラーメンについて教えてください")

        assert len(results) > 0
        assert len(results) <= 5  # Default top_k
        # First result should be about ramen
        assert "ラーメン" in results[0].document

    def test_query_with_custom_top_k(self) -> None:
        """Test query with custom top_k parameter."""
        rag = TinyRAG()
        documents = [f"ドキュメント{i}の内容" for i in range(10)]
        rag.add_documents(documents)

        results = rag.query("テストクエリ", top_k=3)
        assert len(results) == 3

    def test_query_empty_rag(self) -> None:
        """Test querying when no documents are added."""
        rag = TinyRAG()
        results = rag.query("何かを検索")
        assert len(results) == 0

    def test_query_result_structure(self) -> None:
        """Test that query results have correct structure."""
        rag = TinyRAG()
        rag.add_documents(["テストドキュメント"])

        results = rag.query("テスト")
        assert len(results) == 1

        result = results[0]
        assert hasattr(result, "document")
        assert hasattr(result, "similarity")
        assert hasattr(result, "doc_id")
        assert isinstance(result.document, str)
        assert isinstance(result.similarity, float)
        assert isinstance(result.doc_id, int)

    def test_semantic_search_quality(self) -> None:
        """Test that semantic search returns relevant results."""
        rag = TinyRAG()
        documents = [
            "pythonは人気のプログラミング言語です。機械学習によく使われます。",
            "JavaScriptはウェブ開発で広く使用される言語です。",
            "機械学習とディープラーニングはAIの重要な分野です。",
            "データサイエンスにはPythonがよく使われます。",
            "HTMLとCSSはウェブページの作成に必要です。",
        ]
        rag.add_documents(documents)

        # Query about machine learning should return Python and ML related docs
        results = rag.query("機械学習に適したプログラミング言語")

        # Check that top results mention Python or machine learning
        top_docs = [r.document for r in results[:2]]
        relevant_keywords = ["python", "Python", "機械学習", "データサイエンス"]
        assert any(keyword in doc for doc in top_docs for keyword in relevant_keywords)

    def test_empty_document_handling(self) -> None:
        """Test handling of empty documents."""
        rag = TinyRAG()
        rag.add_documents(["", "有効なドキュメント", ""])
        # Empty documents might be skipped or handled gracefully
        assert rag.document_count >= 1

    def test_example_from_readme(self) -> None:
        """Test the example from README.md works correctly."""
        # Initialize tiny-rag
        rag = TinyRAG()

        # Add documents (Japanese text)
        rag.add_documents(
            [
                "ドキュメント1の内容...",
                "ドキュメント2の内容...",
            ]
        )

        # Query (in Japanese)
        results = rag.query("あなたの質問をここに入力")

        # Should return results
        assert isinstance(results, list)
        assert all(hasattr(r, "document") for r in results)
