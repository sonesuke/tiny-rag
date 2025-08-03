"""Type definitions for tiny-rag."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class SearchResult:
    """Result from vector store search."""

    doc_id: int
    document: str
    similarity: float


@dataclass
class QueryResult:
    """Result from RAG query."""

    doc_id: int
    document: str
    similarity: float


EmbeddingVector = NDArray[np.float32]
