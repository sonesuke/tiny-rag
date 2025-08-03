"""Embedding module using static-embedding-japanese model."""

import numpy as np
from sentence_transformers import SentenceTransformer

# Default embedding dimension for static-embedding-japanese
DEFAULT_DIMENSIONS = 1024


class EmbeddingModel:
    """Embedding model using static-embedding-japanese."""

    def __init__(self, dimensions: int = DEFAULT_DIMENSIONS) -> None:
        """Initialize the embedding model.

        Args:
            dimensions: Embedding dimensions (32, 64, 128, 256, 512, or 1024).
        """
        self.dimensions = dimensions
        self.model_name = "hotchpotch/static-embedding-japanese"

        # Initialize with truncate_dim always specified
        self._model = SentenceTransformer(self.model_name, device="cpu", truncate_dim=dimensions)

    def embed(self, texts: str | list[str]) -> np.ndarray:
        """Generate embeddings for text(s).

        Args:
            texts: Input text (string) or list of texts to embed.

        Returns:
            Array of normalized embeddings of shape (n_texts, dimensions).
            For single text input, shape is (1, dimensions).
        """
        # Convert single text to list for uniform processing
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)

        embeddings = self._model.encode(texts, convert_to_numpy=True)  # type: ignore[misc]
        embeddings = embeddings.astype(np.float32)

        # Normalize each embedding
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)  # Avoid division by zero
        return embeddings / norms
