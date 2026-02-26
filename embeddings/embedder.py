"""
Embedding Service
Loads all-mpnet-base-v2 on GPU and provides text → vector conversion.
This is the ONLY place in the codebase that touches the embedding model.
"""

import logging
from typing import List
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, EMBEDDING_DEVICE, EMBEDDING_DIMENSIONS, BATCH_SIZE

logger = logging.getLogger(__name__)


class Embedder:
    """
    Wraps the sentence-transformers model.
    Loads once, embeds many. Uses GPU if available.
    """

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or EMBEDDING_MODEL
        self.device = device or EMBEDDING_DEVICE
        self.model = None
        self.dimensions = EMBEDDING_DIMENSIONS

    def load(self):
        """Load the model into memory (GPU or CPU)."""
        logger.info(f"Loading embedding model: {self.model_name} on {self.device}...")

        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            # Verify dimensions match config
            test_embedding = self.model.encode(["test"], show_progress_bar=False)
            actual_dims = len(test_embedding[0])

            if actual_dims != self.dimensions:
                logger.warning(
                    f"Dimension mismatch! Config says {self.dimensions}, "
                    f"model produces {actual_dims}. Using actual: {actual_dims}"
                )
                self.dimensions = actual_dims

            logger.info(
                f"Model loaded. Dimensions: {self.dimensions}, Device: {self.device}"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            logger.info("Falling back to CPU...")
            self.device = "cpu"
            self.model = SentenceTransformer(self.model_name, device="cpu")
            logger.info("Model loaded on CPU.")

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string.
        Returns a list of floats (the vector).
        """
        if self.model is None:
            self.load()

        embedding = self.model.encode([text], show_progress_bar=False, batch_size=1)
        return embedding[0].tolist()

    def embed_batch(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """
        Embed multiple texts in batches. Much faster than one-by-one.
        Returns a list of vectors.
        """
        if self.model is None:
            self.load()

        if not texts:
            return []

        bs = batch_size or BATCH_SIZE
        logger.info(f"Embedding {len(texts)} texts in batches of {bs}...")

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=bs,
            normalize_embeddings=True,  # cosine similarity works better with normalized vectors
        )

        logger.info(f"Embedding complete. Generated {len(embeddings)} vectors.")
        return [e.tolist() for e in embeddings]

    def get_model_name(self) -> str:
        """Return the model name (stored in DB for tracking)."""
        return self.model_name

    def get_dimensions(self) -> int:
        """Return the vector dimensions."""
        return self.dimensions
