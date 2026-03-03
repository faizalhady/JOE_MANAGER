"""
Embedding Service
Loads the embedding model and provides text → vector conversion.
Supports BGE-family models that require a query instruction prefix.
This is the ONLY place in the codebase that touches the embedding model.
"""

import os

# Offline mode: skip slow HuggingFace update checks.
# Set EMBEDDING_OFFLINE=0 in .env to allow downloading new models.
# Once a model is cached locally, set it back to 1 for speed.
if os.getenv("EMBEDDING_OFFLINE", "1") == "1":
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import logging
from typing import List
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, EMBEDDING_DEVICE, EMBEDDING_DIMENSIONS, BATCH_SIZE, EMBEDDING_QUERY_PREFIX

logger = logging.getLogger(__name__)


class Embedder:
    """
    Wraps the sentence-transformers model.
    Loads once, embeds many. Uses GPU if available.
    
    BGE models require a special prefix on QUERY text (not on documents).
    This is handled automatically via embed_query() vs embed_batch().
    """

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or EMBEDDING_MODEL
        self.device = device or EMBEDDING_DEVICE
        self.model = None
        self.dimensions = EMBEDDING_DIMENSIONS
        self.query_prefix = EMBEDDING_QUERY_PREFIX

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
            logger.error(f"Failed to load embedding model on {self.device}: {e}")
            if self.device != "cpu":
                logger.info("Falling back to CPU...")
                self.device = "cpu"
                self.model = SentenceTransformer(self.model_name, device="cpu")
                logger.info("Model loaded on CPU.")
            else:
                raise

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string (document/passage — no query prefix).
        Returns a list of floats (the vector).
        """
        if self.model is None:
            self.load()

        embedding = self.model.encode([text], show_progress_bar=False, batch_size=1)
        return embedding[0].tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a search query WITH the BGE query instruction prefix.
        
        BGE models perform better when the query (not the document) is
        prefixed with an instruction string. This method handles that.
        For non-BGE models, set EMBEDDING_QUERY_PREFIX="" in .env.
        
        Use this for SEARCH queries only. Use embed_text/embed_batch for documents.
        """
        if self.model is None:
            self.load()

        prefixed_query = f"{self.query_prefix}{query}" if self.query_prefix else query
        embedding = self.model.encode([prefixed_query], show_progress_bar=False, batch_size=1)
        return embedding[0].tolist()

    def embed_batch(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """
        Embed multiple texts in batches (for document ingestion — no query prefix).
        Much faster than one-by-one. Returns a list of vectors.
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
            normalize_embeddings=True,
        )

        logger.info(f"Embedding complete. Generated {len(embeddings)} vectors.")
        return [e.tolist() for e in embeddings]

    def get_model_name(self) -> str:
        """Return the model name (stored in DB for tracking)."""
        return self.model_name

    def get_dimensions(self) -> int:
        """Return the vector dimensions."""
        return self.dimensions
