"""Semantic router translating text inputs into biological sensory currents."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from drosophila_interpreter.config import (
    COSINE_SIMILARITY_THRESHOLD,
    DEFAULT_EMBEDDING_MODEL,
    SENSORY_ANCHORS,
    SensoryModality,
)


class SensoryRouter:
    """Routes arbitrary natural language strings to biological receptor activations."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        threshold: float = COSINE_SIMILARITY_THRESHOLD,
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self._anchor_modalities: list[SensoryModality] = list(SENSORY_ANCHORS.keys())
        self._anchor_embeddings: np.ndarray = self._precompute_anchor_embeddings()

    def _precompute_anchor_embeddings(self) -> np.ndarray:
        """Embeds biological anchor descriptions once at initialization."""
        anchor_texts = [SENSORY_ANCHORS[m] for m in self._anchor_modalities]
        embeddings = self.model.encode(
            anchor_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    def route(self, text: str) -> dict[SensoryModality, float]:
        """Calculates normalized sensory current injections for an input prompt.

        Args:
            text: Arbitrary user prompt (e.g. 'A juicy ripe peach is placed here').

        Returns:
            Dictionary mapping each SensoryModality to a current value in [0.0, 1.0].
        """
        if not text.strip():
            return {modality: 0.0 for modality in self._anchor_modalities}

        text_embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        similarities = np.dot(self._anchor_embeddings, text_embedding)

        currents: dict[SensoryModality, float] = {}
        for modality, sim in zip(self._anchor_modalities, similarities, strict=True):
            if sim < self.threshold:
                scaled_current = 0.0
            else:
                scaled_current = float((sim - self.threshold) / (1.0 - self.threshold))
                scaled_current = min(max(scaled_current, 0.0), 1.0)

            currents[modality] = round(scaled_current, 4)

        return currents