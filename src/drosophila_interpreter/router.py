"""Sensory router translating text prompts into biophysical current injections."""

import numpy as np
from sentence_transformers import SentenceTransformer

from drosophila_interpreter.config import (
    COSINE_SIMILARITY_SATURATION,
    COSINE_SIMILARITY_THRESHOLD,
    SENSORY_ANCHORS,
    SensoryModality,
)


class SensoryRouter:
    """Encodes natural text into normalized sensory currents via semantic embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self._modality_keys: list[SensoryModality] = list(SENSORY_ANCHORS.keys())

        # Precompute and group normalized embeddings per modality
        self._anchor_matrix_map: dict[SensoryModality, np.ndarray] = {}
        for mod, texts in SENSORY_ANCHORS.items():
            embeddings = self.model.encode(texts, normalize_embeddings=True)
            self._anchor_matrix_map[mod] = np.asarray(embeddings, dtype=np.float32)

    def route(self, prompt: str) -> dict[SensoryModality, float]:
        """Convert an input prompt into normalized injection currents [0.0, 1.0]."""
        if not prompt.strip():
            return {mod: 0.0 for mod in self._modality_keys}

        prompt_vec = self.model.encode(prompt, normalize_embeddings=True)
        prompt_vec = np.asarray(prompt_vec, dtype=np.float32)

        currents: dict[SensoryModality, float] = {}
        denominator = COSINE_SIMILARITY_SATURATION - COSINE_SIMILARITY_THRESHOLD

        for mod in self._modality_keys:
            anchors = self._anchor_matrix_map[mod]
            # Dot product against all sub-anchors, take the strongest match
            sub_sims = np.dot(anchors, prompt_vec)
            max_sim = float(np.max(sub_sims))

            if max_sim <= COSINE_SIMILARITY_THRESHOLD:
                currents[mod] = 0.0
            else:
                scaled = (max_sim - COSINE_SIMILARITY_THRESHOLD) / denominator
                currents[mod] = float(np.clip(scaled, 0.0, 1.0))

        return currents