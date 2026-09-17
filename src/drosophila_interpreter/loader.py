"""Connectome data ingestion, graph indexing, and sparse matrix generation."""

from dataclasses import dataclass, field
import logging
import numpy as np
import scipy.sparse as sp

from drosophila_interpreter.config import (
    INGRESS_CELL_TYPES,
    MOTOR_CIRCUIT_TAGS,
    SYNAPSE_WEIGHT_SCALE,
    SensoryModality,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConnectomeData:
    """Immutable in-memory container for the processed connectome graph."""

    weights: sp.csr_matrix
    body_id_to_idx: dict[int, int]
    idx_to_body_id: np.ndarray
    ingress_ports: dict[SensoryModality, np.ndarray] = field(default_factory=dict)
    egress_ports: dict[str, np.ndarray] = field(default_factory=dict)
    neuron_count: int = 0


def build_csr_weights(
    pre_indices: np.ndarray,
    post_indices: np.ndarray,
    syn_counts: np.ndarray,
    signs: np.ndarray,
    neuron_count: int,
    weight_scale: float = SYNAPSE_WEIGHT_SCALE,
) -> sp.csr_matrix:
    """Build a signed CSR matrix representing directed synaptic weights.

    W[pre, post] = sign * syn_count * weight_scale
    """
    scaled_weights = (syn_counts * signs * weight_scale).astype(np.float32)
    weights = sp.csr_matrix(
        (scaled_weights, (pre_indices, post_indices)),
        shape=(neuron_count, neuron_count),
        dtype=np.float32,
    )
    return weights


def resolve_port_indices(
    annotations: list[str],
    target_tags: tuple[str, ...],
) -> np.ndarray:
    """Find row indices whose cell-type annotations match any of the target tags."""
    matched_indices = []
    target_lower = tuple(tag.lower() for tag in target_tags)
    for idx, annotation in enumerate(annotations):
        ann_lower = annotation.lower()
        if any(tag in ann_lower for tag in target_lower):
            matched_indices.append(idx)
    return np.array(matched_indices, dtype=np.int32)


def create_mock_connectome(neuron_count: int = 100) -> ConnectomeData:
    """Generate a deterministic synthetic connectome for testing and CI.

    Constructs clear feedforward test circuits:
    - Modality.SUGAR -> FEEDING_PROBOSCIS
    - Modality.LOOMING_THREAT -> ESCAPE_TAKEOFF
    """
    body_ids = np.arange(1000, 1000 + neuron_count, dtype=np.int64)
    body_id_to_idx = {int(bid): i for i, bid in enumerate(body_ids)}
    idx_to_body_id = body_ids

    # Assign synthetic annotations
    annotations = ["interneuron"] * neuron_count
    # Ingress cells
    annotations[0] = "GRN_sugar_primary"
    annotations[1] = "GRN_sugar_secondary"
    annotations[2] = "LC4_looming_detector"
    annotations[3] = "LPLC2_looming_detector"
    annotations[4] = "JO-AB_sound_vibration"
    annotations[5] = "TRPA1_thermo"
    annotations[6] = "Ir40a_hygro"
    annotations[7] = "Gr21a_co2"

    # Egress cells
    annotations[10] = "Giant_Fiber_takeoff"
    annotations[11] = "DNp01_takeoff"
    annotations[12] = "FIP_feeding_proboscis"
    annotations[13] = "aDN1_grooming"
    annotations[14] = "DNb01_halt"
    annotations[15] = "MDN_moonwalker"
    annotations[16] = "pIP10_courtship"

    # Wire excitatory circuits:
    # 0, 1 (Sugar) -> 12 (FIP)
    # 2, 3 (Threat) -> 10, 11 (Giant Fiber / Takeoff)
    pre = np.array([0, 1, 2, 3, 2, 3], dtype=np.int32)
    post = np.array([12, 12, 10, 11, 11, 10], dtype=np.int32)
    syn_counts = np.array([300, 300, 350, 350, 350, 350], dtype=np.float32)
    signs = np.array([1, 1, 1, 1, 1, 1], dtype=np.int8)  # All excitatory

    weights = build_csr_weights(pre, post, syn_counts, signs, neuron_count)

    ingress_ports = {
        modality: resolve_port_indices(annotations, tags)
        for modality, tags in INGRESS_CELL_TYPES.items()
    }
    egress_ports = {
        action: resolve_port_indices(annotations, tags)
        for action, tags in MOTOR_CIRCUIT_TAGS.items()
    }

    return ConnectomeData(
        weights=weights,
        body_id_to_idx=body_id_to_idx,
        idx_to_body_id=idx_to_body_id,
        ingress_ports=ingress_ports,
        egress_ports=egress_ports,
        neuron_count=neuron_count,
    )