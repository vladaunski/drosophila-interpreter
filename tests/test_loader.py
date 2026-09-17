"""Unit tests for connectome data loader and mock generation."""

import scipy.sparse as sp

from drosophila_interpreter.config import SensoryModality
from drosophila_interpreter.loader import create_mock_connectome


def test_mock_connectome_shape_and_types() -> None:
    """Verify mock connectome dimensions and data structures."""
    data = create_mock_connectome(neuron_count=100)

    assert data.neuron_count == 100
    assert isinstance(data.weights, sp.csr_matrix)
    assert data.weights.shape == (100, 100)
    assert len(data.body_id_to_idx) == 100
    assert len(data.idx_to_body_id) == 100


def test_mock_connectome_port_mappings() -> None:
    """Verify ingress and egress port resolutions match annotations."""
    data = create_mock_connectome(neuron_count=100)

    # Ingress checks
    sugar_idx = data.ingress_ports[SensoryModality.SUGAR]
    threat_idx = data.ingress_ports[SensoryModality.LOOMING_THREAT]
    assert len(sugar_idx) == 2
    assert set(sugar_idx) == {0, 1}
    assert len(threat_idx) == 2
    assert set(threat_idx) == {2, 3}

    # Egress checks
    escape_idx = data.egress_ports["ESCAPE_TAKEOFF"]
    feeding_idx = data.egress_ports["FEEDING_PROBOSCIS"]
    assert set(escape_idx) == {10, 11}
    assert set(feeding_idx) == {12}


def test_mock_connectome_synaptic_weights() -> None:
    """Verify synaptic weights exist between expected connected pairs."""
    data = create_mock_connectome(neuron_count=100)

    # Check connection from threat sensor (idx 2) to Giant Fiber (idx 10)
    w_threat_escape = data.weights[2, 10]
    assert w_threat_escape > 0.0

    # Disconnected pair should be zero
    assert data.weights[0, 10] == 0.0