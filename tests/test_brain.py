"""Unit tests for the ConnectomeEngine dynamical LIF simulation."""

import numpy as np

from drosophila_interpreter.brain import ConnectomeEngine
from drosophila_interpreter.config import V_REST, SensoryModality
from drosophila_interpreter.loader import create_mock_connectome


def test_engine_initialization_and_idle_state() -> None:
    """Verify engine starts at resting potential and remains silent without input."""
    data = create_mock_connectome(neuron_count=100)
    engine = ConnectomeEngine(data)

    assert np.all(engine.v == V_REST)

    # Run for 20 ms with 0 input
    raster = engine.run(duration_ms=20.0)
    assert not np.any(raster.spikes)
    assert np.allclose(engine.v, V_REST)


def test_sensory_injection_causes_spiking() -> None:
    """Verify strong sensory input drives ingress neurons to spike."""
    data = create_mock_connectome(neuron_count=100)
    engine = ConnectomeEngine(data)

    # Inject maximum threat signal
    engine.inject_sensory_inputs({SensoryModality.LOOMING_THREAT: 1.0})

    raster = engine.run(duration_ms=20.0)
    threat_indices = data.ingress_ports[SensoryModality.LOOMING_THREAT]

    # At least one threat sensory neuron should have spiked
    threat_spikes = raster.spikes[:, threat_indices]
    assert np.any(threat_spikes)


def test_synaptic_propagation_to_motor_circuits() -> None:
    """Verify threat signals propagate through synapses to fire Giant Fiber (ESCAPE)."""
    data = create_mock_connectome(neuron_count=100)
    engine = ConnectomeEngine(data)

    engine.inject_sensory_inputs({SensoryModality.LOOMING_THREAT: 1.0})
    raster = engine.run(duration_ms=50.0)

    # Egress: ESCAPE_TAKEOFF corresponds to indices 10, 11 in mock connectome
    escape_indices = data.egress_ports["ESCAPE_TAKEOFF"]
    escape_spikes = raster.spikes[:, escape_indices]

    # Signal must propagate across synapse and cause motor output
    assert np.sum(escape_spikes) > 0

    # Feeding neurons (index 12) should remain silent because no sugar was present
    feeding_indices = data.egress_ports["FEEDING_PROBOSCIS"]
    feeding_spikes = raster.spikes[:, feeding_indices]
    assert np.sum(feeding_spikes) == 0