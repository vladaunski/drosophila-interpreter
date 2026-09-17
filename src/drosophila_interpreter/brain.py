"""Dynamical Leaky Integrate-and-Fire (LIF) simulation engine."""

from dataclasses import dataclass

import numpy as np

from drosophila_interpreter.config import (
    DT,
    I_SENSORY_MAX,
    REFRACTORY_PERIOD,
    TAU_M,
    V_RESET,
    V_REST,
    V_THRESH,
    SensoryModality,
)
from drosophila_interpreter.loader import ConnectomeData


@dataclass(frozen=True)
class SpikeRaster:
    """Immutable record of spikes across a simulation window."""

    spikes: np.ndarray  # Shape: (T_steps, N_neurons), dtype bool
    timestamps: np.ndarray  # Shape: (T_steps,), float in ms
    duration_ms: float


class ConnectomeEngine:
    """Biophysical simulation engine managing voltage state and synaptic integration."""

    def __init__(self, data: ConnectomeData) -> None:
        self.data = data
        self.n = data.neuron_count

        # Internal biophysical state vectors
        self.v = np.full(self.n, V_REST, dtype=np.float32)
        self.refractory_timers = np.zeros(self.n, dtype=np.float32)
        self.spikes_prev = np.zeros(self.n, dtype=bool)
        self.i_ext = np.zeros(self.n, dtype=np.float32)

        # Precomputed LIF analytical decay scalars for speed
        self._alpha = float(np.exp(-DT / TAU_M))
        self._beta = float(TAU_M * (1.0 - self._alpha))

    def reset_state(self) -> None:
        """Reset membrane potentials and refractory timers to baseline."""
        self.v.fill(V_REST)
        self.refractory_timers.fill(0.0)
        self.spikes_prev.fill(False)
        self.i_ext.fill(0.0)

    def inject_sensory_inputs(
        self, sensory_signals: dict[SensoryModality, float]
    ) -> None:
        """Map normalized [0.0, 1.0] sensory signals to injection currents on ingress ports."""
        self.i_ext.fill(0.0)
        for modality, strength in sensory_signals.items():
            if strength <= 0.0:
                continue
            indices = self.data.ingress_ports.get(modality)
            if indices is not None and len(indices) > 0:
                # Scale current by signal strength
                self.i_ext[indices] += float(strength * I_SENSORY_MAX)

    def step(self, dt: float = DT) -> np.ndarray:
        """Advance the biological simulation by one discrete time step dt.

        Returns a 1D boolean array indicating which neurons spiked on this tick.
        """
        # 1. Calculate incoming synaptic current from neurons that spiked on previous tick
        # W has shape (pre, post). Transpose dot product gives sum over pre-synaptic spikes.
        i_syn = self.data.weights.T.dot(self.spikes_prev.astype(np.float32))

        # 2. Update refractory timers
        is_refractory = self.refractory_timers > 0.0
        self.refractory_timers[is_refractory] = np.maximum(
            0.0, self.refractory_timers[is_refractory] - dt
        )

        # 3. Integrate Leaky Integrate-and-Fire equation for non-refractory neurons
        # V(t+dt) = V_rest + alpha * (V(t) - V_rest) + beta * (I_syn + I_ext)
        active_mask = ~is_refractory
        total_current = i_syn + self.i_ext

        self.v[active_mask] = (
            V_REST
            + self._alpha * (self.v[active_mask] - V_REST)
            + self._beta * total_current[active_mask]
        )
        self.v[is_refractory] = V_RESET

        # 4. Spike threshold check
        spikes = (self.v >= V_THRESH) & active_mask

        # 5. Reset spiked cells and engage refractory period
        self.v[spikes] = V_RESET
        self.refractory_timers[spikes] = REFRACTORY_PERIOD
        self.spikes_prev = spikes

        return spikes

    def run(self, duration_ms: float = 100.0, dt: float = DT) -> SpikeRaster:
        """Run the simulation for duration_ms in batch mode and return spike history."""
        num_steps = int(np.round(duration_ms / dt))
        spike_record = np.zeros((num_steps, self.n), dtype=bool)
        timestamps = np.arange(num_steps, dtype=np.float32) * dt

        for step_idx in range(num_steps):
            spike_record[step_idx] = self.step(dt=dt)

        return SpikeRaster(
            spikes=spike_record,
            timestamps=timestamps,
            duration_ms=duration_ms,
        )