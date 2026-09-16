"""Configuration constants, biological mappings, and parameters for Drosophila interpreter."""

from dataclasses import dataclass
from enum import StrEnum


class SensoryModality(StrEnum):
    """Sensory modalities corresponding to mapped receptor entry circuits."""

    SUGAR = "sugar"
    BITTER_ODOR = "bitter_odor"
    LOOMING_THREAT = "looming_threat"
    ACOUSTIC_VIBRATION = "acoustic_vibration"
    THERMOSENSATION = "thermosensation"
    HYGROSENSATION = "hygrosensation"
    CO2_STRESS = "co2_stress"


class MotorAction(StrEnum):
    """Discrete behavioral states determined by downstream descending neurons."""

    FEEDING = "FEEDING"
    ESCAPE_TAKEOFF = "ESCAPE_TAKEOFF"
    AVOIDANCE = "AVOIDANCE"
    STARTLE_FREEZE = "STARTLE_FREEZE"
    THERMAL_TAXIS = "THERMAL_TAXIS"
    GROOMING = "GROOMING"
    QUIESCENT = "QUIESCENT"


# Rich anchor descriptors for cosine similarity matching
SENSORY_ANCHORS: dict[SensoryModality, str] = {
    SensoryModality.SUGAR: (
        "sweet ripe fruit, sugar crystals, honey nectar, molasses, sucrose food source"
    ),
    SensoryModality.BITTER_ODOR: (
        "toxic chemical smell, repellent volatile odor, bitter poison, decaying rotten organic waste"
    ),
    SensoryModality.LOOMING_THREAT: (
        "rapidly approaching shadow, looming predator, hand swatting down, sudden collision hazard"
    ),
    SensoryModality.ACOUSTIC_VIBRATION: (
        "intense air blast, loud shouting voice, high-frequency sound wave, buzzing courtship wing vibration"
    ),
    SensoryModality.THERMOSENSATION: (
        "extreme heat, blazing thermal flame, hot surface, freezing cold ice frost"
    ),
    SensoryModality.HYGROSENSATION: (
        "humid water droplet, mist, extreme dry desert air, desiccation moisture gradient"
    ),
    SensoryModality.CO2_STRESS: (
        "suffocating carbon dioxide, acidic gas plume, stressed crowd chemical alarm signal"
    ),
}

# Embedding model settings
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COSINE_SIMILARITY_THRESHOLD = 0.22  # Tuned slightly lower to capture nuanced cross-modal overlap


@dataclass(frozen=True)
class SimulationParams:
    """Hyperparameters for Leaky Integrate-and-Fire (LIF) graph dynamics."""

    time_steps: int = 50
    dt_ms: float = 1.0
    tau_m_ms: float = 10.0  # Membrane time constant
    v_rest: float = -65.0  # Resting potential (mV)
    v_threshold: float = -45.0  # Firing threshold (mV)
    v_reset: float = -70.0  # Hyperpolarization reset (mV)
    current_scale: float = 30.0  # Multiplier converting normalized [0, 1] to pA


# Total node allocation for the simulation graph (expanded to 150 to accommodate pathways)
NUM_NEURONS: int = 150

# Map sensory modalities directly to input row indices in the adjacency graph
SENSORY_INDEX_MAP: dict[SensoryModality, list[int]] = {
    SensoryModality.SUGAR: [0, 1, 2, 3],  # Gustatory receptor neurons (Gr5a)
    SensoryModality.BITTER_ODOR: [4, 5, 6, 7],  # Olfactory projection neurons (toxic/aversive)
    SensoryModality.LOOMING_THREAT: [8, 9, 10, 11],  # Visual lobula projection neurons (LC4/LPLC2)
    SensoryModality.ACOUSTIC_VIBRATION: [12, 13, 14, 15],  # Johnston's organ mechanosensory (JO-A/B)
    SensoryModality.THERMOSENSATION: [16, 17, 18, 19],  # Antennal thermosensory neurons
    SensoryModality.HYGROSENSATION: [20, 21, 22, 23],  # Antennal hygrosensory sensilla
    SensoryModality.CO2_STRESS: [24, 25, 26, 27],  # Antennal CO2-sensitive neurons (Gr21a/Gr63a)
}

# Map descending motor neuron clusters to output behavioral classifiers
MOTOR_INDEX_MAP: dict[MotorAction, list[int]] = {
    MotorAction.FEEDING: [130, 131, 132],  # Proboscis extension reflex (Fdg command neurons)
    MotorAction.ESCAPE_TAKEOFF: [133, 134, 135],  # Giant Fiber descending neurons (GF / DNp01)
    MotorAction.AVOIDANCE: [136, 137, 138],  # Steering / turn descending neurons (DNb01)
    MotorAction.STARTLE_FREEZE: [139, 140, 141],  # Immobility / stop descending neurons (MAN)
    MotorAction.THERMAL_TAXIS: [142, 143, 144],  # Locomotor heading correction circuits
    MotorAction.GROOMING: [145, 146, 147],  # Antennal / eye sweep motor neurons
}