"""Configuration constants and biological mappings for the Drosophila connectome interpreter."""

from enum import Enum
from typing import Final


class SensoryModality(str, Enum):
    """Sensory input modalities for the fly nervous system."""

    SUGAR = "sugar"
    BITTER_ODOR = "bitter_odor"
    LOOMING_THREAT = "looming_threat"
    ACOUSTIC_VIBRATION = "acoustic_vibration"
    THERMOSENSATION = "thermosensation"
    HYGROSENSATION = "hygrosensation"
    CO2_STRESS = "co2_stress"


# Multi-anchor clusters per modality for maximum semantic capture
SENSORY_ANCHORS: Final[dict[SensoryModality, list[str]]] = {
    SensoryModality.SUGAR: [
        "sweet ripe fruit, sugar crystals, glucose, sucrose, nectar drop, feeding reward",
        "delicious sweet syrup, honey, molasses, nutrient-rich sap",
    ],
    SensoryModality.BITTER_ODOR: [
        "pungent bitter toxin, rotten repulsive decayed stench, foul spoiled matter",
        "noxious chemical deterrent, caustic alkaloid, distasteful repelling odor",
    ],
    SensoryModality.LOOMING_THREAT: [
        "fast approaching dark shadow, incoming descending predator, swatting motion",
        "rapid optical expansion, imminent collision, sudden looming obstacle",
    ],
    SensoryModality.ACOUSTIC_VIBRATION: [
        "loud shouting, screaming, deafening roar, sonic blast, aggressive yelling",
        "airborne sound waves, courtship song pulses, wing hum, buzzing oscillation",
        "mechanical substrate vibrations, physical auditory disturbance, tremors",
    ],
    SensoryModality.THERMOSENSATION: [
        "extreme burning heat, searing hot thermal spike, scald, high temperature",
        "freezing icy cold, chilled thermal drop, frost, sub-zero temperature",
    ],
    SensoryModality.HYGROSENSATION: [
        "ambient moisture, water droplets, damp wet humidity, rain precipitation",
        "arid desiccating dryness, parched atmosphere, zero moisture evaporation",
    ],
    SensoryModality.CO2_STRESS: [
        "elevated carbon dioxide concentration, suffocating gas, hypoxic atmosphere",
        "conspecific alarm pheromone, crowd distress signal, danger warning plume",
    ],
}

# Empirical signal boundaries for MiniLM embedding space
COSINE_SIMILARITY_THRESHOLD: Final[float] = 0.22
COSINE_SIMILARITY_SATURATION: Final[float] = 0.65

# =====================================================================
# Leaky Integrate-and-Fire (LIF) Biophysical Parameters
# =====================================================================
V_REST: float = -52.0       # Resting membrane potential (mV)
V_THRESH: float = -45.0     # Action potential threshold (mV)
V_RESET: float = -55.0      # Post-spike reset potential (mV)
TAU_M: float = 20.0         # Membrane time constant (ms)
DT: float = 0.5             # Numerical integration timestep (ms)
REFRACTORY_PERIOD: float = 2.0  # Refractory period post-spike (ms)

# Electrical scaling for sensory injection (pA or normalized current unit)
# A full 1.0 signal injects enough current to reliably elicit action potentials
I_SENSORY_MAX: float = 12.0

# Base synaptic conductance scale
SYNAPSE_WEIGHT_SCALE: float = 0.05

# Matches specific morphological annotations in the node metadata table
INGRESS_CELL_TYPES: dict[SensoryModality, tuple[str, ...]] = {
    SensoryModality.SUGAR: (
        "GRN_sugar",
        "Gr5a",
        "Gr64f",
    ),
    SensoryModality.BITTER_ODOR: (
        "GRN_bitter",
        "Gr66a",
        "DM2",
        "DL5",
    ),
    SensoryModality.LOOMING_THREAT: (
        "LC4",
        "LPLC2",
    ),
    SensoryModality.ACOUSTIC_VIBRATION: (
        "JO-AB",
        "AMMC",
        "a1",
    ),
    SensoryModality.THERMOSENSATION: (
        "TRPA1",
        "Gr28b(D)",
        "AC_neuron",
    ),
    SensoryModality.HYGROSENSATION: (
        "Ir40a",
        "Ir93a",
    ),
    SensoryModality.CO2_STRESS: (
        "Gr21a",
        "Gr63a",
        "V_glomerulus",
    ),
}

# Target command circuits and descending neurons for decoder readout
MOTOR_CIRCUIT_TAGS: dict[str, tuple[str, ...]] = {
    "ESCAPE_TAKEOFF": (
        "Giant_Fiber",
        "GF",
        "DNp01",
        "DNp06",
    ),
    "FEEDING_PROBOSCIS": (
        "FIP",
        "MN9",
        "SEZ_DN",
    ),
    "GROOMING_ANTENNA": (
        "aDN1",
        "aDN2",
    ),
    "HALT_FREEZE": (
        "DNb01",
        "DNp09",
    ),
    "BACKWARD_WALK": (
        "MDN",
        "DNp24",
    ),
    "COURTSHIP_WING_SONG": (
        "pIP10",
        "dPR1",
        "P1",
    ),
}