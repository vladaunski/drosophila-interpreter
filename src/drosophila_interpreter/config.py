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