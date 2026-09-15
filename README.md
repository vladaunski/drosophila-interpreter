# Drosophila Connectome Sensory Interpreter

[![CI](https://github.com/vladaunski/drosophila-interpreter/actions/workflows/ci.yml/badge.svg)](https://github.com/vladaunski/drosophila-interpreter/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Package Manager](https://img.shields.io/badge/managed%20by-uv-purple.svg)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An experimental natural language to biological dynamical system pipeline. This project explores unconventional computing by treating the *Drosophila melanogaster* connectome as an exotic dynamical processor. 

User text is projected into semantic embeddings, routed as current injections into mapped receptor clusters (gustatory vs. visual looming circuits), and propagated across the fly's sparse brain graph via a leaky integrate-and-fire simulation. Downstream descending motor states are then classified into deterministic behavioral outcomes.

---

## Architecture Overview

```text
 [ User Text Input ]
         │
         ▼
 ┌───────────────────────────┐
 │      Semantic Router      │  (sentence-transformers: all-MiniLM-L6-v2)
 │  Cosine Similarity Filter │  Maps semantic distance to sensory anchors
 └─────────────┬─────────────┘
               │ Scaled Current Injections (I_ext)
               ▼
 ┌───────────────────────────┐
 │      Connectome Core      │  (scipy.sparse: Leaky Integrate-and-Fire)
 │   Topological Simulation  │  Propagates spikes across synaptic sub-graph
 └─────────────┬─────────────┘
               │ Membrane Potentials (V ∈ R^N)
               ▼
 ┌───────────────────────────┐
 │    Motor State Readout    │  Inspects Descending Neurons (DNs)
 │   Threshold Classifier    │  (Giant Fiber vs. Proboscis vs. Steering)
 └─────────────┬─────────────┘
               │
               ▼
   [ "HUNGRY" | "PANICKED" | "EXPLORING" | "QUIESCENT" | ... ]