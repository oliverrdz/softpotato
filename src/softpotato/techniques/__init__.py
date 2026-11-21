"""Electrochemical experimental techniques and excitation waveform generators."""

from .base import BaseTechnique, ControlMode
from .chronoamperometry import Chronoamperometry
from .chronopotentiometry import Chronopotentiometry
from .pipeline import simulate, simulate_sweep
from .pulse import (
    DifferentialPulseVoltammetry,
    SampledCurrentVoltammetry,
    SquareWaveVoltammetry,
)
from .result import TechniqueResult
from .voltammetry import CyclicVoltammetry, LinearSweepVoltammetry
from .waveforms import (
    CallableSegment,
    ConstantSegment,
    CustomWaveform,
    LinearRampSegment,
    SinusoidalSegment,
    WaveformSegment,
)

__all__ = [
    "BaseTechnique",
    "CallableSegment",
    "Chronoamperometry",
    "Chronopotentiometry",
    "ConstantSegment",
    "ControlMode",
    "CustomWaveform",
    "CyclicVoltammetry",
    "DifferentialPulseVoltammetry",
    "LinearRampSegment",
    "LinearSweepVoltammetry",
    "SampledCurrentVoltammetry",
    "SinusoidalSegment",
    "SquareWaveVoltammetry",
    "TechniqueResult",
    "WaveformSegment",
    "simulate",
    "simulate_sweep",
]
