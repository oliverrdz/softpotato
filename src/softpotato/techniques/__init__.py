"""Electrochemical experimental techniques and excitation waveform generators."""

from .base import BaseTechnique, ControlMode
from .chronoamperometry import Chronoamperometry
from .chronopotentiometry import Chronopotentiometry
from .pulse import (
    DifferentialPulseVoltammetry,
    SampledCurrentVoltammetry,
    SquareWaveVoltammetry,
)
from .rde import (
    RDEConvectionTerm,
    RDEModel,
    RDESweep,
    RDESweepResult,
    RotatingDiskElectrode,
    simulate_rde_sweep,
)
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
    "RDEConvectionTerm",
    "RDEModel",
    "RDESweep",
    "RDESweepResult",
    "RotatingDiskElectrode",
    "SampledCurrentVoltammetry",
    "SinusoidalSegment",
    "SquareWaveVoltammetry",
    "WaveformSegment",
    "simulate_rde_sweep",
]
