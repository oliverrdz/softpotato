"""Electrochemical experimental techniques and excitation waveform generators."""

from .base import BaseTechnique, ControlMode
from .chronoamperometry import Chronoamperometry
from .chronopotentiometry import Chronopotentiometry
from .diagnostics import (
    CVDiagnostics,
    NicholsonDiagnostics,
    PeakInfo,
    PeakPolarity,
    SandDiagnostics,
    analyze_cottrell,
    analyze_koutecky_levich,
    analyze_levich,
    analyze_nicholson,
    analyze_randles_sevcik,
    analyze_sand,
    analyze_tafel,
    find_peaks,
)
from .pipeline import simulate, simulate_sweep
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
    "CVDiagnostics",
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
    "NicholsonDiagnostics",
    "PeakInfo",
    "PeakPolarity",
    "RDEConvectionTerm",
    "RDEModel",
    "RDESweep",
    "RDESweepResult",
    "RotatingDiskElectrode",
    "SampledCurrentVoltammetry",
    "SandDiagnostics",
    "SinusoidalSegment",
    "SquareWaveVoltammetry",
    "TechniqueResult",
    "WaveformSegment",
    "analyze_cottrell",
    "analyze_koutecky_levich",
    "analyze_levich",
    "analyze_nicholson",
    "analyze_randles_sevcik",
    "analyze_sand",
    "analyze_tafel",
    "find_peaks",
    "simulate",
    "simulate_rde_sweep",
    "simulate_sweep",
]
