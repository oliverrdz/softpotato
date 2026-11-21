"""Electrochemical experimental techniques and excitation waveform generators."""

from .base import BaseTechnique, ControlMode
from .chronoamperometry import Chronoamperometry
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
    "ConstantSegment",
    "ControlMode",
    "CustomWaveform",
    "CyclicVoltammetry",
    "LinearRampSegment",
    "LinearSweepVoltammetry",
    "SinusoidalSegment",
    "WaveformSegment",
]
