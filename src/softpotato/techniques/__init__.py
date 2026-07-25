from softpotato.core.abcs import BaseTechnique
from softpotato.techniques.chronoamperometry import Chronoamperometry
from softpotato.techniques.custom_waveform import CustomWaveform
from softpotato.techniques.cyclic_voltammetry import CyclicVoltammetry
from softpotato.techniques.linear_sweep_voltammetry import LinearSweepVoltammetry

__all__ = [
    "BaseTechnique",
    "Chronoamperometry",
    "CustomWaveform",
    "CyclicVoltammetry",
    "LinearSweepVoltammetry",
]
