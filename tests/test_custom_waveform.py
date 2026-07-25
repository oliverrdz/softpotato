import pytest
from pydantic import ValidationError

from softpotato.core.abcs import BaseTechnique
from softpotato.techniques import (
    Chronoamperometry,
    CustomWaveform,
    LinearSweepVoltammetry,
)


def test_custom_waveform_inheritance():
    """Verify CustomWaveform satisfies BaseTechnique interface contract."""
    ca = Chronoamperometry(E_step1=0.2, t_step1=2.0)
    lsv = LinearSweepVoltammetry(E_start=0.2, E_end=0.8, scan_rate=0.1)

    custom = CustomWaveform(techniques=[ca, lsv])

    assert isinstance(custom, BaseTechnique)
    assert custom.t_total == pytest.approx(2.0 + 6.0)  # 2.0s + 6.0s
    assert custom.t_span == (0.0, 8.0)


def test_custom_waveform_trajectory():
    """Verify trajectory evaluations across chained segments: CA -> LSV -> CA."""
    step1 = Chronoamperometry(
        E_init=0.0, E_step1=0.0, t_step1=2.0
    )  # 0.0s to 2.0s @ 0.0V
    ramp = LinearSweepVoltammetry(
        E_start=0.0, E_end=0.6, scan_rate=0.1
    )  # 2.0s to 8.0s (6s long)
    step2 = Chronoamperometry(
        E_init=0.6, E_step1=-0.2, t_step1=3.0
    )  # 8.0s to 11.0s @ -0.2V

    custom = CustomWaveform(techniques=[step1, ramp, step2])

    assert custom.t_total == pytest.approx(11.0)

    # Segment 1: Chronoamperometry hold at 0.0V
    assert custom(0.0) == pytest.approx(0.0)
    assert custom(1.0) == pytest.approx(0.0)
    assert custom(2.0) == pytest.approx(0.0)

    # Segment 2: Linear Sweep from 0.0V to 0.6V over 6 seconds
    assert custom(2.001) == pytest.approx(0.0001, abs=1e-3)
    assert custom(5.0) == pytest.approx(0.3)  # Midpoint of ramp
    assert custom(8.0) == pytest.approx(0.6)  # End of ramp

    # Segment 3: Chronoamperometry step to -0.2V
    assert custom(8.1) == pytest.approx(-0.2)
    assert custom(11.0) == pytest.approx(-0.2)
    assert custom(15.0) == pytest.approx(-0.2)  # Beyond t_total


def test_custom_waveform_validation():
    """Verify validation rules prevent empty technique sequences."""
    with pytest.raises(ValidationError):
        CustomWaveform(techniques=[])
