import pytest
from pydantic import ValidationError

from softpotato.core.abcs import BaseTechnique
from softpotato.techniques import LinearSweepVoltammetry


def test_lsv_inheritance_and_interface():
    """Verify LinearSweepVoltammetry implements BaseTechnique contract."""
    lsv = LinearSweepVoltammetry(E_start=0.0, E_end=0.5, scan_rate=0.1)
    assert isinstance(lsv, BaseTechnique)
    assert lsv.t_total == pytest.approx(5.0)
    assert lsv.t_span == (0.0, 5.0)


def test_anodic_lsv_potential_trajectory():
    """Verify trajectory for an anodic sweep (E_start < E_end)."""
    lsv = LinearSweepVoltammetry(E_start=-0.2, E_end=0.8, scan_rate=0.1)
    # total duration = 1.0 V / 0.1 V/s = 10.0 s

    assert lsv.t_total == pytest.approx(10.0)
    assert lsv(-1.0) == pytest.approx(-0.2)  # t <= 0
    assert lsv(0.0) == pytest.approx(-0.2)
    assert lsv(2.5) == pytest.approx(0.05)  # -0.2 + 0.1 * 2.5
    assert lsv(5.0) == pytest.approx(0.3)  # Midpoint
    assert lsv(10.0) == pytest.approx(0.8)  # Boundary
    assert lsv(15.0) == pytest.approx(0.8)  # t >= t_total


def test_cathodic_lsv_potential_trajectory():
    """Verify trajectory for a cathodic sweep (E_start > E_end)."""
    lsv = LinearSweepVoltammetry(E_start=0.6, E_end=0.0, scan_rate=0.2)
    # total duration = 0.6 V / 0.2 V/s = 3.0 s

    assert lsv.t_total == pytest.approx(3.0)
    assert lsv(0.0) == pytest.approx(0.6)
    assert lsv(1.5) == pytest.approx(0.3)  # Midpoint
    assert lsv(3.0) == pytest.approx(0.0)  # End
    assert lsv(5.0) == pytest.approx(0.0)


def test_lsv_pydantic_validation():
    """Verify validation rules for scan_rate."""
    # Negative scan rate should fail
    with pytest.raises(ValidationError):
        LinearSweepVoltammetry(E_start=0.0, E_end=0.5, scan_rate=-0.1)

    # Zero scan rate should fail
    with pytest.raises(ValidationError):
        LinearSweepVoltammetry(E_start=0.0, E_end=0.5, scan_rate=0.0)
