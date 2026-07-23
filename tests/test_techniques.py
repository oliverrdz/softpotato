import pytest

from softpotato.core.abcs import BaseTechnique
from softpotato.techniques import CyclicVoltammetry


def test_single_cycle_cv():
    """Verify standard single-cycle CV potential waveform."""
    cv = CyclicVoltammetry(E_start=0.0, E_vertex1=0.5, scan_rate=0.1)

    assert isinstance(cv, BaseTechnique)
    assert cv.t_span == (0.0, 10.0)

    # Key trajectory points
    assert cv(0.0) == pytest.approx(0.0)
    assert cv(2.5) == pytest.approx(0.25)
    assert cv(5.0) == pytest.approx(0.5)
    assert cv(7.5) == pytest.approx(0.25)
    assert cv(10.0) == pytest.approx(0.0)


def test_multi_cycle_cv():
    """Verify multi-cycle CV trajectory over 3 full cycles."""
    cv = CyclicVoltammetry(
        E_start=0.0, E_vertex1=0.5, E_vertex2=0.0, scan_rate=0.1, n_cycles=3
    )

    # 1 cycle = 10s => 3 cycles = 30s
    assert cv.t_span == (0.0, 30.0)

    # Cycle 1 vertices
    assert cv(5.0) == pytest.approx(0.5)
    assert cv(10.0) == pytest.approx(0.0)

    # Cycle 2 vertices
    assert cv(15.0) == pytest.approx(0.5)
    assert cv(20.0) == pytest.approx(0.0)

    # Cycle 3 vertices
    assert cv(25.0) == pytest.approx(0.5)
    assert cv(30.0) == pytest.approx(0.0)


def test_custom_vertex2_multi_cycle():
    """Verify multi-cycle CV when E_vertex2 differs from E_start."""
    cv = CyclicVoltammetry(
        E_start=0.2, E_vertex1=0.8, E_vertex2=0.0, scan_rate=0.1, n_cycles=2
    )

    # Cycle 0: 0.2 -> 0.8 (6s) -> 0.0 (8s) = 14s total
    # Cycle 1: 0.0 -> 0.8 (8s) -> 0.0 (8s) = 16s total
    # Total duration = 30s
    assert cv.t_span == (0.0, 30.0)

    assert cv(0.0) == pytest.approx(0.2)
    assert cv(6.0) == pytest.approx(0.8)
    assert cv(14.0) == pytest.approx(0.0)
    assert cv(22.0) == pytest.approx(0.8)
    assert cv(30.0) == pytest.approx(0.0)
