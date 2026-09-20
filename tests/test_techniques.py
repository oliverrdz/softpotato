"""
Unit tests for electrochemical techniques and waveform generators.
"""

from __future__ import annotations

import numpy as np
import pytest

from softpotato.techniques import (
    CyclicVoltammetry,
    Technique,
)


def test_cyclic_voltammetry_initialization() -> None:
    """Tests proper initialization of CyclicVoltammetry."""
    cv = CyclicVoltammetry(
        E_initial=0.5,
        E_vertex1=-0.5,
        scan_rate=0.1,
        n_sweeps=2,
        dE=0.001,
    )

    assert isinstance(cv, Technique)
    assert cv.E_initial == 0.5
    assert cv.E_vertex1 == -0.5
    assert cv.E_vertex2 == 0.5
    assert cv.scan_rate == 0.1
    assert cv.n_sweeps == 2
    assert cv.dE == 0.001
    assert np.isclose(cv.dt, 0.01)

    # First point is E_initial, turning point is E_vertex1, end point is E_vertex2
    assert np.isclose(cv.potential[0], 0.5)
    assert np.isclose(np.min(cv.potential), -0.5)
    assert np.isclose(cv.potential[-1], 0.5)

    # Time array check
    assert len(cv.time) == len(cv.potential)
    assert np.isclose(cv.time[0], 0.0)
    assert np.isclose(cv.duration, cv.time[-1])


def test_cyclic_voltammetry_custom_vertex2() -> None:
    """Tests CyclicVoltammetry with custom E_vertex2."""
    cv = CyclicVoltammetry(
        E_initial=0.2,
        E_vertex1=-0.4,
        scan_rate=0.05,
        n_sweeps=2,
        dE=0.002,
        E_vertex2=0.6,
    )
    assert cv.E_vertex2 == 0.6
    assert np.isclose(cv.potential[-1], 0.6)


def test_cyclic_voltammetry_validation() -> None:
    """Tests input validation for CyclicVoltammetry."""
    # Non-numeric potentials
    with pytest.raises(TypeError, match="real number"):
        CyclicVoltammetry(E_initial="0.5", E_vertex1=-0.5, scan_rate=0.1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="real number"):
        CyclicVoltammetry(E_initial=0.5, E_vertex1=None, scan_rate=0.1)  # type: ignore[arg-type]

    # Identical vertices
    with pytest.raises(ValueError, match="distinct"):
        CyclicVoltammetry(E_initial=0.5, E_vertex1=0.5, scan_rate=0.1)

    # Non-positive scan rate
    with pytest.raises(ValueError, match="positive"):
        CyclicVoltammetry(E_initial=0.5, E_vertex1=-0.5, scan_rate=0.0)
    with pytest.raises(ValueError, match="positive"):
        CyclicVoltammetry(E_initial=0.5, E_vertex1=-0.5, scan_rate=-0.1)

    # Invalid n_sweeps
    with pytest.raises(ValueError, match="at least 1"):
        CyclicVoltammetry(E_initial=0.5, E_vertex1=-0.5, scan_rate=0.1, n_sweeps=0)
    with pytest.raises(TypeError, match="integer"):
        CyclicVoltammetry(E_initial=0.5, E_vertex1=-0.5, scan_rate=0.1, n_sweeps=1.5)  # type: ignore[arg-type]

    # Non-positive dE
    with pytest.raises(ValueError, match="positive"):
        CyclicVoltammetry(E_initial=0.5, E_vertex1=-0.5, scan_rate=0.1, dE=-0.001)
