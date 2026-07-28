"""Pytest suite for technique waveform generators."""

import numpy as np
import pytest
from softpotato.technique.base import Waveform
from softpotato.technique.sweep import lsv


class TestLSVGenerator:
    """Verification test suite for lsv waveform generator."""

    def test_lsv_forward_sweep_bounds(self) -> None:
        """Verify standard forward sweep start, end, and duration."""
        E_ini = -0.5
        E_final = 0.5
        scan_rate = 0.1  # V/s

        wf = lsv(E_ini=E_ini, E_final=E_final, scan_rate=scan_rate, dE=0.01)

        assert isinstance(wf, Waveform)
        assert wf.E[0] == pytest.approx(E_ini)
        assert wf.E[-1] == pytest.approx(E_final)
        assert wf.t[0] == pytest.approx(0.0)
        assert wf.t[-1] == pytest.approx(10.0)  # Total time = 1.0 V / 0.1 V/s = 10 s

    def test_lsv_reverse_sweep(self) -> None:
        """Verify reverse sweep (anodic to cathodic) potential decay."""
        E_ini = 0.8
        E_final = 0.0
        scan_rate = 0.2  # V/s

        wf = lsv(E_ini=E_ini, E_final=E_final, scan_rate=scan_rate, dE=0.01)

        assert wf.E[0] == pytest.approx(0.8)
        assert wf.E[-1] == pytest.approx(0.0)
        assert np.all(np.diff(wf.E) < 0)  # Monotonically decreasing

    def test_lsv_linearity_and_scan_rate_slope(self) -> None:
        """Ensure potential rate dE/dt matches specified scan rate."""
        E_ini = 0.0
        E_final = 1.0
        scan_rate = 0.05  # V/s

        wf = lsv(E_ini=E_ini, E_final=E_final, scan_rate=scan_rate, dt=0.1)

        slopes = np.diff(wf.E) / np.diff(wf.t)
        np.testing.assert_allclose(slopes, scan_rate, rtol=1e-5)

    def test_lsv_custom_dt_resolution(self) -> None:
        """Test waveform length calculation under explicit dt parameter."""
        wf = lsv(E_ini=0.0, E_final=0.2, scan_rate=0.1, dt=0.5)

        # Total time = 2.0 s -> Steps = 2.0 / 0.5 = 4 intervals -> 5 points
        assert len(wf.t) == 5
        assert len(wf.E) == 5

    def test_lsv_invalid_scan_rate(self) -> None:
        """Verify error handling for zero or negative scan rates."""
        with pytest.raises(ValueError, match="strictly positive"):
            lsv(E_ini=0.0, E_final=0.5, scan_rate=0.0)

        with pytest.raises(ValueError, match="strictly positive"):
            lsv(E_ini=0.0, E_final=0.5, scan_rate=-0.1)

    def test_lsv_conflicting_step_arguments(self) -> None:
        """Verify exception when supplying both dE and dt."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            lsv(E_ini=0.0, E_final=0.5, scan_rate=0.1, dE=0.01, dt=0.1)

    def test_lsv_invalid_step_sizes(self) -> None:
        """Verify exception for negative or zero dE / dt."""
        with pytest.raises(ValueError, match="must be positive"):
            lsv(E_ini=0.0, E_final=0.5, scan_rate=0.1, dE=-0.01)

        with pytest.raises(ValueError, match="must be positive"):
            lsv(E_ini=0.0, E_final=0.5, scan_rate=0.1, dt=0.0)

    def test_lsv_zero_potential_range(self) -> None:
        """Verify graceful return when E_ini equals E_final."""
        wf = lsv(E_ini=0.2, E_final=0.2, scan_rate=0.1)
        assert len(wf.t) == 1
        assert wf.E[0] == pytest.approx(0.2)