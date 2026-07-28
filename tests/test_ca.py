"""Pytest suite for technique waveform generators."""

import numpy as np
import pytest

from softpotato.technique.base import Waveform
from softpotato.technique.chrono import ca


class TestCAGenerator:
    """Verification test suite for ca waveform generator."""

    def test_ca_standard_step(self) -> None:
        """Verify standard potential step without explicit initial potential."""
        E_step = 0.5  # V
        t_tot = 5.0  # s
        dt = 0.1  # s

        wf = ca(E_step=E_step, t_tot=t_tot, dt=dt)

        assert isinstance(wf, Waveform)
        assert wf.t[0] == pytest.approx(0.0)
        assert wf.t[-1] == pytest.approx(t_tot)
        assert len(wf.t) == 51  # 5.0 / 0.1 + 1 points
        assert np.all(wf.E == E_step)

    def test_ca_with_initial_potential(self) -> None:
        """Verify step waveform with explicit E_ini at t = 0."""
        E_ini = 0.0
        E_step = 0.6
        t_tot = 2.0
        dt = 0.5

        wf = ca(E_step=E_step, t_tot=t_tot, dt=dt, E_ini=E_ini)

        assert wf.E[0] == pytest.approx(E_ini)
        assert np.all(wf.E[1:] == pytest.approx(E_step))
        assert len(wf.E) == 5

    def test_ca_invalid_duration(self) -> None:
        """Verify exception handling for zero or negative total time."""
        with pytest.raises(ValueError, match="t_tot must be positive"):
            ca(E_step=0.5, t_tot=0.0, dt=0.1)

        with pytest.raises(ValueError, match="t_tot must be positive"):
            ca(E_step=0.5, t_tot=-1.0, dt=0.1)

    def test_ca_invalid_dt(self) -> None:
        """Verify exception handling for non-positive or oversized dt."""
        with pytest.raises(ValueError, match="dt must be positive"):
            ca(E_step=0.5, t_tot=5.0, dt=0.0)

        with pytest.raises(ValueError, match="cannot be greater than total time"):
            ca(E_step=0.5, t_tot=5.0, dt=10.0)
