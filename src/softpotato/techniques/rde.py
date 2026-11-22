"""Rotating Disk Electrode (RDE) hydrodynamic technique and sweep utilities."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
from scipy import sparse

from softpotato.analytical.hydrodynamics import (
    levich,
    nernst_diffusion_layer,
    rad_s_to_rpm,
    rpm_to_rad_s,
)
from softpotato.geometry import PlanarGeometry
from softpotato.grid import (
    ExponentialMesh,
    GeometricExpandingMesh,
    Mesh1D,
    Uniform1DMesh,
)
from softpotato.reactions.heterogeneous import (
    HeterogeneousKinetics,
    NernstianEquilibrium,
    create_electrochemical_bcs,
)
from softpotato.solver import (
    BoundaryCondition,
    DiffusionProblem,
    DirichletBC,
    SolverResult,
    get_solver,
)

from .base import (
    BaseTechnique,
    PlotSpec,
    _validate_finite_real,
    _validate_positive_finite,
    _validate_positive_int,
)
from .voltammetry import LinearSweepVoltammetry

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

RDEModel = Literal["nernst_layer", "convective_diffusion"]

_VALID_RDE_MODELS: tuple[str, ...] = ("nernst_layer", "convective_diffusion")
_VON_KARMAN_A: float = 0.51023
_DEFAULT_KINEMATIC_VISCOSITY: float = 1e-6
_DEFAULT_CONVECTIVE_DOMAIN_FACTOR: float = 3.0


def _resolve_rde_model(model: object) -> RDEModel:
    """Validate that model is 'nernst_layer' or 'convective_diffusion'."""
    if not isinstance(model, str) or model not in _VALID_RDE_MODELS:
        raise ValueError(f"model must be one of {_VALID_RDE_MODELS}, got {model!r}.")
    return model  # type: ignore[return-value]


def _resolve_scalar_rotation(
    omega: float | None,
    rpm: float | None,
) -> tuple[float, float]:
    """Validate mutually exclusive positive rotation rate and return (omega, rpm)."""
    if omega is None and rpm is None:
        raise ValueError(
            "Either angular velocity 'omega' (rad/s) or rotational speed 'rpm' "
            "must be provided."
        )
    if omega is not None and rpm is not None:
        raise ValueError("Specify either 'omega' or 'rpm', but not both.")

    if omega is not None:
        omega_val = _validate_positive_finite(omega, "omega")
        rpm_val = float(rad_s_to_rpm(omega_val))
        return omega_val, rpm_val

    assert rpm is not None
    rpm_val = _validate_positive_finite(rpm, "rpm")
    omega_val = float(rpm_to_rad_s(rpm_val))
    return omega_val, rpm_val


def _resolve_rotation_sequence(
    omegas: Sequence[float] | ArrayLike | None,
    rpms: Sequence[float] | ArrayLike | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate mutually exclusive 1D positive rotation-rate sequences."""
    if omegas is None and rpms is None:
        raise ValueError("Either 'omegas' (rad/s) or 'rpms' must be provided.")
    if omegas is not None and rpms is not None:
        raise ValueError("Specify either 'omegas' or 'rpms', but not both.")

    raw = omegas if omegas is not None else rpms
    name = "omegas" if omegas is not None else "rpms"
    if isinstance(raw, (str, bytes)):
        raise TypeError(f"'{name}' must be a numeric sequence.")

    arr = np.asarray(raw, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(
            f"'{name}' must be a non-empty 1D sequence, got shape {arr.shape}."
        )
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(
            f"All values in '{name}' must be finite and strictly positive (> 0)."
        )

    if omegas is not None:
        omega_arr = arr.copy()
        rpm_arr = np.asarray(rad_s_to_rpm(omega_arr), dtype=float)
    else:
        rpm_arr = arr.copy()
        omega_arr = np.asarray(rpm_to_rad_s(rpm_arr), dtype=float)
    return omega_arr, rpm_arr


def _resolve_waveform(
    waveform: BaseTechnique | None,
    *,
    E_start: float | None,
    E_end: float | None,
    scan_rate: float | None,
    dE: float | None,
    n_points: int | None,
) -> BaseTechnique:
    """Resolve an explicit potentiostatic waveform or build a default LSV waveform."""
    lsv_args = (E_start, E_end, scan_rate, dE, n_points)
    has_lsv_args = any(x is not None for x in lsv_args)

    if waveform is not None:
        if has_lsv_args:
            raise ValueError(
                "Cannot specify both 'waveform' and direct LSV parameters "
                "('E_start', 'E_end', 'scan_rate', 'dE', 'n_points')."
            )
        if not isinstance(waveform, BaseTechnique):
            raise TypeError(
                "waveform must be an instance of BaseTechnique, "
                f"got {type(waveform).__name__}."
            )
        if waveform.mode != "potentiostatic":
            raise ValueError(
                "RotatingDiskElectrode requires a potentiostatic waveform, "
                f"got mode={waveform.mode!r}."
            )
        return waveform

    if E_start is None or E_end is None or scan_rate is None:
        raise ValueError(
            "Must provide either 'waveform' or direct LSV parameters "
            "('E_start', 'E_end', 'scan_rate')."
        )
    return LinearSweepVoltammetry(
        E_start=E_start,
        E_end=E_end,
        scan_rate=scan_rate,
        dE=dE,
        n_points=n_points,
    )


def _build_convection_matrix(
    grid: np.ndarray,
    omega: float,
    kinematic_viscosity: float,
    diffusivity: float,
) -> sparse.csc_matrix:
    r"""Assemble the 1D axial advection operator :math:`-v_x(x) \partial/\partial x`.

    Uses Patankar's hybrid central/upwind finite-difference stencil on
    non-uniform 1D grids for the von Kármán axial velocity:

    .. math::

        v_x(x) = -0.51023 \, \omega^{3/2} \nu^{-1/2} x^2 \le 0

    For interior nodes where the local cell Péclet number
    :math:`\mathrm{Pe}_j = |v_x(x_j)| h_R / D \le 2`, a second-order
    non-uniform central difference is used. Where :math:`\mathrm{Pe}_j > 2`,
    first-order upwind differencing (biased toward :math:`x_{j+1}` since
    :math:`v_x \le 0`) guarantees a monotone M-matrix without spurious
    far-field oscillations.
    """
    n_pts = len(grid)
    # u(x) = -v_x(x) >= 0
    coeff = _VON_KARMAN_A * (omega**1.5) * (kinematic_viscosity ** (-0.5))
    u = coeff * (grid**2)

    sub = np.zeros(n_pts - 1, dtype=float)
    diag = np.zeros(n_pts, dtype=float)
    sup = np.zeros(n_pts - 1, dtype=float)

    # Interior nodes j = 1 .. n_pts - 2
    h_l = grid[1:-1] - grid[:-2]
    h_r = grid[2:] - grid[1:-1]
    u_int = u[1:-1]
    peclet = u_int * h_r / diffusivity

    use_central = peclet <= 2.0

    # Central difference (2nd-order non-uniform):
    # dc/dx|_j = -h_r/(h_l*(h_l+h_r)) c_{j-1} + (h_r-h_l)/(h_l*h_r) c_j + h_l/(h_r*(h_l+h_r)) c_{j+1}
    c_sub = -u_int * h_r / (h_l * (h_l + h_r))
    c_diag = u_int * (h_r - h_l) / (h_l * h_r)
    c_sup = u_int * h_l / (h_r * (h_l + h_r))

    # Upwind difference (1st-order, v_x <= 0 => flow from right to left):
    # dc/dx|_j = (c_{j+1} - c_j) / h_r
    u_sub = np.zeros_like(u_int)
    u_diag = -u_int / h_r
    u_sup = u_int / h_r

    sub[:-1] = np.where(use_central, c_sub, u_sub)
    diag[1:-1] = np.where(use_central, c_diag, u_diag)
    sup[1:] = np.where(use_central, c_sup, u_sup)

    # Right boundary j = n_pts - 1 (backward difference if Neumann; unused if Dirichlet)
    h_last = grid[-1] - grid[-2]
    sub[-1] = -u[-1] / h_last
    diag[-1] = u[-1] / h_last

    return sparse.diags(
        [sub, diag, sup],
        offsets=[-1, 0, 1],
        shape=(n_pts, n_pts),
        format="csc",
    )


class RDEConvectionTerm:
    r"""Axial convective transport operator :math:`-v_x(x) \partial c/\partial x` for RDE.

    Implements the ``reactions`` protocol expected by
    :class:`~softpotato.solver.DiffusionProblem` and
    :class:`~softpotato.solver.ScipyIVPSolver`, optionally composing with an
    existing homogeneous reaction network or callable while supplying an exact
    sparse analytical Jacobian when possible.

    Parameters
    ----------
    omega : float
        Angular rotation velocity :math:`\omega > 0` in :math:`\text{rad}/\text{s}`.
    kinematic_viscosity : float
        Kinematic viscosity :math:`\nu > 0` in :math:`\text{m}^2/\text{s}`.
    diffusivity : float or Mapping[str, float]
        Scalar diffusion coefficient or per-species diffusivity mapping used
        for local Péclet number evaluation.
    base_reactions : Any, optional
        Optional homogeneous reaction term (``ReactionNetwork``,
        ``HomogeneousReaction``, ``Mechanism``, or callable) to superpose with
        axial convection.
    """

    def __init__(
        self,
        omega: float,
        kinematic_viscosity: float,
        diffusivity: float | Mapping[str, float],
        *,
        base_reactions: Any = None,
    ) -> None:
        self._omega = _validate_positive_finite(omega, "omega")
        self._kinematic_viscosity = _validate_positive_finite(
            kinematic_viscosity, "kinematic_viscosity"
        )
        if isinstance(diffusivity, Mapping):
            self._diffusivity_map: dict[str, float] | None = {
                sp: _validate_positive_finite(d_val, f"diffusivity[{sp!r}]")
                for sp, d_val in diffusivity.items()
            }
            self._default_diffusivity: float | None = None
        else:
            self._diffusivity_map = None
            self._default_diffusivity = _validate_positive_finite(
                diffusivity, "diffusivity"
            )
        self._base_reactions = base_reactions
        self._matrix_cache: dict[tuple[str, int, float, float], sparse.csc_matrix] = {}

        # Expose analytical jacobian only when base_reactions is absent or provides one
        if base_reactions is None:
            self._can_provide_jacobian = True
        else:
            has_local = callable(getattr(base_reactions, "local_jacobian", None))
            has_jac = callable(getattr(base_reactions, "jacobian", None))
            self._can_provide_jacobian = has_local or has_jac

        if not self._can_provide_jacobian:
            # Shadow the class method on this instance so ScipyIVPSolver falls back
            # to numerical finite-difference Jacobian instead of an incomplete one.
            self.jacobian = None  # type: ignore[assignment,method-assign]

    def _get_diffusivity(self, species_name: str) -> float:
        if self._diffusivity_map is not None:
            if species_name in self._diffusivity_map:
                return self._diffusivity_map[species_name]
            return next(iter(self._diffusivity_map.values()))
        assert self._default_diffusivity is not None
        return self._default_diffusivity

    def _get_operator(self, species_name: str, grid: np.ndarray) -> sparse.csc_matrix:
        key = (species_name, len(grid), float(grid[0]), float(grid[-1]))
        op = self._matrix_cache.get(key)
        if op is None:
            d_val = self._get_diffusivity(species_name)
            op = _build_convection_matrix(
                grid, self._omega, self._kinematic_viscosity, d_val
            )
            self._matrix_cache[key] = op
        return op

    def __call__(
        self,
        c_dict: dict[str, np.ndarray],
        grid: np.ndarray,
        t: float,
    ) -> dict[str, np.ndarray]:
        """Evaluate convective rate (plus optional homogeneous reactions) for each species."""
        grid_arr = np.asarray(grid, dtype=float)
        out: dict[str, np.ndarray] = {}
        for sp, c_arr in c_dict.items():
            op = self._get_operator(sp, grid_arr)
            out[sp] = np.asarray(op @ c_arr, dtype=float)

        if self._base_reactions is not None:
            base_rates = self._base_reactions(c_dict, grid_arr, t)
            for sp, rate in base_rates.items():
                if sp in out:
                    out[sp] = out[sp] + np.asarray(rate, dtype=float)
                else:
                    out[sp] = np.asarray(rate, dtype=float)
        return out

    def jacobian(
        self,
        c_dict: dict[str, np.ndarray],
        grid: np.ndarray,
        t: float,
        species: Sequence[str] | None = None,
    ) -> sparse.csc_matrix:
        """Assemble full sparse Jacobian ``(S*N, S*N)`` for convection + reactions."""
        grid_arr = np.asarray(grid, dtype=float)
        species_names = list(species) if species is not None else list(c_dict.keys())
        n_pts = len(grid_arr)
        blocks = [self._get_operator(sp, grid_arr) for sp in species_names]
        j_conv = sparse.block_diag(blocks, format="csc")

        if self._base_reactions is None:
            return j_conv

        local_jac_fn = getattr(self._base_reactions, "local_jacobian", None)
        if callable(local_jac_fn):
            local_j = np.asarray(
                local_jac_fn(c_dict, grid_arr, t, species=species_names), dtype=float
            )
            if local_j.ndim == 2:
                local_j = np.broadcast_to(
                    local_j[:, :, np.newaxis], (*local_j.shape, n_pts)
                )
            rows: list[np.ndarray] = []
            cols: list[np.ndarray] = []
            data: list[np.ndarray] = []
            j_idx = np.arange(n_pts)
            for i in range(len(species_names)):
                for k in range(len(species_names)):
                    vals = local_j[i, k, :]
                    if np.any(vals != 0.0):
                        rows.append(i * n_pts + j_idx)
                        cols.append(k * n_pts + j_idx)
                        data.append(vals)
            if data:
                total = len(species_names) * n_pts
                j_base = sparse.coo_matrix(
                    (
                        np.concatenate(data),
                        (np.concatenate(rows), np.concatenate(cols)),
                    ),
                    shape=(total, total),
                ).tocsc()
                return (j_conv + j_base).tocsc()
            return j_conv

        rxn_jac_fn = getattr(self._base_reactions, "jacobian", None)
        if callable(rxn_jac_fn):
            try:
                raw_j = rxn_jac_fn(c_dict, grid_arr, t, species=species_names)
            except TypeError:
                raw_j = rxn_jac_fn(c_dict, grid_arr, t)

            if isinstance(raw_j, Mapping):
                total = len(species_names) * n_pts
                sp_index = {sp: idx for idx, sp in enumerate(species_names)}
                rows_d: list[np.ndarray] = []
                cols_d: list[np.ndarray] = []
                data_d: list[np.ndarray] = []
                j_idx = np.arange(n_pts)
                for (sp_i, sp_k), block in raw_j.items():
                    if sp_i not in sp_index or sp_k not in sp_index:
                        continue
                    arr = np.broadcast_to(np.asarray(block, dtype=float), (n_pts,))
                    if np.any(arr != 0.0):
                        rows_d.append(sp_index[sp_i] * n_pts + j_idx)
                        cols_d.append(sp_index[sp_k] * n_pts + j_idx)
                        data_d.append(arr)
                if data_d:
                    j_base = sparse.coo_matrix(
                        (
                            np.concatenate(data_d),
                            (np.concatenate(rows_d), np.concatenate(cols_d)),
                        ),
                        shape=(total, total),
                    ).tocsc()
                    return (j_conv + j_base).tocsc()
                return j_conv

            return (j_conv + sparse.csc_matrix(raw_j)).tocsc()

        return j_conv


class RotatingDiskElectrode(BaseTechnique):
    r"""Rotating Disk Electrode (RDE) hydrodynamic electrochemical technique.

    Wraps an underlying potentiostatic excitation waveform
    (:class:`~softpotato.techniques.LinearSweepVoltammetry`,
    :class:`~softpotato.techniques.CyclicVoltammetry`,
    :class:`~softpotato.techniques.Chronoamperometry`, or any potentiostatic
    :class:`~softpotato.techniques.BaseTechnique`) and couples it with
    rotation-dependent hydrodynamic mass transport:

    - ``model="nernst_layer"`` (default): Nernst stagnant diffusion layer
      approximation on :math:`x \in [0, \delta_N]` with bulk Dirichlet boundary
      condition at :math:`x = \delta_N`, where

      .. math::

          \delta_N = 1.610 \, D^{1/3} \nu^{1/6} \omega^{-1/2}

    - ``model="convective_diffusion"``: Full Levich axial convective-diffusion
      PDE on :math:`x \in [0, k \delta_N]` (default :math:`k = 3.0`) including
      the von Kármán axial fluid velocity toward the disk:

      .. math::

          v_x(x) = -0.51023 \, \omega^{3/2} \nu^{-1/2} x^2

    Parameters
    ----------
    waveform : BaseTechnique, optional
        Underlying potentiostatic excitation waveform (e.g., LSV, CV, or CA).
        Mutually exclusive with direct LSV parameters
        (``E_start``, ``E_end``, ``scan_rate``).
    omega : float, optional
        Angular rotation velocity :math:`\omega > 0` in :math:`\text{rad}/\text{s}`.
        Mutually exclusive with ``rpm``.
    rpm : float, optional
        Rotational speed in revolutions per minute (:math:`\text{rpm} > 0`),
        converted to :math:`\omega` via
        :func:`~softpotato.analytical.hydrodynamics.rpm_to_rad_s`.
        Mutually exclusive with ``omega``.
    kinematic_viscosity : float, default 1e-6
        Electrolyte kinematic viscosity :math:`\nu > 0` in :math:`\text{m}^2/\text{s}`
        (default ``1e-6`` for water at 298 K).
    model : {'nernst_layer', 'convective_diffusion'}, default 'nernst_layer'
        Hydrodynamic mass-transport model.
    E_start : float, optional
        Initial potential in Volts when constructing a default LSV waveform.
    E_end : float, optional
        Final potential in Volts when constructing a default LSV waveform.
    scan_rate : float, optional
        Scan rate magnitude in V/s when constructing a default LSV waveform.
    dE : float, optional
        Default potential step resolution in Volts for the default LSV waveform.
    n_points : int, optional
        Default number of time grid points for the default LSV waveform.
    domain_factor : float, optional
        Multiplier :math:`k > 0` applied to :math:`\delta_N` when constructing
        spatial grids via :meth:`build_grid`. Defaults to ``1.0`` for
        ``model="nernst_layer"`` and ``3.0`` for ``model="convective_diffusion"``.

    Examples
    --------
    >>> from softpotato.techniques import RotatingDiskElectrode
    >>> rde = RotatingDiskElectrode(
    ...     E_start=0.5, E_end=-0.5, scan_rate=0.05, rpm=1600
    ... )
    >>> round(rde.omega, 2)
    167.55
    >>> round(rde.diffusion_layer_thickness(D=1e-9) * 1e6, 2)
    12.44
    """

    def __init__(
        self,
        waveform: BaseTechnique | None = None,
        *,
        omega: float | None = None,
        rpm: float | None = None,
        kinematic_viscosity: float = _DEFAULT_KINEMATIC_VISCOSITY,
        model: RDEModel = "nernst_layer",
        E_start: float | None = None,
        E_end: float | None = None,
        scan_rate: float | None = None,
        dE: float | None = None,
        n_points: int | None = None,
        domain_factor: float | None = None,
    ) -> None:
        self._mode = "potentiostatic"
        self._waveform = _resolve_waveform(
            waveform,
            E_start=E_start,
            E_end=E_end,
            scan_rate=scan_rate,
            dE=dE,
            n_points=n_points,
        )
        self._omega, self._rpm = _resolve_scalar_rotation(omega, rpm)
        self._kinematic_viscosity = _validate_positive_finite(
            kinematic_viscosity, "kinematic_viscosity"
        )
        self._model: RDEModel = _resolve_rde_model(model)
        if domain_factor is None:
            self._domain_factor = (
                1.0
                if self._model == "nernst_layer"
                else _DEFAULT_CONVECTIVE_DOMAIN_FACTOR
            )
        else:
            self._domain_factor = _validate_positive_finite(
                domain_factor, "domain_factor"
            )

    @property
    def waveform(self) -> BaseTechnique:
        """Underlying potentiostatic excitation waveform."""
        return self._waveform

    @property
    def omega(self) -> float:
        r"""Angular rotation velocity :math:`\omega` in :math:`\text{rad}/\text{s}`."""
        return self._omega

    @property
    def rpm(self) -> float:
        """Rotational speed in revolutions per minute (RPM)."""
        return self._rpm

    @property
    def kinematic_viscosity(self) -> float:
        r"""Kinematic viscosity :math:`\nu` in :math:`\text{m}^2/\text{s}`."""
        return self._kinematic_viscosity

    @property
    def nu(self) -> float:
        r"""Alias for :attr:`kinematic_viscosity` (:math:`\nu`)."""
        return self._kinematic_viscosity

    @property
    def model(self) -> RDEModel:
        """Hydrodynamic mass-transport model (``'nernst_layer'`` or ``'convective_diffusion'``)."""
        return self._model

    @property
    def domain_factor(self) -> float:
        r"""Domain multiplier :math:`k` such that :math:`x_{\max} = k \delta_N`."""
        return self._domain_factor

    @property
    def duration(self) -> float:
        """Total experiment duration in seconds (delegated to :attr:`waveform`)."""
        return self._waveform.duration

    @property
    def event_times(self) -> tuple[float, ...]:
        """Sorted interior event timestamps (delegated to :attr:`waveform`)."""
        return self._waveform.event_times

    def _technique_plots(self) -> tuple[PlotSpec, ...]:
        base_specs = self._waveform._technique_plots()
        if not base_specs:
            return ()
        first = base_specs[0]
        rde_title = (
            first.title if first.title.startswith("RDE ") else f"RDE {first.title}"
        )
        updated_first = PlotSpec(
            kind=first.kind,
            title=rde_title,
            x_label=first.x_label,
            y_label=first.y_label,
            description=first.description,
        )
        return (updated_first, *base_specs[1:])

    @overload
    def excitation(self, t: float) -> float: ...

    @overload
    def excitation(self, t: np.ndarray) -> np.ndarray: ...

    @overload
    def excitation(self, t: float | ArrayLike) -> float | np.ndarray: ...

    def excitation(self, t: float | ArrayLike) -> float | np.ndarray:
        """Evaluate the underlying potential waveform :math:`E(t)` at time ``t``."""
        return self._waveform.excitation(t)

    def time_grid(
        self,
        n_points: int | None = None,
        dt: float | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Generate evaluation timestamps (delegated to :attr:`waveform`)."""
        if kwargs:
            return self._waveform.time_grid(n_points=n_points, dt=dt, **kwargs)
        return self._waveform.time_grid(n_points=n_points, dt=dt)

    def diffusion_layer_thickness(self, D: float) -> float:
        r"""Compute the Nernst stagnant diffusion layer thickness :math:`\delta_N`.

        Delegates to :func:`~softpotato.analytical.hydrodynamics.nernst_diffusion_layer`:

        .. math::

            \delta_N = 1.610 \, D^{1/3} \nu^{1/6} \omega^{-1/2}

        Parameters
        ----------
        D : float
            Diffusion coefficient :math:`D > 0` in consistent units with
            :attr:`kinematic_viscosity` (e.g., :math:`\text{m}^2/\text{s}`).

        Returns
        -------
        float
            Diffusion layer thickness :math:`\delta_N > 0`.
        """
        d_val = _validate_positive_finite(D, "D")
        return float(
            nernst_diffusion_layer(
                omega=self._omega,
                D=d_val,
                nu=self._kinematic_viscosity,
            )
        )

    @overload
    def axial_velocity(self, x: float) -> float: ...

    @overload
    def axial_velocity(self, x: np.ndarray) -> np.ndarray: ...

    @overload
    def axial_velocity(self, x: float | ArrayLike) -> float | np.ndarray: ...

    def axial_velocity(self, x: float | ArrayLike) -> float | np.ndarray:
        r"""Evaluate the von Kármán axial fluid velocity :math:`v_x(x)` toward the disk.

        .. math::

            v_x(x) = -0.51023 \, \omega^{3/2} \nu^{-1/2} x^2

        Parameters
        ----------
        x : float or array-like
            Axial distance(s) from the disk electrode surface (:math:`x \ge 0`).

        Returns
        -------
        float or np.ndarray
            Axial velocity :math:`v_x(x) \le 0` in :math:`\text{m}/\text{s}`
            (or consistent units).
        """
        coeff = (
            -_VON_KARMAN_A * (self._omega**1.5) * (self._kinematic_viscosity ** (-0.5))
        )
        if np.isscalar(x):
            x_val = _validate_finite_real(x, "x")
            if x_val < 0.0:
                raise ValueError(f"Axial coordinate x must be >= 0, got {x!r}.")
            return float(coeff * x_val * x_val)

        x_arr = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(x_arr)) or np.any(x_arr < 0.0):
            raise ValueError("All axial coordinates in x must be finite and >= 0.")
        return coeff * (x_arr**2)

    def build_grid(
        self,
        D: float,
        n_points: int = 80,
        kind: str = "geometric",
        gamma: float = 1.06,
        *,
        beta: float = 3.0,
        domain_factor: float | None = None,
    ) -> Mesh1D:
        r"""Construct a 1D spatial mesh bounded by the hydrodynamic diffusion layer.

        For ``model="nernst_layer"``, the default domain is :math:`[0, \delta_N]`.
        For ``model="convective_diffusion"``, the default domain is
        :math:`[0, k \cdot \delta_N]` with :math:`k = 3.0`.

        Parameters
        ----------
        D : float
            Reference diffusion coefficient :math:`D > 0`.
        n_points : int, default 80
            Number of spatial grid nodes (:math:`\ge 3`).
        kind : {'geometric', 'exponential', 'uniform'}, default 'geometric'
            Mesh discretization family.
        gamma : float, default 1.06
            Geometric expansion ratio :math:`\gamma > 1` (used when
            ``kind="geometric"``).
        beta : float, default 3.0
            Exponential stretching factor :math:`\beta > 0` (used when
            ``kind="exponential"``).
        domain_factor : float, optional
            Override for :attr:`domain_factor` (:math:`k > 0`).

        Returns
        -------
        Mesh1D
            Constructed 1D spatial mesh spanning :math:`[0, k \delta_N]`.
        """
        n_pts = _validate_positive_int(n_points, "n_points", minimum=3)
        factor = (
            self._domain_factor
            if domain_factor is None
            else _validate_positive_finite(domain_factor, "domain_factor")
        )
        delta = self.diffusion_layer_thickness(D)
        x_max = factor * delta

        if not isinstance(kind, str):
            raise TypeError(f"kind must be a string, got {type(kind).__name__}.")
        kind_norm = kind.strip().lower()

        if kind_norm == "geometric":
            return GeometricExpandingMesh(0.0, x_max, n_points=n_pts, gamma=gamma)
        if kind_norm == "exponential":
            return ExponentialMesh(0.0, x_max, n_points=n_pts, beta=beta)
        if kind_norm == "uniform":
            return Uniform1DMesh(0.0, x_max, n_points=n_pts)

        raise ValueError(
            f"Unknown mesh kind {kind!r}. Must be one of "
            "('geometric', 'exponential', 'uniform')."
        )

    def convection_term(
        self,
        diffusivity: float | Mapping[str, float],
        *,
        reactions: Any = None,
    ) -> RDEConvectionTerm:
        """Create an :class:`RDEConvectionTerm` for ``DiffusionProblem(reactions=...)``."""
        return RDEConvectionTerm(
            omega=self._omega,
            kinematic_viscosity=self._kinematic_viscosity,
            diffusivity=diffusivity,
            base_reactions=reactions,
        )

    def levich_limiting_current(
        self,
        *,
        n: float = 1,
        D: float = 1e-9,
        c_bulk: float = 1.0,
        area: float = 1.0,
    ) -> float:
        r"""Compute the analytical Levich limiting current :math:`I_L` at this rotation rate.

        Delegates to :func:`~softpotato.analytical.hydrodynamics.levich`.
        """
        return float(
            levich(
                omega=self._omega,
                n=n,
                D=D,
                c_bulk=c_bulk,
                area=area,
                nu=self._kinematic_viscosity,
            )
        )

    def with_rotation(
        self,
        *,
        omega: float | None = None,
        rpm: float | None = None,
    ) -> RotatingDiskElectrode:
        """Return a copy of this RDE technique at a different rotation rate."""
        return RotatingDiskElectrode(
            self._waveform,
            omega=omega,
            rpm=rpm,
            kinematic_viscosity=self._kinematic_viscosity,
            model=self._model,
            domain_factor=self._domain_factor,
        )

    def build_problem(
        self,
        *,
        D: float | None = None,
        diffusivity: Mapping[str, float] | None = None,
        c_bulk: float = 1.0,
        initial_conditions: Mapping[str, Any] | None = None,
        boundary_conditions: Mapping[str, tuple[BoundaryCondition, BoundaryCondition]]
        | None = None,
        kinetics: HeterogeneousKinetics | NernstianEquilibrium | None = None,
        species_name: str = "Ox",
        red_species_name: str = "Red",
        c_red_bulk: float = 0.0,
        n_electrons: int | Mapping[str, int] = 1,
        area: float = 1.0,
        reactions: Any = None,
        n_points: int = 80,
        grid_kind: str = "geometric",
        gamma: float = 1.06,
        beta: float = 3.0,
        domain_factor: float | None = None,
    ) -> DiffusionProblem:
        """Build a configured :class:`~softpotato.solver.DiffusionProblem` for this RDE.

        Automatically sizes the spatial mesh to the hydrodynamic diffusion
        layer, attaches bulk Dirichlet outer boundary conditions, and injects
        the axial convective velocity operator when
        ``model="convective_diffusion"``.
        """
        if D is None and diffusivity is None:
            raise ValueError("Either 'D' or 'diffusivity' must be provided.")
        if D is not None and diffusivity is not None:
            raise ValueError("Specify either 'D' or 'diffusivity', not both.")

        area_val = _validate_positive_finite(area, "area")
        n_electrons_arg: int | dict[str, int] = (
            dict(n_electrons) if isinstance(n_electrons, Mapping) else n_electrons
        )

        if diffusivity is not None:
            diff_map = {
                sp: _validate_positive_finite(val, f"diffusivity[{sp!r}]")
                for sp, val in diffusivity.items()
            }
            if not diff_map:
                raise ValueError("'diffusivity' mapping must not be empty.")
            d_ref = max(diff_map.values())
        else:
            assert D is not None
            d_ref = _validate_positive_finite(D, "D")
            if kinetics is not None and boundary_conditions is None:
                diff_map = {species_name: d_ref, red_species_name: d_ref}
            else:
                diff_map = {species_name: d_ref}

        mesh = self.build_grid(
            D=d_ref,
            n_points=n_points,
            kind=grid_kind,
            gamma=gamma,
            beta=beta,
            domain_factor=domain_factor,
        )

        if boundary_conditions is None and isinstance(kinetics, HeterogeneousKinetics):
            from softpotato.reactions.mechanisms import ElectrochemicalStep, Mechanism
            from softpotato.reactions.species import Species

            d_ox = diff_map.get(species_name, d_ref)
            d_red = diff_map.get(red_species_name, d_ref)
            diff_map.setdefault(species_name, d_ox)
            diff_map.setdefault(red_species_name, d_red)

            ox_sp = Species(
                species_name,
                diffusivity=d_ox,
                bulk_concentration=c_bulk,
                n_electrons=int(kinetics.n),
            )
            red_sp = Species(
                red_species_name,
                diffusivity=d_red,
                bulk_concentration=c_red_bulk,
            )
            mech = Mechanism(
                species=[ox_sp, red_sp],
                electrochemical_steps=[ElectrochemicalStep(ox_sp, red_sp, kinetics)],
            )
            mech_problem = mech.create_problem(
                mesh,
                potential=self,
                geometry=PlanarGeometry(area=area_val),
                diffusivity=diff_map,
                initial_conditions=initial_conditions,
                n_electrons=n_electrons_arg,
                electrodes="left",
                bulk_bc="dirichlet",
            )
            bcs = mech_problem.boundary_conditions
            ics = mech_problem.initial_conditions
            interfacial_rxn = mech_problem.reactions
            if reactions is not None and interfacial_rxn is not None:
                # Compose user homogeneous reactions with interfacial kinetics
                user_rxn = reactions
                mech_rxn = interfacial_rxn

                def _composed_rxn(
                    c_dict: dict[str, np.ndarray],
                    x_arr: np.ndarray,
                    t_val: float,
                ) -> dict[str, np.ndarray]:
                    out_i = mech_rxn(c_dict, x_arr, t_val)
                    out_u = user_rxn(c_dict, x_arr, t_val)
                    return {sp: out_i[sp] + out_u[sp] for sp in out_i}

                base_rxn: Any = _composed_rxn
            elif reactions is not None:
                base_rxn = reactions
            else:
                base_rxn = interfacial_rxn

            effective_reactions = (
                self.convection_term(diff_map, reactions=base_rxn)
                if self._model == "convective_diffusion"
                else base_rxn
            )

            return DiffusionProblem(
                grid=mesh,
                diffusivity=diff_map,
                boundary_conditions=bcs,
                initial_conditions=ics,
                reactions=effective_reactions,
                species=mech.species,
                geometry=PlanarGeometry(area=area_val),
                n_electrons=n_electrons_arg,
                mechanism=mech,
            )

        if boundary_conditions is not None:
            bcs = dict(boundary_conditions)
        elif kinetics is not None:
            d_ox = diff_map.get(species_name, d_ref)
            d_red = diff_map.get(red_species_name, d_ref)
            bcs = create_electrochemical_bcs(
                kinetics=kinetics,
                potential=self,
                ox=species_name,
                red=red_species_name,
                D_ox=d_ox,
                D_red=d_red,
                c_ox_bulk=c_bulk,
                c_red_bulk=c_red_bulk,
                electrode_side="left",
                bulk_bc="dirichlet",
            )
            diff_map.setdefault(species_name, d_ox)
            diff_map.setdefault(red_species_name, d_red)
        else:
            # Default mass-transport-limited (zero surface concentration) Dirichlet BC
            bcs = {sp: (DirichletBC(0.0), DirichletBC(c_bulk)) for sp in diff_map}

        if initial_conditions is not None:
            ics = dict(initial_conditions)
        elif kinetics is not None and boundary_conditions is None:
            ics = {species_name: c_bulk, red_species_name: c_red_bulk}
        else:
            ics = {sp: c_bulk for sp in diff_map}

        effective_reactions = (
            self.convection_term(diff_map, reactions=reactions)
            if self._model == "convective_diffusion"
            else reactions
        )

        return DiffusionProblem(
            grid=mesh,
            diffusivity=diff_map,
            boundary_conditions=bcs,
            initial_conditions=ics,
            reactions=effective_reactions,
            geometry=PlanarGeometry(area=area_val),
            n_electrons=n_electrons_arg,
        )

    def __repr__(self) -> str:
        return (
            f"RotatingDiskElectrode(waveform={self._waveform!r}, "
            f"omega={self._omega!r}, "
            f"kinematic_viscosity={self._kinematic_viscosity!r}, "
            f"model={self._model!r})"
        )


@dataclass(frozen=True)
class RDESweepResult:
    """Container for multi-rotation-rate RDE simulation sweep results.

    Attributes
    ----------
    omegas : np.ndarray
        Angular rotation velocities :math:`\\omega` in :math:`\\text{rad}/\\text{s}`
        of shape ``(M,)``.
    rpms : np.ndarray
        Rotational speeds in RPM of shape ``(M,)``.
    results : tuple[SolverResult, ...]
        Individual :class:`~softpotato.solver.SolverResult` objects for each
        rotation rate.
    limiting_currents : np.ndarray
        Final-time (steady-state) Faradaic current of :attr:`species` at each
        rotation rate, of shape ``(M,)`` in Amperes.
    techniques : tuple[RotatingDiskElectrode, ...]
        Configured :class:`RotatingDiskElectrode` instance for each rotation rate.
    species : str
        Primary species name used to populate :attr:`limiting_currents`.
    """

    omegas: np.ndarray
    rpms: np.ndarray
    results: tuple[SolverResult, ...]
    limiting_currents: np.ndarray
    techniques: tuple[RotatingDiskElectrode, ...]
    species: str

    def __len__(self) -> int:
        return len(self.results)


class RDESweep:
    r"""Multi-rotation-rate Rotating Disk Electrode (RDE) sweep configuration.

    Generates an ordered collection of :class:`RotatingDiskElectrode`
    instances sharing the same potential waveform, electrolyte viscosity, and
    hydrodynamic transport model across a series of rotation rates
    :math:`\{\omega_1, \dots, \omega_M\}` for Levich and Koutecký–Levich
    analysis.

    Parameters
    ----------
    waveform : BaseTechnique, optional
        Shared potentiostatic excitation waveform. Mutually exclusive with
        direct LSV parameters (``E_start``, ``E_end``, ``scan_rate``).
    omegas : Sequence[float] or array-like, optional
        Sequence of positive angular rotation velocities in
        :math:`\text{rad}/\text{s}`. Mutually exclusive with ``rpms``.
    rpms : Sequence[float] or array-like, optional
        Sequence of positive rotational speeds in RPM. Mutually exclusive with
        ``omegas``.
    kinematic_viscosity : float, default 1e-6
        Electrolyte kinematic viscosity :math:`\nu > 0` in :math:`\text{m}^2/\text{s}`.
    model : {'nernst_layer', 'convective_diffusion'}, default 'nernst_layer'
        Hydrodynamic mass-transport model.
    E_start, E_end, scan_rate, dE, n_points : optional
        Direct LSV waveform parameters when ``waveform`` is omitted.
    domain_factor : float, optional
        Domain multiplier :math:`k > 0` applied to :math:`\delta_N`.

    Examples
    --------
    >>> from softpotato.techniques import RDESweep
    >>> sweep = RDESweep(
    ...     E_start=0.5, E_end=-0.5, scan_rate=0.05, rpms=[400, 900, 1600, 2500]
    ... )
    >>> len(sweep)
    4
    """

    def __init__(
        self,
        waveform: BaseTechnique | None = None,
        *,
        omegas: Sequence[float] | ArrayLike | None = None,
        rpms: Sequence[float] | ArrayLike | None = None,
        kinematic_viscosity: float = _DEFAULT_KINEMATIC_VISCOSITY,
        model: RDEModel = "nernst_layer",
        E_start: float | None = None,
        E_end: float | None = None,
        scan_rate: float | None = None,
        dE: float | None = None,
        n_points: int | None = None,
        domain_factor: float | None = None,
    ) -> None:
        self._waveform = _resolve_waveform(
            waveform,
            E_start=E_start,
            E_end=E_end,
            scan_rate=scan_rate,
            dE=dE,
            n_points=n_points,
        )
        self._omegas, self._rpms = _resolve_rotation_sequence(omegas, rpms)
        self._kinematic_viscosity = _validate_positive_finite(
            kinematic_viscosity, "kinematic_viscosity"
        )
        self._model: RDEModel = _resolve_rde_model(model)
        self._domain_factor = (
            None
            if domain_factor is None
            else _validate_positive_finite(domain_factor, "domain_factor")
        )
        self._techniques: tuple[RotatingDiskElectrode, ...] = tuple(
            RotatingDiskElectrode(
                self._waveform,
                omega=float(w),
                kinematic_viscosity=self._kinematic_viscosity,
                model=self._model,
                domain_factor=self._domain_factor,
            )
            for w in self._omegas
        )

    @property
    def waveform(self) -> BaseTechnique:
        """Shared potentiostatic excitation waveform."""
        return self._waveform

    @property
    def omegas(self) -> np.ndarray:
        r"""1D array of angular rotation velocities :math:`\omega` in :math:`\text{rad}/\text{s}`."""
        return self._omegas.copy()

    @property
    def rpms(self) -> np.ndarray:
        """1D array of rotational speeds in RPM."""
        return self._rpms.copy()

    @property
    def kinematic_viscosity(self) -> float:
        r"""Kinematic viscosity :math:`\nu` in :math:`\text{m}^2/\text{s}`."""
        return self._kinematic_viscosity

    @property
    def nu(self) -> float:
        """Alias for :attr:`kinematic_viscosity`."""
        return self._kinematic_viscosity

    @property
    def model(self) -> RDEModel:
        """Hydrodynamic mass-transport model."""
        return self._model

    @property
    def n_rates(self) -> int:
        """Number of rotation rates in the sweep."""
        return len(self._techniques)

    @property
    def techniques(self) -> tuple[RotatingDiskElectrode, ...]:
        """Tuple of configured :class:`RotatingDiskElectrode` instances."""
        return self._techniques

    def diffusion_layer_thicknesses(self, D: float) -> np.ndarray:
        r"""Compute Nernst diffusion layer thickness :math:`\delta_N(\omega)` for each rate."""
        return np.asarray(
            [tech.diffusion_layer_thickness(D) for tech in self._techniques],
            dtype=float,
        )

    def __len__(self) -> int:
        return len(self._techniques)

    def __iter__(self) -> Iterator[RotatingDiskElectrode]:
        return iter(self._techniques)

    @overload
    def __getitem__(self, index: int) -> RotatingDiskElectrode: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[RotatingDiskElectrode, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> RotatingDiskElectrode | tuple[RotatingDiskElectrode, ...]:
        return self._techniques[index]

    def simulate(
        self,
        *,
        D: float | None = None,
        diffusivity: Mapping[str, float] | None = None,
        c_bulk: float = 1.0,
        initial_conditions: Mapping[str, Any] | None = None,
        boundary_conditions: Mapping[str, tuple[BoundaryCondition, BoundaryCondition]]
        | None = None,
        kinetics: HeterogeneousKinetics | NernstianEquilibrium | None = None,
        species_name: str = "Ox",
        red_species_name: str = "Red",
        c_red_bulk: float = 0.0,
        n_electrons: int | Mapping[str, int] = 1,
        area: float = 1.0,
        reactions: Any = None,
        n_points: int = 80,
        grid_kind: str = "geometric",
        gamma: float = 1.06,
        beta: float = 3.0,
        domain_factor: float | None = None,
        solver: str = "scipy_ivp",
        t_eval: np.ndarray | None = None,
        solver_kwargs: Mapping[str, Any] | None = None,
    ) -> RDESweepResult:
        """Run the PDE simulation across all rotation rates in this sweep."""
        return simulate_rde_sweep(
            sweep=self,
            D=D,
            diffusivity=diffusivity,
            c_bulk=c_bulk,
            initial_conditions=initial_conditions,
            boundary_conditions=boundary_conditions,
            kinetics=kinetics,
            species_name=species_name,
            red_species_name=red_species_name,
            c_red_bulk=c_red_bulk,
            n_electrons=n_electrons,
            area=area,
            reactions=reactions,
            n_points=n_points,
            grid_kind=grid_kind,
            gamma=gamma,
            beta=beta,
            domain_factor=domain_factor,
            solver=solver,
            t_eval=t_eval,
            solver_kwargs=solver_kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"RDESweep(waveform={self._waveform!r}, "
            f"n_rates={self.n_rates}, "
            f"kinematic_viscosity={self._kinematic_viscosity!r}, "
            f"model={self._model!r})"
        )


def simulate_rde_sweep(
    *,
    sweep: RDESweep | None = None,
    waveform: BaseTechnique | None = None,
    omegas: Sequence[float] | ArrayLike | None = None,
    rpms: Sequence[float] | ArrayLike | None = None,
    kinematic_viscosity: float = _DEFAULT_KINEMATIC_VISCOSITY,
    model: RDEModel = "nernst_layer",
    E_start: float | None = None,
    E_end: float | None = None,
    scan_rate: float | None = None,
    dE: float | None = None,
    n_time_points: int | None = None,
    D: float | None = None,
    diffusivity: Mapping[str, float] | None = None,
    c_bulk: float = 1.0,
    initial_conditions: Mapping[str, Any] | None = None,
    boundary_conditions: Mapping[str, tuple[BoundaryCondition, BoundaryCondition]]
    | None = None,
    kinetics: HeterogeneousKinetics | NernstianEquilibrium | None = None,
    species_name: str = "Ox",
    red_species_name: str = "Red",
    c_red_bulk: float = 0.0,
    n_electrons: int | Mapping[str, int] = 1,
    area: float = 1.0,
    reactions: Any = None,
    n_points: int = 80,
    grid_kind: str = "geometric",
    gamma: float = 1.06,
    beta: float = 3.0,
    domain_factor: float | None = None,
    solver: str = "scipy_ivp",
    t_eval: np.ndarray | None = None,
    solver_kwargs: Mapping[str, Any] | None = None,
) -> RDESweepResult:
    """Simulate a multi-rotation-rate RDE sweep for Levich or Koutecký–Levich analysis.

    Accepts either a pre-configured :class:`RDESweep` instance or direct sweep
    parameters (``omegas``/``rpms`` plus ``waveform`` or LSV parameters), builds
    a rotation-rate-specific :class:`~softpotato.solver.DiffusionProblem` for
    each speed, and solves all cases in a single call.

    Parameters
    ----------
    sweep : RDESweep, optional
        Pre-configured sweep object. If omitted, one is constructed from
        ``omegas``/``rpms`` and ``waveform`` (or LSV parameters).
    waveform : BaseTechnique, optional
        Shared potentiostatic waveform (ignored if ``sweep`` is supplied).
    omegas, rpms : Sequence[float], optional
        Rotation rates in rad/s or RPM (ignored if ``sweep`` is supplied).
    kinematic_viscosity : float, default 1e-6
        Electrolyte kinematic viscosity in :math:`\\text{m}^2/\\text{s}`.
    model : {'nernst_layer', 'convective_diffusion'}, default 'nernst_layer'
        Hydrodynamic transport model.
    D : float, optional
        Scalar diffusion coefficient :math:`D > 0`. Mutually exclusive with
        ``diffusivity``.
    diffusivity : Mapping[str, float], optional
        Per-species diffusion coefficients. Mutually exclusive with ``D``.
    c_bulk : float, default 1.0
        Bulk concentration of the primary electroactive species.
    initial_conditions : Mapping[str, Any], optional
        Optional explicit initial concentration profiles.
    boundary_conditions : Mapping[str, tuple[BoundaryCondition, BoundaryCondition]], optional
        Optional explicit boundary conditions.
    kinetics : HeterogeneousKinetics or NernstianEquilibrium, optional
        Interfacial charge-transfer model used to generate boundary conditions
        when ``boundary_conditions`` is omitted.
    species_name : str, default 'Ox'
        Primary electroactive species name.
    red_species_name : str, default 'Red'
        Reduced partner species name (used when ``kinetics`` is supplied).
    c_red_bulk : float, default 0.0
        Bulk concentration of the reduced partner species.
    n_electrons : int or Mapping[str, int], default 1
        Electrons transferred per species (matches :class:`~softpotato.solver.DiffusionProblem`;
        ``n_electrons=1`` yields IUPAC negative cathodic currents when ``Ox`` is reduced).
    area : float, default 1.0
        Electrode geometric surface area.
    reactions : Any, optional
        Optional homogeneous reaction term.
    n_points : int, default 80
        Number of spatial grid nodes per simulation.
    grid_kind : {'geometric', 'exponential', 'uniform'}, default 'geometric'
        Mesh family passed to :meth:`RotatingDiskElectrode.build_grid`.
    gamma : float, default 1.06
        Geometric expansion ratio for ``grid_kind='geometric'``.
    beta : float, default 3.0
        Stretching parameter for ``grid_kind='exponential'``.
    domain_factor : float, optional
        Override domain multiplier :math:`k` applied to :math:`\\delta_N`.
    solver : str, default 'scipy_ivp'
        Registered solver name passed to :func:`~softpotato.solver.get_solver`.
    t_eval : np.ndarray, optional
        Evaluation timestamps shared across all rotation rates. Defaults to
        ``sweep.waveform.time_grid()``.
    solver_kwargs : Mapping[str, Any], optional
        Additional keyword arguments forwarded to ``solver.solve(...)``.

    Returns
    -------
    RDESweepResult
        Aggregated rotation rates, per-rate solver results, and limiting
        currents suitable for Levich or Koutecký–Levich analysis.
    """
    if sweep is None:
        sweep = RDESweep(
            waveform,
            omegas=omegas,
            rpms=rpms,
            kinematic_viscosity=kinematic_viscosity,
            model=model,
            E_start=E_start,
            E_end=E_end,
            scan_rate=scan_rate,
            dE=dE,
            n_points=n_time_points,
            domain_factor=domain_factor,
        )
    elif not isinstance(sweep, RDESweep):
        raise TypeError(
            f"sweep must be an instance of RDESweep, got {type(sweep).__name__}."
        )

    solver_inst = get_solver(solver)
    extra_kwargs = dict(solver_kwargs) if solver_kwargs is not None else {}
    t_grid = (
        sweep.waveform.time_grid()
        if t_eval is None
        else np.asarray(t_eval, dtype=float)
    )
    t_span = (0.0, sweep.waveform.duration)

    results_list: list[SolverResult] = []
    limiting: list[float] = []
    primary_species = species_name

    for tech in sweep.techniques:
        problem = tech.build_problem(
            D=D,
            diffusivity=diffusivity,
            c_bulk=c_bulk,
            initial_conditions=initial_conditions,
            boundary_conditions=boundary_conditions,
            kinetics=kinetics,
            species_name=species_name,
            red_species_name=red_species_name,
            c_red_bulk=c_red_bulk,
            n_electrons=n_electrons,
            area=area,
            reactions=reactions,
            n_points=n_points,
            grid_kind=grid_kind,
            gamma=gamma,
            beta=beta,
            domain_factor=domain_factor,
        )
        if primary_species not in problem.species:
            primary_species = problem.species[0]

        res = solver_inst.solve(
            problem,
            t_span=t_span,
            t_eval=t_grid,
            **extra_kwargs,
        )
        results_list.append(res)
        if res.success and primary_species in res.currents:
            limiting.append(float(res.currents[primary_species][-1]))
        else:
            limiting.append(float("nan"))

    return RDESweepResult(
        omegas=sweep.omegas,
        rpms=sweep.rpms,
        results=tuple(results_list),
        limiting_currents=np.asarray(limiting, dtype=float),
        techniques=sweep.techniques,
        species=primary_species,
    )


__all__ = [
    "RDEConvectionTerm",
    "RDEModel",
    "RDESweep",
    "RDESweepResult",
    "RotatingDiskElectrode",
    "simulate_rde_sweep",
]
