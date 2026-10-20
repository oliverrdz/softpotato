# Issue 1: Implement the Geometry Module (`softpotato.geometry`)

**Title**: `feat(geometry): implement spatial geometry module with planar, spherical, and cylindrical coordinate operators`  
**Labels**: `enhancement`, `architecture`, `module:geometry`, `milestone-2`

---

## 1. Context & Motivation

Currently in Soft Potato, spatial coordinate assumptions and differential operator stencils are hardcoded directly as 1D Cartesian ($\frac{\partial^2 c}{\partial x^2}$) inside each solver class.

To support spherical microelectrodes, hanging mercury drops, cylindrical wires, and future 2D axisymmetric microdisc electrodes, spatial physics must be abstracted into a dedicated `softpotato.geometry` module. This encapsulates coordinate metrics, differential operators ($\nabla^2$, $\nabla$), and surface area integration, establishing clean separation between spatial physics and time-stepping algorithms.

---

## 2. Proposed Design & Architecture

### Module Structure
Create `src/softpotato/geometry/`:
- `base.py`: Abstract `Geometry` base class defining the spatial operator contracts.
- `planar.py`: `PlanarGeometry` ($p = 0$).
- `spherical.py`: `SphericalGeometry` ($p = 2$).
- `cylindrical.py`: `CylindricalGeometry` ($p = 1$).
- `__init__.py`: Public exports and factory helpers (`get_geometry`).

### Base Class Interface (`Geometry`)
```python
from abc import ABC, abstractmethod
import numpy as np


class Geometry(ABC):
    """Abstract base class representing spatial domain coordinate geometry."""

    name: str
    dimension: int
    coordinate_name: str  # e.g., 'x' or 'r'

    @abstractmethod
    def laplacian(
        self,
        c: np.ndarray,
        grid: np.ndarray,
        bc_left: Any = None,
        bc_right: Any = None,
        t: float = 0.0,
    ) -> np.ndarray:
        """Evaluate the spatial Laplacian ∇²c at interior and boundary points."""
        ...

    @abstractmethod
    def laplacian_matrix(
        self,
        grid: np.ndarray,
        bc_left: Any = None,
        bc_right: Any = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return banded or sparse matrix representation of ∇² for implicit solvers."""
        ...

    @abstractmethod
    def surface_flux(
        self,
        c: np.ndarray,
        grid: np.ndarray,
        diffusivity: float,
        side: str = "left",
    ) -> float | np.ndarray:
        """Calculate normal interfacial diffusion flux J = -D (∂c/∂n) at boundary."""
        ...

    @abstractmethod
    def current(
        self,
        flux: float | np.ndarray,
        n_electrons: int = 1,
    ) -> float | np.ndarray:
        """Integrate normal flux over the active electrode surface to compute Faradaic current."""
        ...
```

### Concrete Implementations

1. **`PlanarGeometry`** ($p = 0$):
   - Coordinate: $x \in [0, L]$.
   - Laplacian: $\nabla^2 c = \frac{\partial^2 c}{\partial x^2}$.
   - Surface Area: $A$ (user-specified, defaults to $1.0\ \text{cm}^2$ or $1.0\ \text{m}^2$).
   - Current: $I = n F A J$.

2. **`SphericalGeometry`** ($p = 2$):
   - Coordinate: $r \in [r_0, r_{\max}]$ with $r_0 > 0$.
   - Laplacian: $\nabla^2 c = \frac{\partial^2 c}{\partial r^2} + \frac{2}{r} \frac{\partial c}{\partial r}$.
   - Surface Area: $A = 4\pi r_0^2$ (full sphere) or $2\pi r_0^2$ (hemisphere).
   - Current: $I = n F A J$.

3. **`CylindricalGeometry`** ($p = 1$):
   - Coordinate: $r \in [r_0, r_{\max}]$ with $r_0 > 0$.
   - Laplacian: $\nabla^2 c = \frac{\partial^2 c}{\partial r^2} + \frac{1}{r} \frac{\partial c}{\partial r}$.
   - Surface Area: $A = 2\pi r_0 L$.
   - Current: $I = n F A J$.

---

## 3. Tasks & Deliverables

- [ ] Implement `src/softpotato/geometry/base.py` with abstract class `Geometry`.
- [ ] Implement `src/softpotato/geometry/planar.py` with `PlanarGeometry`.
- [ ] Implement `src/softpotato/geometry/spherical.py` with `SphericalGeometry` (supporting full sphere and hemisphere).
- [ ] Implement `src/softpotato/geometry/cylindrical.py` with `CylindricalGeometry`.
- [ ] Add factory function `get_geometry(name_or_instance, **kwargs)` with alias support (`"planar"`, `"spherical"`, `"cylindrical"`).
- [ ] Export classes in `src/softpotato/geometry/__init__.py` and top-level `src/softpotato/__init__.py`.
- [ ] Add unit tests in `tests/test_geometry.py`:
  - Operator accuracy against analytical functions with known Laplacians (e.g. $c(r) = 1/r \implies \nabla^2 c = 0$ in spherical coordinates).
  - Flux and current scaling across geometry instances.
  - Parameter validation ($r_0 > 0$, valid grid bounds).
- [ ] Add MkDocs API documentation in `docs/api/geometry.md` (using `mkdocstrings`) and update `mkdocs.yml` under `API Reference`.

---

## 4. Acceptance Criteria & Backwards Compatibility

- **Backwards Compatibility**:
  - `PlanarGeometry` is the default geometry for all solvers and problem definitions.
  - When no geometry is specified, the behavior, stencils, and results match the existing implementation exactly.
- `Geometry` subclasses provide vectorized, pure spatial operators without any dependency on time integration.
- Laplacian finite-difference operators achieve second-order spatial convergence $\mathcal{O}(\Delta r^2)$.
- 100% test pass rate with full type annotations (`py.typed` compliance).
- MkDocs site builds cleanly (`mkdocs build --strict`) with the new `docs/api/geometry.md` page rendered via `mkdocstrings`.

