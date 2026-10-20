# Geometry Module (`softpotato.geometry`)

The `softpotato.geometry` subpackage provides spatial coordinate abstractions, finite-difference differential operators ($\nabla^2$, $\nabla$), normal interfacial flux evaluation, and active electrode surface integration across planar, spherical, and cylindrical domains.

## Coordinate Geometries and Differential Operators

| Geometry Class & Aliases | Symmetry Factor ($p$) | Laplacian Operator ($\nabla^2 c$) | Active Surface Area ($A$) |
| :--- | :---: | :--- | :--- |
| [`PlanarGeometry`][softpotato.geometry.PlanarGeometry] (`'planar'`, `'cartesian'`, `'1d'`, `'linear'`) | $p = 0$ | $\dfrac{\partial^2 c}{\partial x^2}$ | User-specified $A$ (default $1.0$) |
| [`CylindricalGeometry`][softpotato.geometry.CylindricalGeometry] (`'cylindrical'`, `'cylinder'`, `'wire'`) | $p = 1$ | $\dfrac{\partial^2 c}{\partial r^2} + \dfrac{1}{r}\dfrac{\partial c}{\partial r}$ | $2 \pi r_0 L$ |
| [`SphericalGeometry`][softpotato.geometry.SphericalGeometry] (`'spherical'`, `'sphere'`, `'hemispherical'`, `'hemisphere'`) | $p = 2$ | $\dfrac{\partial^2 c}{\partial r^2} + \dfrac{2}{r}\dfrac{\partial c}{\partial r}$ | $4 \pi r_0^2$ (full sphere) or $2 \pi r_0^2$ (hemisphere) |

## Module Contents

::: softpotato.geometry

