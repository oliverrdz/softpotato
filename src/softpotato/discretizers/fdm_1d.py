import numpy as np
import scipy.sparse as sp

from softpotato.core.abcs import BaseDiscretizer, BaseMesh, BaseModel


class FDM1DDiscretizer(BaseDiscretizer):
    """
    1D Finite Difference Method (FDM) discretizer.

    Constructs 2nd-order central difference operators for spatial diffusion equations.
    """

    def build_laplacian_matrix(self, mesh: BaseMesh) -> sp.csc_matrix:
        """
        Construct second-derivative operator matrix $L = \\frac{\\partial^2}{\\partial x^2}$
        for 1D interior nodes using central finite differences.

        Boundary rows (i = 0 and i = N-1) are set to zero to allow boundary conditions
        to be applied independently during system assembly.

        Parameters
        ----------
        mesh : BaseMesh
            1D spatial mesh.

        Returns
        -------
        sp.csc_matrix
            Sparse laplacian matrix of size $(N \\times N)$.
        """
        n = mesh.num_nodes
        if n < 3:
            raise ValueError("1D FDM discretizer requires at least 3 mesh nodes.")

        dx = mesh.dx
        if isinstance(dx, np.ndarray):
            raise NotImplementedError(
                "FDM1DDiscretizer currently requires uniform grid spacing (scalar dx)."
            )

        inv_dx2 = 1.0 / (dx**2)

        # Diagonals for 2nd-order central difference stencil: [1, -2, 1] / dx^2
        main_diag = np.full(n, -2.0 * inv_dx2)
        off_diag = np.full(n - 1, 1.0 * inv_dx2)

        # Zero out boundary rows (node 0 and node N-1)
        main_diag[0] = 0.0
        main_diag[-1] = 0.0
        off_diag[0] = 0.0  # Disconnect node 0 from node 1 in row 0
        off_diag[-1] = 0.0  # Disconnect node N-1 from node N-2 in row N-1

        laplacian = sp.diags(
            diagonals=[off_diag, main_diag, off_diag],
            offsets=[-1, 0, 1],
            shape=(n, n),
            format="csc",
        )

        return laplacian

    def build_system_matrix(self, mesh: BaseMesh, model: BaseModel) -> sp.csc_matrix:
        """
        Construct block-diagonal system linear matrix $A$ for multi-species diffusion:

        $$\\frac{\\partial y}{\\partial t} = A y$$

        where $y = [C_{s_1}^T, C_{s_2}^T, \\dots]^T$ is the concatenated state vector.

        Parameters
        ----------
        mesh : BaseMesh
            Spatial mesh grid.
        model : BaseModel
            Multi-species transport model providing diffusion coefficients.

        Returns
        -------
        sp.csc_matrix
            Block diagonal operator matrix of shape $(M \\cdot N \\times M \\cdot N)$
            where $M$ is the number of species and $N$ is the number of spatial nodes.
        """
        laplacian = self.build_laplacian_matrix(mesh)
        diff_coeffs = model.get_diffusion_coefficients()

        species_blocks = []
        for species_name in model.species_names:
            D = diff_coeffs[species_name]
            species_blocks.append(D * laplacian)

        system_matrix = sp.block_diag(species_blocks, format="csc")
        return system_matrix
