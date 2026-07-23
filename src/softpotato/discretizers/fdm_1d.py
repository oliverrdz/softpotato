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

        main_diag = np.full(n, -2.0 * inv_dx2)
        lower_diag = np.full(n - 1, 1.0 * inv_dx2)
        upper_diag = np.full(n - 1, 1.0 * inv_dx2)

        # Zero out boundary rows (node 0 and node n-1)
        # Row 0: uses main_diag[0] and upper_diag[0]
        main_diag[0] = 0.0
        upper_diag[0] = 0.0

        # Row n-1: uses main_diag[-1] and lower_diag[-1]
        main_diag[-1] = 0.0
        lower_diag[-1] = 0.0

        laplacian = sp.diags(
            diagonals=[lower_diag, main_diag, upper_diag],
            offsets=[-1, 0, 1],
            shape=(n, n),
            format="csc",
        )

        return laplacian

    def build_system_matrix(self, mesh: BaseMesh, model: BaseModel) -> sp.csc_matrix:
        """
        Construct block-diagonal system linear matrix $A$ for multi-species diffusion.
        """
        laplacian = self.build_laplacian_matrix(mesh)
        diff_coeffs = model.get_diffusion_coefficients()

        species_blocks = []
        for species_name in model.species_names:
            D = diff_coeffs[species_name]
            species_blocks.append(D * laplacian)

        system_matrix = sp.block_diag(species_blocks, format="csc")
        return system_matrix
