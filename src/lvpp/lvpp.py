from petsc4py import PETSc
import dolfinx
from enum import Enum

import warnings


__all__ = ["AlphaScheme", "NewtonSolver"]


class AlphaScheme(Enum):
    constant = 1  # Constant alpha (alpha_0)
    linear = 2  # Linearly increasing alpha (alpha_0 + alpha_c * i) where i is the iteration number
    doubling = 3  # Doubling alpha (alpha_0 * 2^i) where i is the iteration number

    @classmethod
    def from_string(cls, method: str):
        if method == "constant":
            return AlphaScheme.constant
        elif method == "linear":
            return AlphaScheme.linear
        elif method == "doubling":
            return AlphaScheme.doubling
        else:
            raise ValueError(f"Unknown alpha scheme {method}")


class NewtonSolver:
    max_iterations: int
    bcs: list[dolfinx.fem.DirichletBC]
    A: PETSc.Mat
    b: PETSc.Vec
    J: dolfinx.fem.Form
    b: dolfinx.fem.Form
    dx: PETSc.Vec
    error_on_nonconvergence: bool

    def __init__(
        self,
        F: list[dolfinx.fem.form],
        J: list[list[dolfinx.fem.form]],
        w: list[dolfinx.fem.Function],
        bcs: list[dolfinx.fem.DirichletBC] | None = None,
        max_iterations: int = 5,
        petsc_options: dict[str, str | float | int | None] = None,
        error_on_nonconvergence: bool = True,
    ):
        """Newton solver for blocked nonlinear problems.

        Note:
            Special feature of this solver is that it only measures the norm of the primal space
            increments when checking convergence (primal being the first space in the block).

        :param F: Residual on block form
        :param J: Block formulation of Jacobian
        :param w: List of solution vectors
        :param bcs: List of Dirichlet boundary conditions
        :param max_iterations: Max Newton iterations
        :param petsc_options: Krylov subspace solver options
        :param error_on_nonconvergence: Throw error if solver doesn't converge.
        """
        self.max_iterations = max_iterations
        self.bcs = [] if bcs is None else bcs
        self.b = dolfinx.fem.petsc.create_vector_block(F)
        self.F = F
        self.J = J
        self.A = dolfinx.fem.petsc.create_matrix_block(J)
        self.dx = dolfinx.fem.petsc.create_vector_block(F)
        self.w = w
        self.x = dolfinx.fem.petsc.create_vector_block(F)
        self.norm_array = dolfinx.fem.Function(w[0].function_space)
        self.error_on_nonconvergence = error_on_nonconvergence
        # Set PETSc options
        opts = PETSc.Options()
        if petsc_options is not None:
            for k, v in petsc_options.items():
                opts[k] = v

        # Define KSP solver
        self._solver = PETSc.KSP().create(self.b.getComm().tompi4py())
        self._solver.setOperators(self.A)
        self._solver.setFromOptions()

        # Set matrix and vector PETSc options
        self.A.setFromOptions()
        self.b.setFromOptions()

    def _update_solution(self, beta):
        """Update solution vector ``w`` and internal variable ``x``.

        Two steps are performed:
        1. Update local arrays in ``w`` with the correction ``dx``.
        2. Scatter local arrays to global vector ``x``.
        """
        maps = [
            (
                si.function_space.dofmap.index_map,
                si.function_space.dofmap.index_map_bs,
            )
            for si in self.w
        ]
        # Get local vectors and update ``w`` with correction
        local_dx = dolfinx.cpp.la.petsc.get_local_vectors(self.dx, maps)
        for ldx, s in zip(local_dx, self.w):
            s.x.array[:] -= beta * ldx
            s.x.scatter_forward()

        # Scatter local vectors to blocked vector
        dolfinx.cpp.la.petsc.scatter_local_vectors(
            self.x, [si.x.petsc_vec.array_r for si in self.w], maps
        )
        self.x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def solve(self, tol=1e-6, beta=1.0) -> int:
        """Solve nonlinear problem

        Args:
            tol: Stopping tolerance for primal variable update
            beta: Step-size

        Raises:
            RuntimeError: If solver doesn't converge and ``error_on_nonconvergence=True``

        Returns:
            Number of iterations. If Krylov subspace solver doesn't converge, return 0.
        """
        i = 1
        tol_ = tol
        blocked_maps = [
            (
                si.function_space.dofmap.index_map,
                si.function_space.dofmap.index_map_bs,
            )
            for si in self.w
        ]
        while i <= self.max_iterations:
            if i < self.max_iterations // 2:
                tol = 10 * tol_
            else:
                tol = tol_
            # Assemble F(u_{i-1}) - J(u_D - u_{i-1}) and set du|_bc= u_D - u_{i-1}
            with self.b.localForm() as b_loc:
                b_loc.set(0)
            dolfinx.fem.petsc.assemble_vector_block(
                self.b, self.F, self.J, bcs=self.bcs, x0=self.x, scale=-1.0
            )
            self.b.ghostUpdate(PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD)

            # Assemble Jacobian
            self.A.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix_block(self.A, self.J, bcs=self.bcs)
            self.A.assemble()

            # Solve linear system for correction
            with self.dx.localForm() as dx_loc:
                dx_loc.set(0)
            self._solver.solve(self.b, self.dx)
            self.dx.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            # Update solution
            self._update_solution(beta)

            # Check for convergence
            if self.error_on_nonconvergence:
                assert (
                    self._solver.getConvergedReason() > 0
                ), "Linear solver did not converge, received reason {}".format(
                    self._solver.getConvergedReason()
                )
            else:
                converged = self._solver.getConvergedReason()
                if converged <= 0:
                    warnings.warn("Linear solver did not converge, exiting", RuntimeWarning)
                    return 0

            # Compute norm of primal space diff

            local_du, _ = dolfinx.cpp.la.petsc.get_local_vectors(self.dx, blocked_maps)

            self.norm_array.x.array[:] = local_du
            self.norm_array.x.petsc_vec.ghostUpdate(
                PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD
            )
            self.norm_array.x.petsc_vec.normBegin(1)
            correction_norm = self.norm_array.x.petsc_vec.normEnd(1)

            print(f"Iteration {i}: Correction norm {correction_norm}")
            if correction_norm < tol:
                break
            i += 1
        if self.error_on_nonconvergence and i == self.max_iterations:
            raise RuntimeError("Newton solver did not converge")
        return i
