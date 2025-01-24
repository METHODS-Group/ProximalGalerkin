from pathlib import Path
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
import basix.ufl
import ufl
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from typing import Optional
import argparse
from enum import Enum
from typing import Callable


class AlphaScheme(Enum):
    constant = 1  # Constant alpha (alpha_0)
    # Linearly increasing alpha (alpha_0 + alpha_c * i) where i is the iteration number
    linear = 2
    # Doubling alpha (alpha_0 * 2^i) where i is the iteration number
    doubling = 3

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


def solve_problem(
    N: int,
    primal_degree: int,
    alpha_scheme: AlphaScheme,
    alpha_0: float,
    alpha_c: float,
    max_iterations: int,
    stopping_tol: float,
    result_dir: Path,
):
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, N)
    print(N)

    el_0 = basix.ufl.element("Lagrange", mesh.topology.cell_name(), primal_degree + 1, shape=(3,))
    el_1 = basix.ufl.element("Lagrange", mesh.topology.cell_name(), primal_degree, shape=(3,))

    trial_el = basix.ufl.mixed_element([el_0, el_1])
    V = dolfinx.fem.functionspace(mesh, trial_el)

    sol = dolfinx.fem.Function(V)
    u, psi = ufl.split(sol)
    v, w = ufl.TestFunctions(V)

    dx = ufl.Measure("dx", domain=mesh)

    alpha = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(alpha_0))

    U, U_to_W = V.sub(0).collapse()

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

    Q, Q_to_W = V.sub(1).collapse()

    w0 = dolfinx.fem.Function(V)
    _, psi0 = ufl.split(w0)

    x = ufl.SpatialCoordinate(mesh)
    phi = ufl.sqrt(x[0] ** 2 + (2 + ufl.cos(2 * ufl.pi * x[0])) ** 2)
    # dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0))
    gamma = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(100000))

    # Variational form
    F = alpha * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    F += gamma * (ufl.dot(psi, psi) - phi) * ufl.inner(u, v) * dx
    F += ufl.inner(psi, v) * dx
    # f = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type((0, 0)))
    # F -= alpha * ufl.inner(f, v) * dx
    F -= ufl.inner(psi0, v) * dx

    F += ufl.inner(u, w) * dx

    non_lin_term = 1 / (ufl.sqrt(ufl.dot(psi, psi)))
    F -= phi * non_lin_term * ufl.dot(psi, w) * dx

    # def bc_func(x, theta=5.7 * np.pi):
    #     u_x = np.cos(theta * x[0])
    #     u_z = 1 - 2 * x[0] - np.sin(0.8 * np.pi * x[0])
    #     u_y = np.sin(theta * x[0])
    #     return (u_x, u_y, u_z) / np.sqrt(u_x**2 + u_y**2 + u_z**2)
    def bc_left(x):
        values = np.zeros((3, x.shape[1]))
        values[2] = 3.0
        return values

    def bc_right(x):
        values = np.zeros((3, x.shape[1]))
        values[0] = 1.0
        values[2] = 3.0
        return values

    # Create boundary conditions
    u_left = dolfinx.fem.Function(U)
    u_left.interpolate(bc_left)
    left_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[0], 0.0)
    )
    left_dofs = dolfinx.fem.locate_dofs_topological(
        (V.sub(0), U), mesh.topology.dim - 1, left_facets
    )
    u_right = dolfinx.fem.Function(U)
    u_right.interpolate(bc_right)

    right_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[0], 1.0)
    )
    right_dofs = dolfinx.fem.locate_dofs_topological(
        (V.sub(0), U), mesh.topology.dim - 1, right_facets
    )

    bcs = [
        dolfinx.fem.dirichletbc(u_left, left_dofs, V.sub(0)),
        dolfinx.fem.dirichletbc(u_right, right_dofs, V.sub(0)),
    ]

    # sol.sub(0).interpolate(lambda x: (
    #    np.ones(x.shape[1]), np.zeros(x.shape[1]), np.zeros(x.shape[1])))
    # [bc.set(sol.x.array) for bc in bcs]
    def initial_guess(x):
        return (x[0], np.zeros(x.shape[1]), -2 - np.cos(2 * np.pi * x[0]))

    sol.sub(0).interpolate(initial_guess)
    sol.sub(1).interpolate(initial_guess)
    problem = NonlinearProblem(F, sol, bcs=bcs)
    solver = NewtonSolver(mesh.comm, problem)
    solver.convergence_criterion = "residual"
    solver.rtol = 1e-9
    solver.atol = 1e-9
    solver.max_it = 25
    solver.error_on_nonconvergence = False

    ksp = solver.krylov_solver
    opts = PETSc.Options()  # type: ignore
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "preonly"
    opts[f"{option_prefix}pc_type"] = "lu"
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
    ksp.setFromOptions()

    dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

    u_out = dolfinx.fem.Function(U)
    u_out.name = "u"
    bp_u = dolfinx.io.VTXWriter(mesh.comm, result_dir / "u.bp", [u_out], engine="BP4")
    u_out.x.array[:] = sol.x.array[U_to_W]
    bp_u.write(0)

    psi_out = dolfinx.fem.Function(Q)
    psi_out.name = "psi"
    bp_psi = dolfinx.io.VTXWriter(mesh.comm, result_dir / "psi.bp", [psi_out], engine="BP4")
    psi_out.x.array[:] = sol.x.array[Q_to_W]
    bp_psi.write(0)

    diff = sol.sub(0) - w0.sub(0)
    L2_squared = ufl.dot(diff, diff) * dx
    compiled_diff = dolfinx.fem.form(L2_squared)
    newton_iterations = np.zeros(max_iterations, dtype=np.int32)
    L2_diff = np.zeros(max_iterations, dtype=np.float64)
    for i in range(1, max_iterations + 1):
        if alpha_scheme == AlphaScheme.constant:
            pass
        elif alpha_scheme == AlphaScheme.linear:
            alpha.value = alpha_0 + alpha_c * i
        elif alpha_scheme == AlphaScheme.doubling:
            alpha.value = alpha_0 * 2**i

        num_newton_iterations, converged = solver.solve(sol)
        newton_iterations[i - 1] = num_newton_iterations
        local_diff = dolfinx.fem.assemble_scalar(compiled_diff)
        global_diff = np.sqrt(mesh.comm.allreduce(local_diff, op=MPI.SUM))
        L2_diff[i - 1] = global_diff
        if mesh.comm.rank == 0:
            print(
                f"Iteration {i}: {converged =} {num_newton_iterations = } {
                    ksp.getConvergedReason() =}",
                f"|delta u |= {global_diff}",
            )

        u_out.x.array[:] = sol.x.array[U_to_W]
        psi_out.x.array[:] = sol.x.array[Q_to_W]

        bp_u.write(i)
        bp_psi.write(i)
        if global_diff < stopping_tol:
            break
        w0.x.array[:] = sol.x.array
    bp_u.close()
    bp_psi.close()
    # import scipy
    # import matplotlib.pyplot as plt
    # ai, aj, av = solver.A.getValuesCSR()
    # A_csr = scipy.sparse.csr_matrix((av, aj, ai))
    # plt.spy(A_csr)
    # plt.grid()
    # plt.savefig(f"local_sparsity.png")
    # breakpoint()
    return newton_iterations[: i + 1], L2_diff[: i + 1]


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def main(
    argv: Optional[list[str]] = None,
    phi_func: Callable[[np.ndarray], np.ndarray] = None,
    f_func: Callable[[np.ndarray], np.ndarray] = None,
):
    parser = argparse.ArgumentParser(formatter_class=CustomFormatter)
    mesh_options = parser.add_argument_group("Mesh options")
    mesh_options.add_argument("-N", type=int, default=40, help="Number of elements in x-direction")
    element_options = parser.add_argument_group("Finite element discretization options")
    element_options.add_argument(
        "--primal_degree",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help="Polynomial degree for primal variable",
    )
    alpha_options = parser.add_argument_group(
        "Options for alpha-variable in Proximal Galerkin scheme"
    )
    alpha_options.add_argument(
        "--alpha_scheme",
        type=str,
        default="linear",
        choices=["constant", "linear", "doubling"],
        help="Scheme for updating alpha",
    )
    alpha_options.add_argument("--alpha_0", type=float, default=1.0, help="Initial value of alpha")
    alpha_options.add_argument(
        "--alpha_c", type=float, default=1.0, help="Increment of alpha in linear scheme"
    )
    pg_options = parser.add_argument_group("Proximal Galerkin options")
    pg_options.add_argument(
        "--max_iterations", type=int, default=25, help="Maximum number of iterations"
    )
    pg_options.add_argument(
        "-s",
        "--stopping_tol",
        type=float,
        default=1e-8,
        help="Stopping tolerance between two successive PG iterations (L2-difference)",
    )
    result_options = parser.add_argument_group("Output options")
    result_options.add_argument(
        "--result_dir", type=Path, default=Path("results"), help="Directory to store results"
    )
    parsed_args = parser.parse_args(argv)

    iteration_counts, L2_diffs = solve_problem(
        N=parsed_args.N,
        primal_degree=parsed_args.primal_degree,
        alpha_scheme=AlphaScheme.from_string(parsed_args.alpha_scheme),
        alpha_0=parsed_args.alpha_0,
        alpha_c=parsed_args.alpha_c,
        max_iterations=parsed_args.max_iterations,
        result_dir=parsed_args.result_dir,
        stopping_tol=parsed_args.stopping_tol,
    )
    print(iteration_counts)
    print(L2_diffs)


if __name__ == "__main__":
    main()
