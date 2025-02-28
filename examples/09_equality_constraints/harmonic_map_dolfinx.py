import argparse
from pathlib import Path
from typing import Literal, Optional

from mpi4py import MPI
from petsc4py import PETSc

import basix.ufl
import dolfinx
import numpy as np
import ufl
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

AlphaScheme = Literal["constant", "linear", "doubling"]


def solve_problem(
    N: int,
    primal_degree: int,
    alpha_scheme: AlphaScheme,
    alpha_0: float,
    alpha_c: float,
    max_iterations: int,
    stopping_tol: float,
    result_dir: Path,
    gamma: float,
):
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, N)

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

    one = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1))
    gamma_ = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(gamma))

    # Variational form
    F = alpha * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    F += gamma_ * (ufl.dot(psi, psi) - one) * ufl.inner(u, v) * dx
    F += ufl.inner(psi, v) * dx
    F -= ufl.inner(psi0, v) * dx

    F += ufl.inner(u, w) * dx

    non_lin_term = 1 / (ufl.sqrt(ufl.dot(psi, psi)))
    phi = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0))
    F -= phi * non_lin_term * ufl.dot(psi, w) * dx

    def bc_func(x, theta=5.7 * np.pi):
        u_x = np.cos(theta * x[0])
        u_z = 1 - 2 * x[0] - np.sin(0.8 * np.pi * x[0])
        u_y = np.sin(theta * x[0])
        return (u_x, u_y, u_z) / np.sqrt(u_x**2 + u_y**2 + u_z**2)

    # Create boundary conditions
    u_left = dolfinx.fem.Function(U)
    u_left.interpolate(bc_func)
    left_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[0], 0.0)
    )
    left_dofs = dolfinx.fem.locate_dofs_topological(
        (V.sub(0), U), mesh.topology.dim - 1, left_facets
    )
    u_right = dolfinx.fem.Function(U)
    u_right.interpolate(bc_func)

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
    sol.sub(0).interpolate(bc_func)
    sol.sub(1).interpolate(bc_func)
    problem = NonlinearProblem(F, sol, bcs=bcs)
    solver = NewtonSolver(mesh.comm, problem)
    solver.convergence_criterion = "residual"
    solver.rtol = 1e-9
    solver.atol = 1e-9
    solver.max_it = 100
    solver.error_on_nonconvergence = True

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
        if alpha_scheme == "constant":
            pass
        elif alpha_scheme == "linear":
            alpha.value = alpha_0 + alpha_c * i
        elif alpha_scheme == "doubling":
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

    return newton_iterations[: i + 1], L2_diff[: i + 1]


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def main(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(formatter_class=CustomFormatter)
    mesh_options = parser.add_argument_group("Mesh options")
    mesh_options.add_argument(
        "-N", type=int, default=1000, help="Number of elements in x-direction"
    )
    element_options = parser.add_argument_group("Finite element discretization options")
    element_options.add_argument(
        "--primal_degree",
        type=int,
        default=2,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help="Polynomial degree for primal variable",
    )
    alpha_options = parser.add_argument_group(
        "Options for alpha-variable in Proximal Galerkin scheme"
    )
    alpha_options.add_argument(
        "--alpha_scheme",
        type=str,
        default="constant",
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
    pg_options.add_argument(
        "-g", "--gamma", dest="gamma", type=float, default=1e5, help="Gamma parameter"
    )
    result_options = parser.add_argument_group("Output options")
    result_options.add_argument(
        "--result_dir", type=Path, default=Path("results"), help="Directory to store results"
    )
    parsed_args = parser.parse_args(argv)

    iteration_counts, L2_diffs = solve_problem(
        N=parsed_args.N,
        primal_degree=parsed_args.primal_degree,
        alpha_scheme=parsed_args.alpha_scheme,
        alpha_0=parsed_args.alpha_0,
        alpha_c=parsed_args.alpha_c,
        max_iterations=parsed_args.max_iterations,
        result_dir=parsed_args.result_dir,
        stopping_tol=parsed_args.stopping_tol,
        gamma=parsed_args.gamma,
    )
    print(iteration_counts)
    print(L2_diffs)


if __name__ == "__main__":
    main()
