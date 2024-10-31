import argparse
from enum import Enum
from pathlib import Path
from typing import Optional

from mpi4py import MPI
from petsc4py import PETSc

import basix.ufl
import dolfinx
import numpy as np
import ufl
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver


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
    M: int,
    num_species: int,
    primal_degree: int,
    cell_type: str,
    alpha_scheme: AlphaScheme,
    alpha_0: float,
    alpha_c: float,
    max_iterations: int,
    stopping_tol: float,
    result_dir: Path,
):
    _cell_type = dolfinx.mesh.to_type(cell_type)

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, M, cell_type=_cell_type)

    # el_0 = basix.ufl.element(
    #    "P", mesh.topology.cell_name(), primal_degree+1, shape=(num_species,))
    el_s = basix.ufl.element("P", mesh.topology.cell_name(), primal_degree)
    el_b = basix.ufl.element("Bubble", mesh.topology.cell_name(), primal_degree + 2)
    el_0 = basix.ufl.blocked_element(basix.ufl.enriched_element([el_s, el_b]), shape=(num_species,))
    el_1 = basix.ufl.element(
        "DG", mesh.topology.cell_name(), primal_degree - 1, shape=(num_species,)
    )

    trial_el = basix.ufl.mixed_element([el_0, el_1])
    V_trial = dolfinx.fem.functionspace(mesh, trial_el)
    V_test = V_trial

    sol = dolfinx.fem.Function(V_trial)
    u, psi = ufl.split(sol)

    v, w = ufl.TestFunctions(V_test)

    dx = ufl.Measure("dx", domain=mesh)
    alpha = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(alpha_0))
    U, U_to_W = V_trial.sub(0).collapse()
    phi = dolfinx.fem.Function(U)  # Previous iterate

    tol = 1e-14
    for i in range(num_species):
        sol.sub(0).sub(i).interpolate(
            lambda x: np.full(x.shape[1], 1 / num_species, dtype=dolfinx.default_scalar_type)
        )

    h = ufl.Circumradius(mesh)
    epsilon = ufl.sqrt(h)
    ones = dolfinx.fem.Constant(
        mesh,
        dolfinx.default_scalar_type(
            [
                1,
            ]
            * num_species
        ),
    )
    F = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    i = ufl.indices(1)
    F += 1 / (epsilon**2) * ((ones[i] - u[i]) + 1 / alpha * psi[i]) * v[i] * dx

    F -= 1 / alpha * phi[i] * v[i] * dx
    sum_psi = sum(ufl.exp(psi[j]) for j in range(num_species))
    F += sum((u[i] - ufl.exp(psi[i]) / sum_psi) * w[i] for i in range(num_species)) * dx

    bcs = []

    problem = NonlinearProblem(F, sol, bcs=bcs)
    solver = NewtonSolver(mesh.comm, problem)
    solver.convergence_criterion = "residual"
    solver.rtol = 1e-8
    solver.atol = 1e-8
    solver.max_it = 20
    solver.error_on_nonconvergence = True

    ksp = solver.krylov_solver
    opts = PETSc.Options()  # type: ignore
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "cg"
    opts[f"{option_prefix}pc_type"] = "lu"
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
    ksp.setFromOptions()

    dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

    # W = dolfinx.fem.functionspace(mesh, ("DG", (primal_degree-1), (mesh.geometry.dim, )))
    # global_feasible_gradient = phi * psi / ufl.sqrt(1+ ufl.dot(psi, psi))
    # feas_grad = dolfinx.fem.Expression(global_feasible_gradient, W.element.interpolation_points())
    # pg_grad = dolfinx.fem.Function(W)
    # pg_grad.name = "Global feasible gradient"
    V_out = dolfinx.fem.functionspace(mesh, ("Lagrange", primal_degree + 3, (num_species,)))
    u_out = dolfinx.fem.Function(V_out)
    bp_u = dolfinx.io.VTXWriter(mesh.comm, result_dir / "u.bp", [u_out], engine="BP4")

    u_out.interpolate(sol.sub(0).collapse())
    bp_u.write(0)
    # grad_u = dolfinx.fem.Function(W)
    # grad_u.name = "grad(u)"
    # grad_u_expr = dolfinx.fem.Expression(ufl.grad(u), W.element.interpolation_points())
    # bp_grad_u = dolfinx.io.VTXWriter(mesh.comm, result_dir / "grad_u.bp", [grad_u, pg_grad], engine="BP4")
    diff = sol.sub(0) - phi

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
        solver.rtol *= 0.5
        solver.atol *= 0.5

        num_newton_iterations, converged = solver.solve(sol)
        newton_iterations[i] = num_newton_iterations
        local_diff = dolfinx.fem.assemble_scalar(compiled_diff)
        global_diff = np.sqrt(mesh.comm.allreduce(local_diff, op=MPI.SUM))
        L2_diff[i - 1] = global_diff
        if mesh.comm.rank == 0:
            print(
                f"Iteration {i}: {converged=} {num_newton_iterations=} {ksp.getConvergedReason()=}",
                f"|delta u |= {global_diff}",
            )
        # Update solution
        phi.x.array[:] = sol.x.array[U_to_W]

        u_out.interpolate(sol.sub(0).collapse())
        bp_u.write(i)
        # bp_grad_u.write(i)
        if global_diff < stopping_tol:
            break

    bp_u.close()
    # bp_grad_u.close()
    return newton_iterations, L2_diff


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def main(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(formatter_class=CustomFormatter)
    parser.add_argument("--num_species", type=int, default=3, help="Number of species")
    mesh_options = parser.add_argument_group("Mesh options")
    mesh_options.add_argument("-N", type=int, default=40, help="Number of elements in x-direction")
    mesh_options.add_argument("-M", type=int, default=40, help="Number of elements in y-direction")
    mesh_options.add_argument(
        "--cell_type",
        "-c",
        type=str,
        default="triangle",
        choices=["triangle", "quadrilateral"],
        help="Cell type",
    )
    element_options = parser.add_argument_group("Finite element discretization options")
    element_options.add_argument(
        "--primal_degree",
        type=int,
        default=2,
        choices=[2, 3, 4, 5, 6, 7, 8],
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
        "--max_iterations", type=int, default=20, help="Maximum number of iterations"
    )
    pg_options.add_argument(
        "-s",
        "--stopping_tol",
        type=float,
        default=1e-9,
        help="Stopping tolerance between two successive PG iterations (L2-difference)",
    )
    result_options = parser.add_argument_group("Output options")
    result_options.add_argument(
        "--result_dir", type=Path, default=Path("results"), help="Directory to store results"
    )
    parsed_args = parser.parse_args(argv)

    iteration_counts, L2_diffs = solve_problem(
        N=parsed_args.N,
        M=parsed_args.M,
        num_species=parsed_args.num_species,
        primal_degree=parsed_args.primal_degree,
        cell_type=parsed_args.cell_type,
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
