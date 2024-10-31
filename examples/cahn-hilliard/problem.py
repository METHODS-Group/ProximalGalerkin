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

    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, N, M, cell_type=_cell_type, diagonal=dolfinx.mesh.DiagonalType.crossed
    )

    el_0 = basix.ufl.element(
        "P", mesh.topology.cell_name(), primal_degree, shape=(num_species,))
    el_1 = basix.ufl.element(
        "P", mesh.topology.cell_name(), primal_degree, shape=(num_species,))

    # el_s = basix.ufl.element("P", mesh.topology.cell_name(), primal_degree)
    # el_b = basix.ufl.element(
    #     "Bubble", mesh.topology.cell_name(), primal_degree + 2)
    # el_0 = basix.ufl.blocked_element(
    #     basix.ufl.enriched_element([el_s, el_b]), shape=(num_species,))
    # el_1 = basix.ufl.element(
    #     "DG", mesh.topology.cell_name(), primal_degree - 1, shape=(num_species,)
    # )

    # Trial space is (u, z, psi)
    trial_el = basix.ufl.mixed_element([el_0, el_0, el_1])
    V_trial = dolfinx.fem.functionspace(mesh, trial_el)
    V_test = V_trial

    sol = dolfinx.fem.Function(V_trial)
    u, z, psi = ufl.split(sol)

    v, y, w = ufl.TestFunctions(V_test)

    dx = ufl.Measure("dx", domain=mesh)
    alpha = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(alpha_0))
    epsilon = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.05))

    w_old = dolfinx.fem.Function(V_trial)
    _, _, psi_old = ufl.split(w_old)

    w_prev = dolfinx.fem.Function(V_trial)
    u_prev, _, _ = ufl.split(w_prev)

    _, c_to_V = V_trial.sub(0).collapse()
    _, psi_to_V = V_trial.sub(2).collapse()

    i, k = ufl.indices(2)
    # EQ 1
    F = alpha * z[i] * y[i] * dx
    F += epsilon**2 * alpha * ufl.grad(u[i])[k] * ufl.grad(y[i])[k] * dx

    F += -2 * alpha * u[i] * y[i] * dx
    F += psi[i] * y[i] * dx
    F -= psi_old[i] * y[i] * dx
    F -= alpha * sum(y[m] * dx for m in range(num_species))

    # EQ 2
    tau = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.00001))
    F += u[i] * v[i] * dx
    F -= tau * ufl.grad(z[i])[k] * ufl.grad(v[i])[k] * dx
    F -= u_prev[i] * v[i] * dx

    # EQ 3
    # eps_0 = 0
    # eps = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(eps_0))
    sum_psi = sum(ufl.exp(psi[m]) for m in range(num_species))
    F += (
        sum((u[m] - ufl.exp(psi[m]) / sum_psi) * w[m]  # - eps * psi[m]
            for m in range(num_species))
        * dx
    )

    # Random values between 0 and 1 that sum to 1
    # NOTE: We could use a e_i species distribution in squares, which should evolve
    # into "hexagonal" patterns
    if num_species == 1:
        num_dofs = len(w_prev.sub(0).collapse().x.array[:])
    else:
        num_dofs = len(w_prev.sub(0).sub(0).collapse().x.array[:])
    np.random.seed(12)
    rands = np.random.rand(num_dofs, num_species)
    norm_rand = np.linalg.norm(rands, axis=1, ord=1)
    w_prev.x.array[c_to_V] = (rands / norm_rand.reshape(-1, 1)).reshape(-1)
    bcs = []
    problem = NonlinearProblem(F, sol, bcs=bcs)
    solver = NewtonSolver(mesh.comm, problem)
    solver.convergence_criterion = "residual"
    solver.rtol = 1e-8
    solver.atol = 1e-8
    solver.max_it = 10
    solver.error_on_nonconvergence = False

    ksp = solver.krylov_solver
    opts = PETSc.Options()  # type: ignore
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "preonly"
    opts[f"{option_prefix}pc_type"] = "lu"
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
    opts[f"{option_prefix}ksp_error_if_not_converged"] = True
    # Increase MUMPS working memory
    opts[f"{option_prefix}mat_mumps_icntl_14"] = 500
    # opts[f"{option_prefix}ksp_view"] = None
    ksp.setFromOptions()
    dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

    V_out = dolfinx.fem.functionspace(
        mesh, ("Lagrange", primal_degree + 2, (num_species,)))
    c_out = dolfinx.fem.Function(V_out)
    bp_c = dolfinx.io.VTXWriter(
        mesh.comm, result_dir / "u.bp", [c_out], engine="BP4")
    c_out.interpolate(w_prev.sub(0).collapse())
    bp_c.write(0)

    psi_space, psi_to_V = V_trial.sub(2).collapse()
    psi_out = dolfinx.fem.Function(psi_space)
    bp_psi = dolfinx.io.VTXWriter(
        mesh.comm, result_dir / "psi.bp", [psi_out], engine="BP4")
    bp_psi.write(0)

    diff = sol.sub(0) - w_old.sub(0)

    L2_squared = ufl.dot(diff, diff) * dx
    compiled_diff = dolfinx.fem.form(L2_squared)

    num_steps = 100
    newton_iterations = np.zeros(num_steps, dtype=np.int32)
    T = num_steps * float(tau)
    t = 0
    for j in range(num_steps):
        t += float(tau)
        # eps.value = eps_0
        sol.x.array[c_to_V] = 1.0 / num_species
        sol.x.array[psi_to_V] = 0
        w_old.x.array[psi_to_V] = 0
        for i in range(1, max_iterations + 1):
            L2_diff = np.zeros(max_iterations, dtype=np.float64)

            if alpha_scheme == AlphaScheme.constant:
                pass
            elif alpha_scheme == AlphaScheme.linear:
                alpha.value = alpha_0 + alpha_c * i
            elif alpha_scheme == AlphaScheme.doubling:
                alpha.value = alpha_0 * 2**i

            num_newton_iterations, converged = solver.solve(sol)

            local_diff = dolfinx.fem.assemble_scalar(compiled_diff)
            global_diff = np.sqrt(mesh.comm.allreduce(local_diff, op=MPI.SUM))
            L2_diff[i - 1] = global_diff
            if mesh.comm.rank == 0:
                print(
                    f"Iteration {i}: {converged=} {num_newton_iterations=}",
                    f"|delta u |= {global_diff}",
                    f"{ksp.getConvergedReason()=}",
                )

            # ksp.view()
            # print(ksp.getConvergedReason())
            # Update solution
            w_old.x.array[:] = sol.x.array[:]
            # bp_grad_u.write(i)
            if global_diff < stopping_tol:
                break

        w_prev.x.array[:] = sol.x.array[:]
        c_out.interpolate(sol.sub(0).collapse())
        psi_out.x.array[:] = sol.x.array[psi_to_V]
        bp_c.write(t)
        bp_psi.write(t)
        # eps.value *= 0.5
    bp_c.close()
    bp_psi.close()
    return newton_iterations, L2_diff


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def main(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(formatter_class=CustomFormatter)
    parser.add_argument("--num_species", type=int,
                        default=3, help="Number of species")
    mesh_options = parser.add_argument_group("Mesh options")
    mesh_options.add_argument(
        "-N", type=int, default=40, help="Number of elements in x-direction")
    mesh_options.add_argument(
        "-M", type=int, default=40, help="Number of elements in y-direction")
    mesh_options.add_argument(
        "--cell_type",
        "-c",
        type=str,
        default="triangle",
        choices=["triangle", "quadrilateral"],
        help="Cell type",
    )
    element_options = parser.add_argument_group(
        "Finite element discretization options")
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
        default="constant",
        choices=["constant", "linear", "doubling"],
        help="Scheme for updating alpha",
    )
    alpha_options.add_argument(
        "--alpha_0", type=float, default=1.0, help="Initial value of alpha")
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
