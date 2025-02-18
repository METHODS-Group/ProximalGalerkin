import argparse
from pathlib import Path
from typing import Callable, Literal, Optional, get_args

from mpi4py import MPI
from petsc4py import PETSc

import basix.ufl
import dolfinx
import numpy as np
import ufl
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

AlphaScheme = Literal["constant", "linear", "doubling"]


def solve_problem(
    N: int,
    M: int,
    primal_space: str,
    latent_space: str,
    primal_degree: int,
    cell_type: str,
    alpha_scheme: AlphaScheme,
    alpha_0: float,
    alpha_c: float,
    max_iterations: int,
    stopping_tol: float,
    result_dir: Path,
    phi_func: Callable[[np.ndarray], np.ndarray],
    f_func: Callable[[np.ndarray], np.ndarray],
    warm_start: bool,
):
    _cell_type = dolfinx.mesh.to_type(cell_type)

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, M, cell_type=_cell_type)

    el_0 = basix.ufl.element(primal_space, mesh.topology.cell_name(), primal_degree)
    if latent_space == "RT":
        el_1 = basix.ufl.element("RT", mesh.topology.cell_name(), primal_degree - 1)
    elif latent_space == "DG":
        el_1 = basix.ufl.element(
            "DG", mesh.topology.cell_name(), primal_degree - 1, shape=(mesh.geometry.dim,)
        )
    elif latent_space == "Lagrange":
        assert primal_degree > 1
        el_1 = basix.ufl.element(
            "Lagrange", mesh.topology.cell_name(), primal_degree - 1, shape=(mesh.geometry.dim,)
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
    phi = dolfinx.fem.Function(U)
    phi.interpolate(phi_func)

    w0 = dolfinx.fem.Function(V_trial)

    f = dolfinx.fem.Function(U)
    f.interpolate(f_func)

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)

    _, Q_to_W = V_trial.sub(1).collapse()
    boundary_dofs = dolfinx.fem.locate_dofs_topological(
        (V_trial.sub(0), U), mesh.topology.dim - 1, boundary_facets
    )

    if warm_start:
        # Create initial condition
        p = ufl.TrialFunction(U)
        q = ufl.TestFunction(U)
        a = ufl.inner(ufl.grad(p), ufl.grad(q)) * dx
        L = f * q * dx
        u_bc_init = dolfinx.fem.Function(U)
        bc_init = dolfinx.fem.dirichletbc(u_bc_init, boundary_dofs[1])
        lin_prob = LinearProblem(a, L, bcs=[bc_init])
        u_init_out = lin_prob.solve()
        u_init_out.name = "InitialU"
        # u init is equal to the solution of the linear problem
        sol.x.array[U_to_W] = u_init_out.x.array
        sol.x.array[Q_to_W] = 0.0

    _, psi0 = ufl.split(w0)

    F = alpha * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    F += ufl.inner(psi, ufl.grad(v)) * dx
    F -= alpha * ufl.inner(f, v) * dx
    F -= ufl.inner(psi0, ufl.grad(v)) * dx

    F += ufl.inner(ufl.grad(u), w) * dx
    non_lin_term = 1 / (ufl.sqrt(1 + ufl.dot(psi, psi)))
    F -= phi * non_lin_term * ufl.dot(psi, w) * dx

    u_bc = dolfinx.fem.Function(U)
    u_bc.x.array[:] = 0
    bcs = [dolfinx.fem.dirichletbc(u_bc, boundary_dofs, V_trial.sub(0))]

    problem = NonlinearProblem(F, sol, bcs=bcs)
    solver = NewtonSolver(mesh.comm, problem)
    solver.convergence_criterion = "residual"
    solver.rtol = 1e-9
    solver.atol = 1e-9
    solver.max_it = 20
    solver.error_on_nonconvergence = True

    ksp = solver.krylov_solver
    opts = PETSc.Options()  # type: ignore
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "preonly"
    opts[f"{option_prefix}pc_type"] = "lu"
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "superlu"
    ksp.setFromOptions()

    dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

    W = dolfinx.fem.functionspace(mesh, ("DG", (primal_degree - 1), (mesh.geometry.dim,)))
    global_feasible_gradient = phi * psi / ufl.sqrt(1 + ufl.dot(psi, psi))
    feas_grad = dolfinx.fem.Expression(global_feasible_gradient, W.element.interpolation_points())
    pg_grad = dolfinx.fem.Function(W)
    pg_grad.name = "Global feasible gradient"

    u_out = sol.sub(0).collapse()
    u_out.name = "u"
    bp_u = dolfinx.io.VTXWriter(mesh.comm, result_dir / "u.bp", [u_out], engine="BP4")
    grad_u = dolfinx.fem.Function(W)
    grad_u.name = "grad(u)"
    grad_u_expr = dolfinx.fem.Expression(ufl.grad(u), W.element.interpolation_points())
    bp_grad_u = dolfinx.io.VTXWriter(
        mesh.comm, result_dir / "grad_u.bp", [grad_u, pg_grad], engine="BP4"
    )
    diff = sol.sub(0) - w0.sub(0)
    L2_squared = ufl.dot(diff, diff) * dx
    compiled_diff = dolfinx.fem.form(L2_squared)
    newton_iterations = np.zeros(max_iterations, dtype=np.int32)
    L2_diff = np.zeros(max_iterations, dtype=np.float64)
    for i in range(max_iterations):
        if alpha_scheme == "constant":
            pass
        elif alpha_scheme == "linear":
            alpha.value = alpha_0 + alpha_c * i
        elif alpha_scheme == "doubling":
            alpha.value = alpha_0 * 2**i

        num_newton_iterations, converged = solver.solve(sol)
        newton_iterations[i] = num_newton_iterations
        local_diff = dolfinx.fem.assemble_scalar(compiled_diff)
        global_diff = np.sqrt(mesh.comm.allreduce(local_diff, op=MPI.SUM))
        L2_diff[i] = global_diff
        if mesh.comm.rank == 0:
            print(
                f"Iteration {i}: {converged=} {num_newton_iterations=} {ksp.getConvergedReason()=}",
                f"|delta u |= {global_diff}",
            )

        u_out.x.array[:] = sol.sub(0).x.array[U_to_W]
        grad_u.interpolate(grad_u_expr)
        pg_grad.interpolate(feas_grad)
        bp_u.write(i)
        bp_grad_u.write(i)

        if global_diff < stopping_tol:
            break

        w0.x.array[:] = sol.x.array
    bp_u.close()
    bp_grad_u.close()
    return newton_iterations[: i + 1], L2_diff[: i + 1]


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


latent_spaces = Literal["Lagrange", "RT", "DG"]


def main(
    argv: Optional[list[str]] = None,
    phi_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    f_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
):
    parser = argparse.ArgumentParser(formatter_class=CustomFormatter)
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
        "--primal_space",
        type=str,
        default="Lagrange",
        choices=["Lagrange", "P", "CG"],
        help="Finite Element family for primal variable",
    )
    element_options.add_argument(
        "--latent_space",
        type=str,
        default="Lagrange",
        choices=get_args(latent_spaces),
        help="Finite element family for auxiliary variable",
    )
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
        choices=get_args(AlphaScheme),
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
        "--warm_start",
        action="store_true",
        help="Use warm start (solve Poisson problem to get initial guess)",
    )
    result_options = parser.add_argument_group("Output options")
    result_options.add_argument(
        "--result_dir", type=Path, default=Path("results"), help="Directory to store results"
    )
    parsed_args = parser.parse_args(argv)

    if phi_func is None:

        def phi_func(x):
            return 0.1 + 0.1 * x[0] + x[1] * 0.4
            # return np.full(x.shape[1], 0.1)

    if f_func is None:

        def f_func(x):
            # return np.full(x.shape[1], 1)
            return 10 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[0])

    iteration_counts, L2_diffs = solve_problem(
        N=parsed_args.N,
        M=parsed_args.M,
        primal_space=parsed_args.primal_space,
        latent_space=parsed_args.latent_space,
        primal_degree=parsed_args.primal_degree,
        cell_type=parsed_args.cell_type,
        alpha_scheme=parsed_args.alpha_scheme,
        alpha_0=parsed_args.alpha_0,
        alpha_c=parsed_args.alpha_c,
        max_iterations=parsed_args.max_iterations,
        result_dir=parsed_args.result_dir,
        phi_func=phi_func,
        f_func=f_func,
        warm_start=parsed_args.warm_start,
        stopping_tol=parsed_args.stopping_tol,
    )
    print(f"Number of LVPP iterations {len(iteration_counts)}")
    print(f"Minimum number of solves {np.min(iteration_counts)}")
    print(f"Maximum number of solves {np.max(iteration_counts)}")
    print(f"Total number of Newton iterations: {np.sum(iteration_counts)}")
    print(iteration_counts)
    print(L2_diffs)


if __name__ == "__main__":
    main()
