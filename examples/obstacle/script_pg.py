"""
    FEniCSx code to reproduce obstacle problem results in [1].

    [1] Keith, B. and Surowiec, T. (2023) Proximal Galerkin: A structure-
       preserving finite element method for pointwise bound constraints.
       arXiv:2307.12444 [math.NA]
"""

import argparse
from math import pi
from pathlib import Path

import numpy as np
import pandas as pd
import ufl
import basix
from dolfinx import fem, io, log, mesh
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.mesh import CellType, GhostMode
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from ufl import conditional, div, dot, dx, exp, grad, gt, inner, lt, sin
from petsc4py import PETSc

def rank_print(string: str, comm: MPI.Comm, rank: int = 0):
    """Helper function to print on a single rank

    :param string: String to print
    :param comm: The MPI communicator
    :param rank: Rank to print on, defaults to 0
    """
    if comm.rank == rank:
        print(string)


def allreduce_scalar(form: fem.Form, op: MPI.Op = MPI.SUM) -> np.floating:
    """Assemble a scalar form over all processes and perform a global reduction

    :param form: Scalar form
    :param op: MPI reduction operation
    """
    comm = form.mesh.comm
    return comm.allreduce(fem.assemble_scalar(form), op=op)


def solve_problem(
    example: int,
    m: int,
    polynomial_order: int,
    maximum_number_of_outer_loop_iterations: int,
    alpha_max: float,
    tol_exit: float,
):
    """
    Solve the obstacle problem in different example settings

    Example 1 exhibits two properties that are challenging for active set solvers:
    a) The transition from inactive to active is high order
    b) There is a nontrivial biactive set

    Example 2 is the non-pathological example.
    Here, we witness linear convergence with a fixed step size

    ..note::

        The exact solution is not known

    Example 3 is a second biactive example with a non-smooth multiplier

    """

    # Set problem specific parameters
    if example == 1:
        C = 1.0
        r = 1.5
        q = 1.5
        step_size_rule = "double_exponential"
    elif example == 2:
        C = 1.0
        r = 2
        step_size_rule = "geometric"
    elif example == 3:
        C = 1.0
        r = 1.2
        step_size_rule = "geometric"
    else:
        raise RuntimeError(f"Unknown example number {example}")

    # Create mesh
    num_cells_per_direction = 2**m
    msh = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((-1.0, -1.0), (1.0, 1.0)),
        n=(num_cells_per_direction, num_cells_per_direction),
        cell_type=CellType.triangle,
        ghost_mode=GhostMode.shared_facet,
    )

    # Define FE subspaces
    P = basix.ufl.element("Lagrange", msh.basix_cell(), polynomial_order)
    B = basix.ufl.element("Bubble", msh.basix_cell(), polynomial_order + 2)
    Pm1 = basix.ufl.element("Lagrange",msh.basix_cell(),polynomial_order-1, discontinuous=True )
    mixed_element= basix.ufl.mixed_element([basix.ufl.enriched_element(
    [P, B]), Pm1])
    V = fem.functionspace(msh, mixed_element)

    # Define functions and parameters
    alpha = fem.Constant(msh, ScalarType(1))
    x = ufl.SpatialCoordinate(msh)

    if example == 1:
        f = conditional(gt(x[0], 0.0), -12 * x[0] ** 2, fem.Constant(msh, 0.0))
        u_exact = conditional(gt(x[0], 0.0), x[0] ** 4, fem.Constant(msh, 0.0))
        lambda_exact = fem.Constant(msh, 0.0)

    elif example == 2:
        f = 2.0 * pi * pi * sin(pi * x[0]) * sin(pi * x[1])
        u_exact = fem.Constant(msh, 0.0)  # placeholder
        lambda_exact = fem.Constant(msh, 0.0)  # placeholder

    elif example == 3:
        z_1 = dot(x, x)
        z_2 = (x[0] ** 2 + x[1] ** 2 - 0.25) ** 2
        z_3 = 7 * x[0] ** 2 + x[1] ** 2 - 0.25
        z_4 = x[0] ** 2 + 7 * x[1] ** 2 - 0.25
        lambda_exact = conditional(
            lt(0.75, z_1), fem.Constant(msh, 1.0), fem.Constant(msh, 0.0)
        )
        f = (
            256
            * conditional(
                lt(z_1, 0.25), -8 * z_2 * z_3 - 8 *
                z_2 * z_4, fem.Constant(msh, 0.0)
            )
            - lambda_exact
        )
        u_exact = 256 * conditional(
            lt(z_1, 0.25), (0.25 - z_1) ** 4, fem.Constant(msh, 0.0)
        )

    # Define BCs
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    facets = mesh.exterior_facet_indices(msh.topology)
    V0, _ = V.sub(0).collapse()
    dofs = fem.locate_dofs_topological(
        (V.sub(0), V0), entity_dim=1, entities=facets)

    u_bc = fem.Function(V0)
    u_bc.interpolate(fem.Expression(
        u_exact, V.sub(0).element.interpolation_points()))
    bcs = fem.dirichletbc(value=u_bc, dofs=dofs, V=V.sub(0))

    # Define solution variables
    sol = fem.Function(V)
    sol_k = fem.Function(V)

    u, psi = ufl.split(sol)
    u_k, psi_k = ufl.split(sol_k)

    # Define non-linear residual
    (v, w) = ufl.TestFunctions(V)
    F = (
        alpha * inner(grad(u), grad(v)) * dx
        + psi * v * dx
        + u * w * dx
        - exp(psi) * w * dx
        - alpha * f * v * dx
        - psi_k * v * dx
    )
    J = ufl.derivative(F, sol)

    # Setup non-linear problem
    problem = NonlinearProblem(F, sol, bcs=[bcs], J=J)

    # Setup newton solver
    log.set_log_level(log.LogLevel.WARNING)
    newton_solver = NewtonSolver(
        comm=msh.comm, problem=problem)
    newton_solver.convergence_criterion = "incremental"
    #newton_solver.max_it = 50
    ksp = newton_solver.krylov_solver
    # petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
                                                                    #   "ksp_error_if_not_converged":True,
                                                                    #   "ksp_monitor":None,
    #                                                                  }
    petsc_options = {}
    opts = PETSc.Options()  # type: ignore
    option_prefix = ksp.getOptionsPrefix()
    opts.prefixPush(option_prefix)
    for k, v in petsc_options.items():
       opts[k] = v
    opts.prefixPop()
    ksp.setFromOptions()

    # observables
    energy_form = fem.form(0.5 * inner(grad(u), grad(u)) * dx - f * u * dx)
    complementarity_form = fem.form((psi_k - psi) / alpha * u * dx)
    feasibility_form = fem.form(conditional(
        lt(u, 0), -u, fem.Constant(msh, 0.0)) * dx)
    dual_feasibility_form = fem.form(
        conditional(lt(psi_k, psi), (psi - psi_k) /
                    alpha, fem.Constant(msh, 0.0)) * dx
    )
    H1increment_form = fem.form(
        inner(grad(u - u_k), grad(u - u_k)) * dx + (u - u_k) ** 2 * dx
    )
    L2increment_form = fem.form((exp(psi) - exp(psi_k)) ** 2 * dx)
    H1primal_error_form = fem.form(
        inner(grad(u - u_exact), grad(u - u_exact)) *
        dx + (u - u_exact) ** 2 * dx
    )
    L2primal_error_form = fem.form((u - u_exact) ** 2 * dx)
    L2latent_error_form = fem.form((exp(psi) - u_exact) ** 2 * dx)
    L2multiplier_error_form = fem.form(
        (lambda_exact - ((psi_k - psi) / alpha)) ** 2 * dx
    )

    # Proximal point outer loop
    n = 0
    increment_k = 0.0
    sol.x.array[:] = 0.0
    sol_k.x.array[:] = sol.x.array[:]
    alpha_k = C

    energies = []
    complementarities = []
    feasibilities = []
    dual_feasibilities = []
    H1primal_errors = []
    L2primal_errors = []
    L2latent_errors = []
    L2multiplier_errors = []
    Newton_steps = []
    step_sizes = []
    primal_increments = []
    latent_increments = []
    for k in range(maximum_number_of_outer_loop_iterations):
        # Update step size
        if step_size_rule == "constant":
            alpha.value = C
        elif step_size_rule == "double_exponential":
            try:
                alpha.value = max(C * r ** (q**k) - alpha_k, C)
            except OverflowError:
                pass
            alpha_k = alpha.value
            alpha.value = min(alpha.value, alpha_max)
        else:
            step_size_rule == "geometric"
            alpha.value = C * r**k
        rank_print(f"OUTER LOOP {k + 1} alpha: {alpha.value}", msh.comm)

        # Solve problem
        log.set_log_level(log.LogLevel.INFO)
        (n, converged) = newton_solver.solve(sol)
        log.set_log_level(log.LogLevel.WARNING)
        rank_print(f"Newton steps: {n}   Converged: {converged}", msh.comm)

        # Check outer loop convergence
        energy = allreduce_scalar(energy_form)
        complementarity = np.abs(allreduce_scalar(complementarity_form))
        feasibility = allreduce_scalar(feasibility_form)
        dual_feasibility = allreduce_scalar(dual_feasibility_form)
        increment = np.sqrt(allreduce_scalar(H1increment_form))
        latent_increment = np.sqrt(allreduce_scalar(L2increment_form))
        H1primal_error = np.sqrt(allreduce_scalar(H1primal_error_form))
        L2primal_error = np.sqrt(allreduce_scalar(L2primal_error_form))
        L2latent_error = np.sqrt(allreduce_scalar(L2latent_error_form))
        L2multiplier_error = np.sqrt(allreduce_scalar(L2multiplier_error_form))

        tol_Newton = increment

        rank_print(f"‖u - uₕ‖_H¹: {H1primal_error}" +
                   f"  ‖u - ũₕ‖_L² : {L2latent_error}", msh.comm)

        if increment_k > 0.0:
            rank_print(f"Increment size: {increment}" +
                       f"   Ratio: {increment / increment_k}", msh.comm)
        else:
            rank_print(f"Increment size: {increment}", msh.comm)
        rank_print("", msh.comm)

        energies.append(energy)
        complementarities.append(complementarity)
        feasibilities.append(feasibility)
        dual_feasibilities.append(dual_feasibility)
        H1primal_errors.append(H1primal_error)
        L2primal_errors.append(L2primal_error)
        L2latent_errors.append(L2latent_error)
        L2multiplier_errors.append(L2multiplier_error)
        Newton_steps.append(n)
        step_sizes.append(np.copy(alpha.value))
        primal_increments.append(increment)
        latent_increments.append(latent_increment)
        if tol_Newton < tol_exit:
            break

        # Reset Newton solver options
        newton_solver.atol = 1e-3
        newton_solver.rtol = tol_Newton * 1e-4

        # Update sol_k with sol_new
        sol_k.x.array[:] = sol.x.array[:]
        increment_k = increment

    # Save data
    cwd = Path.cwd()
    output_dir = cwd / "output"
    output_dir.mkdir(exist_ok=True)

    df = pd.DataFrame()
    df["Energy"] = energies
    df["Complementarity"] = complementarities
    df["Feasibility"] = feasibilities
    df["Dual Feasibility"] = dual_feasibilities
    df["H1 Primal errors"] = H1primal_errors
    df["L2 Primal errors"] = L2primal_errors
    df["L2 Latent errors"] = L2latent_errors
    df["L2 Multiplier errors"] = L2multiplier_errors
    df["Newton steps"] = Newton_steps
    df["Step sizes"] = step_sizes
    df["Primal increments"] = primal_increments
    df["Latent increments"] = latent_increments
    df["Polynomial order"] = [polynomial_order] * (k + 1)
    df["Mesh size"] = [1 / 2 ** (m - 1)] * (k + 1)
    df["dofs"] = [np.size(sol_k.x.array[:])] * (k + 1)
    df["Step size rule"] = [step_size_rule] * (k + 1)
    filename = f"./example{example}_polyorder{polynomial_order}_m{m}.csv"
    rank_print(f"Saving data to: {str(output_dir / filename)}", msh.comm)
    df.to_csv(output_dir / filename, index=False)
    rank_print(df, msh.comm)

    # Create output space for bubble function
    V_out = fem.functionspace(
        msh, basix.ufl.element(
            "Lagrange", msh.basix_cell(), polynomial_order + 2)
    )
    u_out = fem.Function(V_out)
    u_out.interpolate(sol.sub(0).collapse())

    # Export primal solution variable
    # Use VTX to capture high-order dofs
    with io.VTXWriter(msh.comm, output_dir / "u.bp", [u_out]) as vtx:
        vtx.write(0.0)

    # Export interpolant of exact solution u
    V_alt = fem.functionspace(msh, ("Lagrange", polynomial_order))
    q = fem.Function(V_alt)
    expr = fem.Expression(u_exact, V_alt.element.interpolation_points())
    q.interpolate(expr)
    with io.VTXWriter(msh.comm, output_dir / "u_exact.bp", [q]) as vtx:
        vtx.write(0.0)

    # Export interpolant of Lagrange multiplier λ
    W_out = fem.functionspace(msh, ("DG", max(1, polynomial_order - 1)))
    q = fem.Function(W_out)
    expr = fem.Expression(lambda_exact, W_out.element.interpolation_points())
    q.interpolate(expr)
    with io.VTXWriter(msh.comm, output_dir / "lambda_exact.bp", [q]) as vtx:
        vtx.write(0.0)

    # Export latent solution variable
    q = fem.Function(W_out)
    expr = fem.Expression(sol.sub(1), W_out.element.interpolation_points())
    q.interpolate(expr)
    with io.VTXWriter(msh.comm, output_dir / "psi.bp", [q]) as vtx:
        vtx.write(0.0)

    # Export feasible discrete solution
    exp_psi = exp(sol.sub(1))
    expr = fem.Expression(exp_psi, W_out.element.interpolation_points())
    q.interpolate(expr)
    with io.VTXWriter(msh.comm, output_dir / "tilde_u.bp", [q]) as vtx:
        vtx.write(0.0)

    # Export "Lagrange multiplier"
    lam = (sol_k.sub(1) - sol.sub(1)) / alpha.value
    expr = fem.Expression(lam, W_out.element.interpolation_points())
    q.interpolate(expr)
    with io.VTXWriter(msh.comm, output_dir / "lambda.bp", [q]) as vtx:
        vtx.write(0.0)

# -------------------------------------------------------
if __name__ == "__main__":
    desc = "Run examples from paper"
    parser = argparse.ArgumentParser(
        description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--example",
        "-e",
        dest="example",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="The example number",
    )
    parser.add_argument(
        "--mesh-density",
        "-m",
        dest="m",
        type=int,
        default=5,
        help="MESH DENSITY (m = 2 corresponds to h_∞)",
    )
    parser.add_argument(
        "--polynomial_order",
        "-p",
        dest="polynomial_order",
        type=int,
        default=1,
        choices=[1, 2],
        help="Polynomial order of primal space",
    )
    parser.add_argument(
        "--max-iter",
        "-i",
        dest="maximum_number_of_outer_loop_iterations",
        type=int,
        default=100,
        help="Maximum number of outer loop iterations",
    )
    parser.add_argument(
        "--alpha-max",
        "-a",
        dest="alpha_max",
        type=float,
        default=1e3,
        help="Maximum alpha",
    )
    parser.add_argument(
        "--tol",
        "-t",
        dest="tol_exit",
        type=float,
        default=1e-10,
        help="Tolerance for exiting Newton iteration",
    )
    args = parser.parse_args()
    solve_problem(
        args.example,
        args.m,
        args.polynomial_order,
        args.maximum_number_of_outer_loop_iterations,
        args.alpha_max,
        args.tol_exit,
    )
