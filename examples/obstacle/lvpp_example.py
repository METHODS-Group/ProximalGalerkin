"""

Obstacle problem based of example 3 (P3) in Hintermüller and Kunisch [1].

FEniCSx code solve this problem is based of the paper [2,3]:

SPXD License: MIT License

Original license file [../../licenses/LICENSE.surowiec](../../licenses/LICENSE.surowiec)
is included in the repository.

[1] Hintermüller, M. and Kunisch K., Path-following Methods for a Class of
    Constrained Minimization Problems in Function Space, SIAM Journal on Optimization 2006,
    https://doi.org/10.1137/040611598
[2] Keith, B. and Surowiec, T.M., Proximal Galerkin: A Structure-Preserving Finite Element Method
for Pointwise Bound Constraints. Found Comput Math (2024). https://doi.org/10.1007/s10208-024-09681-8
[3] Keith, B., Surowiec, T. M., & Dokken, J. S. (2023). Examples for the Proximal Galerkin Method
    (Version 0.1.0) [Computer software]. https://github.com/thomas-surowiec/proximal-galerkin-examples
"""

import argparse
from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc

import basix
import numpy as np
import pandas as pd
import ufl
from dolfinx import default_scalar_type, fem, io, log, mesh
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from ufl import Measure, conditional, exp, grad, inner, lt


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
    filename: Path,
    polynomial_order: int,
    maximum_number_of_outer_loop_iterations: int,
    alpha_scheme: str,
    alpha_max: float,
    tol_exit: float,
):
    """ """

    # Create mesh
    with io.XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
        msh = xdmf.read_mesh(name="mesh")

    # Define FE subspaces
    P = basix.ufl.element("Lagrange", msh.basix_cell(), polynomial_order)
    mixed_element = basix.ufl.mixed_element([P, P])
    V = fem.functionspace(msh, mixed_element)

    # Define functions and parameters
    alpha = fem.Constant(msh, default_scalar_type(1))
    f = fem.Constant(msh, 0.0)
    # Define BCs
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    facets = mesh.exterior_facet_indices(msh.topology)
    V0, _ = V.sub(0).collapse()
    dofs = fem.locate_dofs_topological((V.sub(0), V0), entity_dim=1, entities=facets)

    u_bc = fem.Function(V0)
    u_bc.x.array[:] = 0.0
    bcs = fem.dirichletbc(value=u_bc, dofs=dofs, V=V.sub(0))

    # Define solution variables
    sol = fem.Function(V)
    sol_k = fem.Function(V)

    u, psi = ufl.split(sol)
    u_k, psi_k = ufl.split(sol_k)

    def phi_set(x):
        r = np.sqrt(x[0] ** 2 + x[1] ** 2)
        r0 = 0.5
        beta = 0.9
        b = r0 * beta
        tmp = np.sqrt(r0**2 - b**2)
        B = tmp + b * b / tmp
        C = -b / tmp
        cond_true = B + r * C
        cond_false = np.sqrt(r0**2 - r**2)
        true_indices = np.flatnonzero(r > b)
        cond_false[true_indices] = cond_true[true_indices]
        return cond_false

    quadrature_degree = 6
    Qe = basix.ufl.quadrature_element(msh.topology.cell_name(), degree=quadrature_degree)
    Vq = fem.functionspace(msh, Qe)
    # Lower bound for the obstacle
    phi = fem.Function(Vq, name="phi")
    phi.interpolate(phi_set)

    # Define non-linear residual
    (v, w) = ufl.TestFunctions(V)
    dx = Measure("dx", domain=msh, metadata={"quadrature_degree": quadrature_degree})
    F = (
        alpha * inner(grad(u), grad(v)) * dx
        + psi * v * dx
        + u * w * dx
        - exp(psi) * w * dx
        - phi * w * dx
        - alpha * f * v * dx
        - psi_k * v * dx
    )
    J = ufl.derivative(F, sol)

    # Setup non-linear problem
    problem = NonlinearProblem(F, sol, bcs=[bcs], J=J)

    # Setup newton solver
    log.set_log_level(log.LogLevel.WARNING)
    newton_solver = NewtonSolver(comm=msh.comm, problem=problem)
    newton_solver.convergence_criterion = "incremental"
    newton_solver.max_it = 100
    ksp = newton_solver.krylov_solver
    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "ksp_error_if_not_converged": True,
        "ksp_monitor": None,
    }
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
    feasibility_form = fem.form(conditional(lt(u, 0), -u, fem.Constant(msh, 0.0)) * dx)
    dual_feasibility_form = fem.form(
        conditional(lt(psi_k, psi), (psi - psi_k) / alpha, fem.Constant(msh, 0.0)) * dx
    )
    H1increment_form = fem.form(inner(grad(u - u_k), grad(u - u_k)) * dx + (u - u_k) ** 2 * dx)
    L2increment_form = fem.form((exp(psi) - exp(psi_k)) ** 2 * dx)

    # Proximal point outer loop
    n = 0
    increment_k = 0.0
    sol.x.array[:] = 0.0
    sol_k.x.array[:] = sol.x.array[:]
    alpha_k = 1
    step_size_rule = alpha_scheme
    C = 0.1
    r = 1.5
    q = 1.5

    energies = []
    complementarities = []
    feasibilities = []
    dual_feasibilities = []
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
        # log.set_log_level(log.LogLevel.WARNING)
        rank_print(f"Newton steps: {n}   Converged: {converged}", msh.comm)

        # Check outer loop convergence
        energy = allreduce_scalar(energy_form)
        complementarity = np.abs(allreduce_scalar(complementarity_form))
        feasibility = allreduce_scalar(feasibility_form)
        dual_feasibility = allreduce_scalar(dual_feasibility_form)
        increment = np.sqrt(allreduce_scalar(H1increment_form))
        latent_increment = np.sqrt(allreduce_scalar(L2increment_form))

        tol_Newton = increment

        if increment_k > 0.0:
            rank_print(
                f"Increment size: {increment}" + f"   Ratio: {increment / increment_k}", msh.comm
            )
        else:
            rank_print(f"Increment size: {increment}", msh.comm)
        rank_print("", msh.comm)

        energies.append(energy)
        complementarities.append(complementarity)
        feasibilities.append(feasibility)
        dual_feasibilities.append(dual_feasibility)
        Newton_steps.append(n)
        step_sizes.append(np.copy(alpha.value))
        primal_increments.append(increment)
        latent_increments.append(latent_increment)

        if tol_Newton < tol_exit:
            break

        # Reset Newton solver options
        newton_solver.atol = 1e-4
        newton_solver.rtol = tol_Newton * 1e-4

        # Update sol_k with sol_new
        sol_k.x.array[:] = sol.x.array[:]
        increment_k = increment

    # # Save data
    cwd = Path.cwd()
    output_dir = cwd / "output"
    output_dir.mkdir(exist_ok=True)

    # Create output space for bubble function
    V_primal, primal_to_mixed = V.sub(0).collapse()

    num_primal_dofs = V_primal.dofmap.index_map.size_global

    phi_out_space = fem.functionspace(msh, basix.ufl.element("Lagrange", msh.basix_cell(), 6))
    phi_out = fem.Function(phi_out_space, name="phi")
    phi_out.interpolate(phi_set)
    with io.VTXWriter(msh.comm, output_dir / "phi.bp", [phi_out]) as bp:
        bp.write(0.0)
    if MPI.COMM_WORLD.rank == 0:
        df = pd.DataFrame()
        df["Energy"] = energies
        df["Complementarity"] = complementarities
        df["Feasibility"] = feasibilities
        df["Dual Feasibility"] = dual_feasibilities
        df["Newton steps"] = Newton_steps
        df["Step sizes"] = step_sizes
        df["Primal increments"] = primal_increments
        df["Latent increments"] = latent_increments
        df["Polynomial order"] = np.full(k + 1, polynomial_order)
        df["dofs"] = np.full(k + 1, num_primal_dofs)
        df["Step size rule"] = [step_size_rule] * (k + 1)
        filename = f"./example_polyorder{polynomial_order}_{num_primal_dofs}.csv"
        print(f"Saving data to: {str(output_dir / filename)}")
        df.to_csv(output_dir / filename, index=False)
        rank_print(df, msh.comm)

    if k == maximum_number_of_outer_loop_iterations - 1:
        rank_print("Maximum number of outer loop iterations reached", msh.comm)
    return sol, k


# -------------------------------------------------------
if __name__ == "__main__":
    desc = "Run examples from paper"
    parser = argparse.ArgumentParser(
        description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--file-path",
        "-f",
        dest="filename",
        type=Path,
        required=True,
        help="Name of input file",
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
        "--alpha-scheme",
        dest="alpha_scheme",
        type=str,
        default="constant",
        choices=["constant", "double_exponential", "geometric"],
        help="Step size rule",
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
        default=1e5,
        help="Maximum alpha",
    )
    parser.add_argument(
        "--tol",
        "-t",
        dest="tol_exit",
        type=float,
        default=1e-6,
        help="Tolerance for exiting Newton iteration",
    )
    args = parser.parse_args()
    solve_problem(
        args.filename,
        args.polynomial_order,
        args.maximum_number_of_outer_loop_iterations,
        args.alpha_scheme,
        args.alpha_max,
        args.tol_exit,
    )
