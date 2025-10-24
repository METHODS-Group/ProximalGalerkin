import argparse
from pathlib import Path
from typing import Literal, Optional, get_args

from mpi4py import MPI
from petsc4py import PETSc

import basix.ufl
import dolfinx.fem.petsc
import numpy as np
import ufl

AlphaScheme = Literal["constant", "linear", "doubling"]


def solve_problem(
    N: int,
    M: int,
    primal_degree: int,
    cell_type: str,
    alpha_max: float,
    alpha_scheme: AlphaScheme,
    alpha_0: float,
    alpha_c: float,
    max_iterations: int,
    stopping_tol: float,
    result_dir: Path,
    write_frequency: int,
    tau0: float,
    T: float,
):
    _cell_type = dolfinx.mesh.to_type(cell_type)

    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, N, M, cell_type=_cell_type, diagonal=dolfinx.mesh.DiagonalType.crossed
    )
    num_species = 4
    el_0 = basix.ufl.element("P", mesh.topology.cell_name(), primal_degree, shape=(num_species,))
    el_1 = basix.ufl.element("P", mesh.topology.cell_name(), primal_degree, shape=(num_species,))

    # Trial space is (u, z, psi)
    trial_el = basix.ufl.mixed_element([el_0, el_0, el_1])
    V_trial = dolfinx.fem.functionspace(mesh, trial_el)

    sol = dolfinx.fem.Function(V_trial)
    u, z, psi = ufl.split(sol)

    v, y, w = ufl.TestFunctions(V_trial)

    dx = ufl.Measure("dx", domain=mesh)
    alpha = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(alpha_0))
    h = 2 * ufl.Circumradius(mesh)
    epsilon = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(2)) * h

    C, c_to_V = V_trial.sub(0).collapse()

    u_prev = dolfinx.fem.Function(C)  # u from previous time step

    # psi from previous LVPP iteration
    lvpp_old = dolfinx.fem.Function(V_trial)
    psi_old = lvpp_old.sub(2)
    u_old = dolfinx.fem.Function(C)  # u from previous LVPP iteration

    i, k = ufl.indices(2)
    # EQ 1
    F = alpha * z[i] * y[i] * dx
    F += epsilon**2 * alpha * ufl.grad(u[i])[k] * ufl.grad(y[i])[k] * dx

    F += -2 * alpha * u[i] * y[i] * dx
    F += psi[i] * y[i] * dx
    F -= psi_old[i] * y[i] * dx
    F -= alpha * sum(y[m] * dx for m in range(num_species))

    # EQ 2
    tau = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(tau0))
    F += u[i] * v[i] * dx
    F -= tau * ufl.grad(z[i])[k] * ufl.grad(v[i])[k] * dx
    F -= u_prev[i] * v[i] * dx

    # EQ 3
    eps_0 = 1e-9
    eps = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(eps_0))
    sum_psi = sum(ufl.exp(psi[m]) for m in range(num_species))
    F += (
        sum(
            (u[m] - ufl.exp(psi[m]) / sum_psi) * w[m] - eps * psi[m] * w[m]
            for m in range(num_species)
        )
        * dx
    )

    def rectangle(x, tol=1e-14):
        return (
            (0.2 - tol <= x[1]) & (x[1] <= 0.75 + tol) & (0.2 - tol <= x[0]) & (x[0] <= 0.8 + tol)
        )

    def lower_left(x, tol=1e-14):
        return (x[1] <= 0.5 + tol) & (0.2 - tol <= x[1]) & (0.2 - tol <= x[0]) & (x[0] <= 0.5 + tol)

    def lower_right(x, tol=1e-14):
        return (x[1] <= 0.5 + tol) & (0.2 <= x[1] + tol) & (0.5 - tol <= x[0]) & (x[0] <= 0.8 + tol)

    def field1(x):
        values = np.zeros((num_species, x.shape[1]))
        values[1] = 1.0
        return values

    def field2(x):
        values = np.zeros((num_species, x.shape[1]))
        values[2] = 1.0
        return values

    def field3(x):
        values = np.zeros((num_species, x.shape[1]))
        values[3] = 1.0
        return values

    cells1 = dolfinx.mesh.locate_entities(mesh, mesh.topology.dim, rectangle)
    cells2 = dolfinx.mesh.locate_entities(mesh, mesh.topology.dim, lower_left)
    cells3 = dolfinx.mesh.locate_entities(mesh, mesh.topology.dim, lower_right)
    u_prev.x.array[::4] = 1.0
    u_prev.interpolate(field1, cells0=cells1)
    u_prev.interpolate(field2, cells0=cells2)
    u_prev.interpolate(field3, cells0=cells3)
    u_prev.x.scatter_forward()

    bcs: list[dolfinx.fem.DirichletBC] = []
    petsc_options = {
        "snes_type": "newtonls",
        "snes_atol": 1e-8,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "ksp_error_if_not_converged": True,
        "snes_error_if_not_converged": True,
        "mat_mumps_icntl_14": 600,
        "mat_mumps_icntl_24": 1,
        "snes_monitor": None,
        "snes_atol": 1e-8,
        "snes_rtol": 1e-8,
        "snes_rtol": 1e-8,
        "snes_max_it": 25,
    }

    problem = dolfinx.fem.petsc.NonlinearProblem(
        F, u=sol, bcs=bcs, petsc_options=petsc_options, petsc_options_prefix="nls_"
    )

    bp_c = dolfinx.io.VTXWriter(
        mesh.comm,
        result_dir / "u.bp",
        [u_prev],
        engine="BP5",
        mesh_policy=dolfinx.io.VTXMeshPolicy.reuse,
    )
    bp_c.write(0)

    psi_space, psi_to_V = V_trial.sub(2).collapse()
    psi_out = dolfinx.fem.Function(psi_space)
    bp_psi = dolfinx.io.VTXWriter(
        mesh.comm,
        result_dir / "psi.bp",
        [psi_out],
        engine="BP5",
        mesh_policy=dolfinx.io.VTXMeshPolicy.reuse,
    )
    bp_psi.write(0)

    diff = sol.sub(0) - u_old

    L2_squared = ufl.dot(diff, diff) * dx
    compiled_diff = dolfinx.fem.form(L2_squared)

    num_steps = np.ceil(T / tau0).astype(np.int32)

    newton_iterations = np.zeros(num_steps, dtype=np.int32)
    lvpp_iterations = np.zeros(num_steps, dtype=np.int32)
    t = 0.0

    psi_scalar = [V_trial.sub(2).sub(i).collapse() for i in range(num_species)]
    psi_init = [
        dolfinx.fem.Expression(
            ufl.ln(abs(u[i]) + 1e-7) + 1, psi_scalar[i][0].element.interpolation_points
        )
        for i in range(num_species)
    ]
    psi_sub = [dolfinx.fem.Function(psi_scalar[i][0]) for i in range(num_species)]
    for j in range(1, num_steps + 1):
        if mesh.comm.rank == 0:
            print(f"Step {j}/{num_steps}", flush=True)
        t += float(tau)
        # Set psi_i = ln(u_i) + 1
        for i in range(num_species):
            psi_sub[i].interpolate(psi_init[i])
            sol.x.array[psi_scalar[i][1]] = psi_sub[i].x.array
            psi_old.x.array[psi_scalar[i][1]] = psi_sub[i].x.array

        u_old.x.array[:] = 0
        for i in range(1, max_iterations + 1):
            if alpha_scheme == "constant":
                pass
            elif alpha_scheme == "linear":
                alpha.value = min(alpha_0 + alpha_c * i, alpha_max)
            elif alpha_scheme == "doubling":
                alpha.value = min(alpha_0 * 2**i, alpha_max)
            problem.solve()
            num_iterations = problem.solver.getIterationNumber()
            converged = problem.solver.getConvergedReason()
            newton_iterations[j - 1] += num_iterations
            local_diff = dolfinx.fem.assemble_scalar(compiled_diff)
            global_diff = np.sqrt(mesh.comm.allreduce(local_diff, op=MPI.SUM))
            if mesh.comm.rank == 0:
                print(
                    f"Iteration {i}: {converged=} alpha={float(alpha):.2e} {num_iterations=}",
                    f"|delta u |= {global_diff}",
                    flush=True,
                )

            # ksp.view()
            # print(ksp.getConvergedReason())
            # Update solution
            u_old.x.array[:] = sol.x.array[c_to_V]
            psi_old.x.array[:] = sol.x.array
            if global_diff < stopping_tol:
                break

        u_prev.x.array[:] = sol.x.array[c_to_V]
        psi_out.x.array[:] = sol.x.array[psi_to_V]
        if j % write_frequency == 0:
            bp_c.write(t)
            bp_psi.write(t)

        lvpp_iterations[j - 1] += i
    bp_c.close()
    bp_psi.close()
    print("Newton iterations:", newton_iterations)
    print("LVPP iterations:", lvpp_iterations)
    return newton_iterations, lvpp_iterations


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def main(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(formatter_class=CustomFormatter)
    parser.add_argument("--dt", dest="tau0", type=float, default=1e-5, help="Time step")
    parser.add_argument("--T", dest="T", type=float, default=7e-3, help="End time")
    parser.add_argument("-l", "--logging", action="store_true", help="Enable logging")
    mesh_options = parser.add_argument_group("Mesh options")
    mesh_options.add_argument("-N", type=int, default=50, help="Number of elements in x-direction")
    mesh_options.add_argument("-M", type=int, default=50, help="Number of elements in y-direction")
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
        choices=get_args(AlphaScheme),
        help="Scheme for updating alpha",
    )
    alpha_options.add_argument("--alpha_0", type=float, default=1.0, help="Initial value of alpha")
    alpha_options.add_argument(
        "--alpha_c", type=float, default=1.0, help="Increment of alpha in linear scheme"
    )
    alpha_options.add_argument(
        "--alpha_max", type=float, default=50.0, help="Maximum value of alpha"
    )
    pg_options = parser.add_argument_group("Proximal Galerkin options")
    pg_options.add_argument(
        "--max_iterations", type=int, default=20, help="Maximum number of iterations"
    )
    pg_options.add_argument(
        "-s",
        "--stopping_tol",
        type=float,
        default=1e-5,
        help="Stopping tolerance between two successive PG iterations (L2-difference)",
    )
    result_options = parser.add_argument_group("Output options")
    result_options.add_argument("--write_frequency", type=int, default=25, help="Write frequency")
    result_options.add_argument(
        "--result_dir", type=Path, default=Path("results"), help="Directory to store results"
    )
    parsed_args = parser.parse_args(argv)
    if parsed_args.logging:
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
    else:
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)
    newton_its, lvpp_its = solve_problem(
        N=parsed_args.N,
        M=parsed_args.M,
        primal_degree=parsed_args.primal_degree,
        cell_type=parsed_args.cell_type,
        alpha_max=parsed_args.alpha_max,
        alpha_scheme=parsed_args.alpha_scheme,
        alpha_0=parsed_args.alpha_0,
        alpha_c=parsed_args.alpha_c,
        max_iterations=parsed_args.max_iterations,
        result_dir=parsed_args.result_dir,
        write_frequency=parsed_args.write_frequency,
        stopping_tol=parsed_args.stopping_tol,
        tau0=parsed_args.tau0,
        T=parsed_args.T,
    )
    newton_its = 0
    lvpp_its = 1
    if MPI.COMM_WORLD.rank == 0:
        Path(parsed_args.result_dir).mkdir(parents=True, exist_ok=True)
        np.savez(
            parsed_args.result_dir / "iteration_count.npz",
            newton_its=newton_its,
            lvpp_its=lvpp_its,
            N=parsed_args.N,
            M=parsed_args.M,
            primal_degree=parsed_args.primal_degree,
            cell_type=parsed_args.cell_type,
            a_scheme=parsed_args.alpha_scheme,
            alpha_0=parsed_args.alpha_0,
            alpha_c=parsed_args.alpha_c,
            alpha_max=parsed_args.alpha_max,
            max_iterations=parsed_args.max_iterations,
            stopping_tol=parsed_args.stopping_tol,
        )


if __name__ == "__main__":
    main()
