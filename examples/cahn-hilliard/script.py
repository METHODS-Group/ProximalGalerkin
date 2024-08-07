from pathlib import Path
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
import basix.ufl
import ufl
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem.petsc import NonlinearProblem
from typing import Optional
import argparse
from enum import Enum


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


def solve_problem(N: int, M: int,
                  num_species: int,
                  primal_degree: int,
                  cell_type: str,
                  alpha_scheme: AlphaScheme,
                  alpha_0: float,
                  alpha_c: float,
                  max_iterations: int,
                  stopping_tol: float,
                  result_dir: Path):
    _cell_type = dolfinx.mesh.to_type(cell_type)

    mesh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, N, M, cell_type=_cell_type,
        diagonal=dolfinx.mesh.DiagonalType.crossed)

    el_0 = basix.ufl.element(
       "P", mesh.topology.cell_name(), primal_degree, shape=(num_species,))
    el_1 = basix.ufl.element(
        "P", mesh.topology.cell_name(), primal_degree, shape=(num_species,))

    # Trial space is (c, mu, psi)
    trial_el = basix.ufl.mixed_element([el_0, el_0, el_1])
    V_trial = dolfinx.fem.functionspace(mesh, trial_el)
    V_test = V_trial

    sol = dolfinx.fem.Function(V_trial)
    c, mu, psi = ufl.split(sol)

    v, nu, w = ufl.TestFunctions(V_test)

    dx = ufl.Measure("dx",  domain=mesh)
    alpha = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(alpha_0))

    w_prev = dolfinx.fem.Function(V_trial)
    c_prev, _, psi_prev = ufl.split(w_prev)
    dt = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.01))
    m_ij = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(-1))
    m_ii = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(2))
    M = ufl.as_tensor([[m_ii if i == j else m_ij for i in range(num_species)] for j in range(num_species)])
    M = ufl.Identity(num_species)
    u = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type((0.0, 0.0)))
    i, j, k = ufl.indices(3)
    F_0 = (c[i] - c_prev[i])/dt * v[i] * dx 
    F_0 += ufl.dot(u, ufl.grad(c[i])) * v[i] * dx
    F_0 +=  M[i,j] * ufl.grad(mu[j])[k] * ufl.grad(v[i])[k] * dx
    
    h = ufl.Circumradius(mesh)
    epsilon = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1))#ufl.sqrt(h) 
    sigma = ufl.Identity(num_species)
    ones = dolfinx.fem.Constant(
        mesh, dolfinx.default_scalar_type([1,]*num_species))
    F_1 = mu[i] * nu[i] * dx
    F_1 -= epsilon**2 * sigma[i,j] * ufl.grad(c[j])[k]*ufl.grad(nu[i])[k] * dx
    F_1 -= ((0.5*ones[i] - c[i])) * nu[i] *dx
    F_1 -= 1/alpha * (psi[i] - psi_prev[i]) * nu[i] * dx

    sum_psi = sum(ufl.exp(psi[m]) for m in range(num_species))
    F_2 = sum((c[m] - ufl.exp(psi[m]) / sum_psi) * w[m]
             for m in range(num_species))*dx

    # Random values between 0 and 1 that sum to 1
    num_dofs = len(w_prev.sub(0).sub(0).collapse().x.array[:])
    rands = np.random.rand(num_dofs, num_species)
    norm_rand = np.linalg.norm(rands,axis=1)
    c_0, c_to_V = V_trial.sub(0).collapse()
    w_prev.x.array[c_to_V] = (rands / norm_rand.reshape(-1, 1)).reshape(-1)
    sol.sub(0).interpolate(w_prev.sub(0))
    
    bcs = []
    F = F_0 + F_1 + F_2
    problem = NonlinearProblem(F, sol, bcs=bcs)
    solver = NewtonSolver(mesh.comm, problem)
    solver.convergence_criterion = "residual"
    solver.rtol = 1e-8
    solver.atol = 1e-8
    solver.max_it = 1
    solver.error_on_nonconvergence = False

    ksp = solver.krylov_solver
    opts = PETSc.Options()  # type: ignore
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "preonly"
    opts[f"{option_prefix}pc_type"] = "lu"
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
    ksp.setFromOptions()

    dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

    c_out = dolfinx.fem.Function(V_trial.sub(0).collapse()[0])
    bp_c = dolfinx.io.VTXWriter(
        mesh.comm, result_dir / "c.bp", [c_out], engine="BP4")

    c_out.interpolate(w_prev.sub(0).collapse())
    bp_c.write(0)

    prox_prev = dolfinx.fem.Function(V_trial)
    diff = sol.sub(0)-prox_prev.sub(0)

    L2_squared = ufl.dot(diff, diff)*dx
    compiled_diff = dolfinx.fem.form(L2_squared)
    
    T = 1
    num_steps = int(T/float(dt))
    for j in range(num_steps):
        for i in range(1, max_iterations+1):
            newton_iterations = np.zeros(max_iterations, dtype=np.int32)
            L2_diff = np.zeros(max_iterations, dtype=np.float64)

            if alpha_scheme == AlphaScheme.constant:
                pass
            elif alpha_scheme == AlphaScheme.linear:
                alpha.value = alpha_0 + alpha_c * i
            elif alpha_scheme == AlphaScheme.doubling:
                alpha.value = alpha_0 * 2**i
            # solver.rtol *= 0.5
            # solver.atol *= 0.5

            num_newton_iterations, converged = solver.solve(sol)
            newton_iterations[i] = num_newton_iterations
            local_diff = dolfinx.fem.assemble_scalar(compiled_diff)
            global_diff = np.sqrt(mesh.comm.allreduce(local_diff, op=MPI.SUM))
            L2_diff[i-1] = global_diff
            if mesh.comm.rank == 0:
                print(
                    f"Iteration {i}: {converged=} {num_newton_iterations=} {ksp.getConvergedReason()=}",
                    f"|delta u |= {global_diff}")
            # Update solution
            prox_prev.x.array[:] = sol.x.array[:]
            print(sol.x.array)
            c_out.interpolate(sol.sub(0).collapse())
            bp_c.write(i)
            # bp_grad_u.write(i)
            if global_diff < stopping_tol:
                break

        break
        
    bp_c.close()
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
    mesh_options.add_argument("--cell_type", "-c", type=str, default="triangle",
                              choices=["triangle", "quadrilateral"], help="Cell type")
    element_options = parser.add_argument_group(
        "Finite element discretization options")
    element_options.add_argument("--primal_degree", type=int, default=2, choices=[
                                 2, 3, 4, 5, 6, 7, 8], help="Polynomial degree for primal variable")
    alpha_options = parser.add_argument_group(
        "Options for alpha-variable in Proximal Galerkin scheme")
    alpha_options.add_argument("--alpha_scheme", type=str, default="linear", choices=[
                               "constant", "linear", "doubling"], help="Scheme for updating alpha")
    alpha_options.add_argument(
        "--alpha_0", type=float, default=1.0, help="Initial value of alpha")
    alpha_options.add_argument(
        "--alpha_c", type=float, default=1.0, help="Increment of alpha in linear scheme")
    pg_options = parser.add_argument_group("Proximal Galerkin options")
    pg_options.add_argument("--max_iterations", type=int,
                            default=20, help="Maximum number of iterations")
    pg_options.add_argument("-s", "--stopping_tol", type=float, default=1e-9,
                            help="Stopping tolerance between two successive PG iterations (L2-difference)")
    result_options = parser.add_argument_group("Output options")
    result_options.add_argument(
        "--result_dir", type=Path, default=Path("results"), help="Directory to store results")
    parsed_args = parser.parse_args(argv)

    iteration_counts, L2_diffs = solve_problem(N=parsed_args.N, M=parsed_args.M,
                                               num_species=parsed_args.num_species,
                                               primal_degree=parsed_args.primal_degree,
                                               cell_type=parsed_args.cell_type,
                                               alpha_scheme=AlphaScheme.from_string(
                                                   parsed_args.alpha_scheme),
                                               alpha_0=parsed_args.alpha_0,
                                               alpha_c=parsed_args.alpha_c,
                                               max_iterations=parsed_args.max_iterations,
                                               result_dir=parsed_args.result_dir,
                                               stopping_tol=parsed_args.stopping_tol)
    print(iteration_counts)
    print(L2_diffs)


if __name__ == "__main__":
    main()
