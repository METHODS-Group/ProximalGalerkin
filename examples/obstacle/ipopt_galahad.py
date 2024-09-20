"""
Solving the obstacle problem using Galahad or IPOPT with DOLFINx generating the system matrices
Author: Jørgen S. Dokken
SPDX-License-Identifier: MIT
"""

import cyipopt
from pathlib import Path
from mpi4py import MPI
import dolfinx
import ufl
import argparse
import numpy as np
import scipy.sparse

parser = argparse.ArgumentParser(
    description="Solve the obstacle problem on a unit square using Galahad.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--path",
    "-P",
    dest="infile",
    type=Path,
    required=True,
    help="Path to infile",
)
parser.add_argument("--ipopt", action="store_true", default=False, help="Use Ipopt")
parser.add_argument("--galahad", action="store_true", default=False, help="Use Galahad")
parser.add_argument(
    "--max-iter", type=int, default=200, help="Maximum number of iterations"
)
parser.add_argument("--tol", type=float, default=1e-6, help="Convergence tolerance")
parser.add_argument("--hessian", dest="use_hessian", action="store_true", default=False, help="Use exact hessian")

def setup_problem(
     filename: Path,
):
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
        mesh = xdmf.read_mesh(name="mesh")

    Vh = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(Vh)
    v = ufl.TestFunction(Vh)

    mass = ufl.inner(u, v) * ufl.dx
    stiffness = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(Vh, tdim - 1, boundary_facets)

    # Get dofs to deactivate
    bcs = [dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0.0), boundary_dofs, Vh)]


    def psi(x):
        r = np.sqrt(x[0] ** 2 + x[1] ** 2)
        r0 = 0.5
        beta =  0.9
        b = r0*beta
        tmp = np.sqrt(r0**2 - b**2)
        B = tmp + b * b / tmp
        C = -b / tmp
        cond_true = B + r * C
        cond_false = np.sqrt(r0 ** 2 - r ** 2)
        true_indices = np.flatnonzero(r > b)
        cond_false[true_indices] = cond_true[true_indices]
        return cond_false

    lower_bound = dolfinx.fem.Function(Vh)
    upper_bound = dolfinx.fem.Function(Vh)
    lower_bound.interpolate(psi)
    upper_bound.x.array[:] = np.inf

    f = dolfinx.fem.Function(Vh)
    f.x.array[:] = 0.0
    S = dolfinx.fem.assemble_matrix(dolfinx.fem.form(stiffness))
    M = dolfinx.fem.assemble_matrix(dolfinx.fem.form(mass))

    return S.to_scipy(), M.to_scipy(), Vh, f, (lower_bound, upper_bound), bcs


def galahad(
    problem,
    x,
    bounds,
    log_level: int = 1,
    use_hessian: bool = True,
    max_iter: int = 100,
    tol: float = 1e-6,
):
    """
    :param problem: Problem instance
    :param x: Initial condition
    :param bounds: (lower_bound, upper_bound)_
    :param loglevel: Verbosity level for galahad (0-3)
    :param use_hessian: If True use second order method, otherwise use first order
    :param max_iter: Maximum number of iterations
    :param tol: Relative convergence tolerance
    :return: Optimized solution
    """

    from galahad import trb

    options = trb.initialize()

    # set some non-default options
    options["print_level"] = log_level
    options["model"] = 2 if use_hessian else 1
    options["maxit"] = max_iter
    options["hessian_available"] = True
    options["stop_pg_relative"] = tol
    options["subproblem_direct"] = True
    n = len(x)
    H_type = "coordinate"
    H_ne = len(problem.sparsity[0])
    H_ptr = None
    # Add Dirichlet bounds 0 here
    trb.load(
        n,
        bounds[0].copy(),
        bounds[1].copy(),
        H_type,
        H_ne,
        problem.sparsity[0].astype(np.int64),
        problem.sparsity[1].astype(np.int64),
        H_ptr=H_ptr,
        options=options,
    )
    x_out, _ = trb.solve(
        n, H_ne, x, problem.objective, problem.gradient, problem.pure_hessian
    )
    return x_out


class ObstacleProblem:
    def __init__(self, S, M, f):
        S.eliminate_zeros()
        self._S = S
        self._M = M
        self._Mf = M @ f
        self._f = f
        tri_S = scipy.sparse.tril(self._S)
        self.sparsity = tri_S.nonzero()
        self._H_data = np.copy(tri_S.todense())[self.sparsity[0], self.sparsity[1]]

    def objective(self, x):
        """Returns the scalar value of the objective given x."""
        return 0.5 * x.T @ (self._S @ x) - self._f.T @ (self._M @ x)

    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""

        return self._S @ x - self._Mf

    def pure_hessian(self, x):
        return self._H_data

    def hessian(self, x, lagrange, obj_factor):
        return self.pure_hessian(x)

    def hessianstructure(self):
        return self.sparsity
    


def ipopt(
    problem, x, bounds, log_level: int = 5, max_iter: int = 100, tol: float = 1e-6,
    activate_hessian:bool=True
):
    """
    :param problem: Problem instance
    :param x: Initial condition
    :param log_level: Veribosity level for Ipopt
    :param bounds: (lower_bound, upper_bound)
    :param tol: Relative convergence tolerance
    :return: Optimized solution
    """
    options = {
        "print_level": log_level,
        "max_iter": max_iter,
        "tol": tol,
        "jacobian_approximation": "exact",
    }

    if activate_hessian:
        options["hessian_approximation"] = "exact"
        options["hessian_constant"] = "yes"
    else:
        options["hessian_approximation"] = "limited-memory"

    nlp = cyipopt.Problem(
        n=len(x), m=0, lb=bounds[0], ub=bounds[1], problem_obj=problem
    )
    for key, val in options.items():
        nlp.add_option(key, val)

    x_opt, _ = nlp.solve(x)
    return x_opt


if __name__ == "__main__":
    args = parser.parse_args()

    S_, M_, V, f_, bounds, bcs = setup_problem(args.infile)
    dof_indices = np.unique(np.hstack([bc._cpp_object.dof_indices()[0] for bc in bcs]))
    keep = np.full(len(f_.x.array), True, dtype=np.bool_)
    keep[dof_indices] = False
    keep_indices = np.flatnonzero(keep)

    # Restrict all matrices and vectors to interior dofs
    S_d = S_[keep_indices].tocsc()[:, keep_indices].tocsr()
    M_d = M_[keep_indices].tocsc()[:, keep_indices].tocsr()
    f_d = f_.x.array[keep_indices]

    problem = ObstacleProblem(S_d, M_d, f_d)

    lower_bound = bounds[0].x.array[keep_indices]
    upper_bound = bounds[1].x.array[keep_indices]

    if args.galahad:
        x_g = dolfinx.fem.Function(V, name="galahad")
        x_g.x.array[:] = 0.0
        init_galahad = x_g.x.array[keep_indices].copy()
        x_galahad = galahad(
            problem,
            init_galahad,
            (lower_bound, upper_bound),
            max_iter=args.max_iter,
            use_hessian=args.use_hessian,
            tol=args.tol
        )
        x_g.x.array[keep_indices] = x_galahad
        dolfinx.fem.set_bc(x_g.x.array, bcs)

        with dolfinx.io.VTXWriter(V.mesh.comm, "galahad.bp", [x_g]) as bp:
            bp.write(0.0)

    if args.ipopt:
        x_i = dolfinx.fem.Function(V, name="ipopt")
        x_i.x.array[:] = 0.0
        init_ipopt = x_i.x.array[keep_indices].copy()
        x_ipopt = ipopt(
            problem, init_ipopt, (lower_bound, upper_bound), max_iter=args.max_iter,
            tol=args.tol, activate_hessian=args.use_hessian
        )

        x_i.x.array[keep_indices] = x_ipopt
        dolfinx.fem.set_bc(x_i.x.array, bcs)

        # Output on geometry space
        mesh = x_i.function_space.mesh
        degree = mesh.geometry.cmap.degree
        V_out = dolfinx.fem.functionspace(mesh, ("Lagrange", degree))
        x_i_out = dolfinx.fem.Function(V_out, name="ipopt")
        x_i_out.interpolate(x_i)
        with dolfinx.io.VTXWriter(mesh.comm, "ipopt.bp", [x_i_out]) as bp:
            bp.write(0.0)
