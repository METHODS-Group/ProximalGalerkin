"""
Solving the obstacle problem using Galahad or IPOPT with DOLFINx generating the system matrices
Author: Jørgen S. Dokken
SPDX-License-Identifier: MIT
"""

import cyipopt

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
    "--N",
    "-N",
    dest="N",
    type=int,
    default=10,
    help="Number of elements in each direction",
)
parser.add_argument("--ipopt", action="store_true", default=False, help="Use Ipopt")
parser.add_argument("--galahad", action="store_true", default=False, help="Use Galahad")
parser.add_argument(
    "--max-iter", type=int, default=50, help="Maximum number of iterations"
)


def setup_problem(
    N: int, cell_type: dolfinx.mesh.CellType = dolfinx.mesh.CellType.triangle
):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type)

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
        return -1.0 / 4 + 1.0 / 10 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])

    lower_bound = dolfinx.fem.Function(Vh)
    upper_bound = dolfinx.fem.Function(Vh)
    lower_bound.interpolate(psi)
    upper_bound.x.array[:] = np.inf

    x = ufl.SpatialCoordinate(mesh)
    v = ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])
    f_expr = dolfinx.fem.Expression(
        ufl.div(ufl.grad(v)), Vh.element.interpolation_points()
    )
    f = dolfinx.fem.Function(Vh)
    f.interpolate(f_expr)

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
    problem, x, bounds, log_level: int = 5, max_iter: int = 100, tol: float = 1e-6
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
        "hessian_approximation": "exact",
        "jacobian_approximation": "exact",
        "hessian_constant": "yes",
    }

    nlp = cyipopt.Problem(
        n=len(x), m=0, lb=bounds[0], ub=bounds[1], problem_obj=problem
    )
    for key, val in options.items():
        nlp.add_option(key, val)

    x_opt, _ = nlp.solve(x)
    return x_opt


if __name__ == "__main__":
    args = parser.parse_args()

    S_, M_, V, f_, bounds, bcs = setup_problem(args.N)
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
            use_hessian=True,
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
            problem, init_ipopt, (lower_bound, upper_bound), max_iter=args.max_iter
        )

        x_i.x.array[keep_indices] = x_ipopt
        dolfinx.fem.set_bc(x_i.x.array, bcs)

        with dolfinx.io.VTXWriter(V.mesh.comm, "ipopt.bp", [x_i]) as bp:
            bp.write(0.0)
