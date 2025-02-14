"""
Solving the obstacle problem using Galahad or IPOPT with DOLFINx generating the system matrices
Author: Jørgen S. Dokken
SPDX-License-Identifier: MIT
"""

import argparse
from pathlib import Path

from mpi4py import MPI

import dolfinx
import numpy as np
import scipy.sparse
import ufl

from lvpp.optimization import galahad_solver, ipopt_solver

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
parser.add_argument("--max-iter", type=int, default=50, help="Maximum number of iterations")
parser.add_argument("--O", type=Path, dest="outdir", default="results", help="Output directory")


def setup_problem(N: int, cell_type: dolfinx.mesh.CellType = dolfinx.mesh.CellType.triangle):
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

    lower_bound = dolfinx.fem.Function(Vh, name="lower_bound")
    upper_bound = dolfinx.fem.Function(Vh, name="upper_bound")
    lower_bound.interpolate(psi)
    upper_bound.x.array[:] = np.inf
    # Deactivate boundary dofs
    dolfinx.fem.set_bc(upper_bound.x.array, bcs)
    dolfinx.fem.set_bc(lower_bound.x.array, bcs)

    x = ufl.SpatialCoordinate(mesh)
    v = ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])
    f_expr = dolfinx.fem.Expression(ufl.div(ufl.grad(v)), Vh.element.interpolation_points())
    f = dolfinx.fem.Function(Vh)
    f.interpolate(f_expr)

    S = dolfinx.fem.assemble_matrix(dolfinx.fem.form(stiffness))
    M = dolfinx.fem.assemble_matrix(dolfinx.fem.form(mass))

    return S.to_scipy(), M.to_scipy(), f, (lower_bound, upper_bound)


class ObstacleProblem:
    total_iteration_count: int

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

    def intermediate(self, *args):
        """Ipopt callback function"""
        self.total_iteration_count = args[1]


if __name__ == "__main__":
    args = parser.parse_args()

    S_, M_, f_, bounds_ = setup_problem(args.N)
    V = f_.function_space
    bounds = tuple(b.x.array for b in bounds_)

    # Restrict all matrices and vectors to interior dofs
    problem = ObstacleProblem(S_.copy(), M_.copy(), f_.x.array)

    if args.galahad:
        x_g = dolfinx.fem.Function(V, name="galahad")
        x_g.x.array[:] = 0.0
        init_galahad = x_g.x.array.copy()
        x_galahad, iterations = galahad_solver(
            problem,
            init_galahad,
            bounds,
            max_iter=args.max_iter,
            use_hessian=True,
        )
        x_g.x.array[:] = x_galahad
        outdir = args.outdir
        with dolfinx.io.VTXWriter(V.mesh.comm, outdir / "galahad.bp", [x_g]) as bp:
            bp.write(0.0)
        print(f"Galahad iterations: {iterations}")
    if args.ipopt:
        x_i = dolfinx.fem.Function(V, name="ipopt")
        x_i.x.array[:] = 0.0
        init_ipopt = x_i.x.array.copy()
        x_ipopt = ipopt_solver(problem, init_ipopt, bounds, max_iter=args.max_iter)

        x_i.x.array[:] = x_ipopt

        with dolfinx.io.VTXWriter(V.mesh.comm, outdir / "ipopt.bp", [x_i]) as bp:
            bp.write(0.0)
        print(f"Ipopt iterations: {problem.total_iteration_count}")
