"""
Solving the obstacle problem using Galahad or IPOPT with DOLFINx generating the system matrices
Author: Jørgen S. Dokken
SPDX-License-Identifier: MIT
"""

import argparse
from pathlib import Path

from mpi4py import MPI

from lvpp import galahad_solver, ipopt_solver
import dolfinx
import numpy as np
import scipy.sparse
import ufl

parser = argparse.ArgumentParser(
    description="""Solve the obstacle problem on a general mesh using a spatially varying
      phi using Galahad or IPOPT""",
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
parser.add_argument("--max-iter", type=int, default=200, help="Maximum number of iterations")
parser.add_argument("--tol", type=float, default=1e-6, help="Convergence tolerance")
parser.add_argument(
    "--hessian", dest="use_hessian", action="store_true", default=False, help="Use exact hessian"
)
parser.add_argument(
    "--output", "-o", dest="outdir", type=Path, default=Path("results"), help="Output directory"
)


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

    lower_bound = dolfinx.fem.Function(Vh, name="lower_bound")
    upper_bound = dolfinx.fem.Function(Vh, name="upper_bound")
    lower_bound.interpolate(psi)
    upper_bound.x.array[:] = np.inf
    dolfinx.fem.set_bc(upper_bound.x.array, bcs)
    dolfinx.fem.set_bc(lower_bound.x.array, bcs)

    f = dolfinx.fem.Function(Vh)
    f.x.array[:] = 0.0
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
        self._sparsity = tri_S.nonzero()
        self._H_data = tri_S.data

    def objective(self, x):
        """Returns the scalar value of the objective given x."""
        return 0.5 * x.T @ (self._S @ x) - self._f.T @ (self._M @ x)

    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""

        return self._S @ x - self._Mf

    def pure_hessian(self, x):
        return self._H_data

    def hessian(self, x, lagrange, obj_factor):
        return obj_factor * self.pure_hessian(x)

    def hessianstructure(self):
        return self._sparsity

    def intermediate(self, *args):
        """Ipopt callback function"""
        self.total_iteration_count = args[1]


if __name__ == "__main__":
    args = parser.parse_args()

    S_, M_, f_, bounds_ = setup_problem(args.infile)
    V = f_.function_space
    bounds = tuple(b.x.array for b in bounds_)
    # Restrict all matrices and vectors to interior dofs
    problem = ObstacleProblem(S_.copy(), M_.copy(), f_.x.array)
    outdir = args.outdir
    if args.galahad:
        x_g = dolfinx.fem.Function(V, name="galahad")
        x_g.x.array[:] = 0.0
        init_galahad = x_g.x.array.copy()
        x_galahad, iterations = galahad_solver(
            problem,
            init_galahad,
            bounds,
            max_iter=args.max_iter,
            use_hessian=args.use_hessian,
            tol=args.tol,
        )
        x_g.x.array[:] = x_galahad
        with dolfinx.io.VTXWriter(V.mesh.comm, outdir / "galahad_obstacle.bp", [x_g]) as bp:
            bp.write(0.0)

    if args.ipopt:
        x_i = dolfinx.fem.Function(V, name="ipopt")
        x_i.x.array[:] = 0.0
        init_ipopt = x_i.x.array.copy()
        x_ipopt = ipopt_solver(
            problem,
            init_ipopt,
            bounds,
            max_iter=args.max_iter,
            tol=args.tol,
            activate_hessian=args.use_hessian,
        )

        x_i.x.array[:] = x_ipopt

        # Output on geometry space
        mesh = x_i.function_space.mesh
        degree = mesh.geometry.cmap.degree
        V_out = dolfinx.fem.functionspace(mesh, ("Lagrange", degree))
        x_i_out = dolfinx.fem.Function(V_out, name="ipopt")
        x_i_out.interpolate(x_i)
        with dolfinx.io.VTXWriter(mesh.comm, outdir / "ipopt_obstacle.bp", [x_i_out]) as bp:
            bp.write(0.0)
