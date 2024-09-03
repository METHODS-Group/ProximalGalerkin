"""
Solving the obstacle problem using Galahad with DOLFINx generating the system matrices
Author: Jørgen S. Dokken
SPDX-License-Identifier: MIT
"""


from mpi4py import MPI
import dolfinx
import ufl
import argparse
import numpy as np
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


def setup_problem(N: int, cell_type: dolfinx.mesh.CellType = dolfinx.mesh.CellType.triangle):
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type)

    Vh = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(Vh)
    v = ufl.TestFunction(Vh)

    mass = ufl.inner(u, v) * ufl.dx
    stiffness = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim-1, tdim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(Vh, tdim-1, boundary_facets)

    # Get dofs to deactivate
    bcs = [dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0.), boundary_dofs, Vh)]
    
    def psi(x):
        return -1./4 + 1./10 * np.sin(np.pi*x[0])*np.sin(np.pi*x[1])

    lower_bound = dolfinx.fem.Function(Vh)
    upper_bound = dolfinx.fem.Function(Vh)
    lower_bound.interpolate(psi)
    upper_bound.x.array[:] = np.inf

    x = ufl.SpatialCoordinate(mesh)
    v = ufl.sin(3*ufl.pi*x[0])*ufl.sin(3*ufl.pi*x[1])
    f_expr = dolfinx.fem.Expression(ufl.div(ufl.grad(v)), Vh.element.interpolation_points())
    f = dolfinx.fem.Function(Vh)
    f.interpolate(f_expr)

    S = dolfinx.fem.assemble_matrix(dolfinx.fem.form(stiffness))
    M = dolfinx.fem.assemble_matrix(dolfinx.fem.form(mass))

    return S.to_scipy(), M.to_scipy(), Vh, f, (lower_bound, upper_bound), bcs



if __name__ == "__main__":
    args = parser.parse_args()
    
    S, M, V, f, bounds, bcs = setup_problem(args.N)
    dof_indices = np.unique(np.hstack([bc._cpp_object.dof_indices()[0] for bc in bcs]))
    keep = np.full(len(f.x.array), True, dtype=np.bool_)
    keep[dof_indices] = False
    keep_indices = np.flatnonzero(keep)

    S_d = S[keep_indices].tocsc()[:, keep_indices].tocsr()
    M_d = M[keep_indices].tocsc()[:, keep_indices].tocsr()
    f_d = f.x.array[keep_indices]

    Mf = (M_d @ f_d)
    S_flattened = S_d.todense().reshape(-1)

    def J(x):
        return 0.5 * x.T @ (S_d @ x) - f_d.T @ (M_d @ x)        

    def G(x):
        return S_d @ x - Mf

    def H(x):
        return S_flattened




    from galahad import trb
    np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
    options = trb.initialize()

    # set some non-default options
    options['print_level'] = 3
    options['model'] = 2
    options['maxit'] = 1000
    options['hessian_available'] = True
    n = len(keep_indices)
    H_type="dense"
    H_ne = 0
    H_row = np.zeros(H_ne, dtype=np.int64)
    H_col = np.zeros(H_ne, dtype=np.int64)
    H_ptr = None
    x_l = bounds[0].x.array[keep_indices]
    x_u = bounds[1].x.array[keep_indices]
    # Add Dirichlet bounds 0 here
    trb.load(n, x_l, x_u, H_type, H_ne, H_row, H_col, H_ptr=None, options=options)


    x = dolfinx.fem.Function(V)
    dolfinx.fem.set_bc(x.x.array, bcs)
    x_d = x.x.array[keep_indices]
    trb.solve(n, H_ne, x_d, J, G, H)
    x.x.array[keep_indices] = x_d

    with dolfinx.io.VTXWriter(V.mesh.comm, "galahad.bp", [x]) as bp:
        bp.write(0.0)