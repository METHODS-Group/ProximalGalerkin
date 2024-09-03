from mpi4py import MPI
from petsc4py import PETSc
import dolfinx.fem.petsc
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
    bcs = [dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0.), boundary_dofs, Vh)]

    def psi(x):
        return 1./4 - 1./10 * np.sin(np.pi*x[0])*np.sin(np.pi*x[1])

    lower_bound = dolfinx.fem.Function(Vh)
    upper_bound = dolfinx.fem.Function(Vh)
    lower_bound.x.petsc_vec.set(-PETSc.INFINITY)
    upper_bound.interpolate(psi)




    x = ufl.SpatialCoordinate(mesh)
    v = ufl.sin(3*ufl.pi*x[0])*ufl.sin(3*ufl.pi*x[1])
    f_expr = dolfinx.fem.Expression(-ufl.div(ufl.grad(v)), Vh.element.interpolation_points())
    f = dolfinx.fem.Function(Vh)
    f.interpolate(f_expr)

    S = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(stiffness))
    M = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(mass))
    S.assemble()
    M.assemble()
    return S, M, Vh, f, bcs, (lower_bound, upper_bound)


def create_structures(S, M, V, f, bcs):
    x_vec = dolfinx.fem.Function(V)
    work_vec = dolfinx.fem.Function(V)
    def J(x):
        # Set input values and set bc dofs
        x_vec.x.petsc_vec.array_w[:] = x
        dolfinx.fem.petsc.set_bc(x_vec.x.petsc_vec, bcs)
        
        # Compute 0.5 x^T S x
        work_vec.x.petsc_vec.set(0)
        S.mult(x_vec.x.petsc_vec, work_vec.x.petsc_vec)        
        xTSx = work_vec.x.petsc_vec.dot(x_vec.x.petsc_vec)

        # Compute I(f)^T M x
        work_vec.x.petsc_vec.set(0)
        M.mult(x_vec.x.petsc_vec, work_vec.x.petsc_vec)        
        IfTMx = work_vec.x.petsc_vec.dot(f.x.petsc_vec)
        return 0.5 * xTSx - IfTMx

    x_vec_G = dolfinx.fem.Function(V)
    work_vec_G = dolfinx.fem.Function(V)
    Mf = dolfinx.fem.Function(V)
    Mf.x.array[:] = 0.0
    M.mult(f.x.petsc_vec, Mf.x.petsc_vec)
    def G(x):
        x_vec_G.x.petsc_vec.array_w[:] = x
        work_vec_G.x.petsc_vec.set(0)
        S.mult(x_vec_G.x.petsc_vec, work_vec_G.x.petsc_vec)
        work_vec_G.x.petsc_vec.axpy(-1, Mf.x.petsc_vec)
        return work_vec_G.x.array

    return J, G

if __name__ == "__main__":
    args = parser.parse_args()
    
    S, M, V, f, bcs, bounds = setup_problem(args.N)

    Jh, Gh = create_structures(S, M, V,f, bcs)
    input = dolfinx.fem.Function(V)
    input.x.array[:] = 10


    from galahad import trb
    np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
    options = trb.initialize()

    # set some non-default options
    options['print_level'] = 2
    options['model'] = 1

    n = len(f.x.array)
    H_type="dense"
    H_ne = 0
    H_row = np.zeros(H_ne, dtype=np.int64)
    H_col = np.zeros(H_ne, dtype=np.int64)
    H_ptr = None
    x_l = bounds[0].x.array
    x_u = bounds[1].x.array

    breakpoint()
    trb.load(n, x_l, x_u, H_type, H_ne, H_row, H_col, H_ptr=None, options=options)


    x = dolfinx.fem.Function(V)
    x.x.array[:] = 0
    trb.solve(n, H_ne, x.x.array, Jh, Gh, lambda x: np.zeros((len(x), len(x)), dtype=x.dtype))
        
    # from galahad import trb
    # import numpy as np
    # np.set_printoptions(precision=4,suppress=True,floatmode='fixed')
    # print("\n** python test: trb")

    # # allocate internal data and set default options
    # options = trb.initialize()

    # # set some non-default options
    # options['print_level'] = 0
    # #options['trs_options']['print_level'] = 0
    # #print("options:", options)

    # # set parameters
    # p = 4
    # # set bounds
    # n = 3
    # x_l = np.array([-np.inf,-np.inf,0.0])
    # x_u = np.array([1.1,1.1,1.1])

    # # set Hessian sparsity
    # H_type = 'coordinate'
    # H_ne = 5
    # H_row = np.array([0,2,1,2,2])
    # H_col = np.array([0,0,1,1,2])
    # H_ptr = None

    # load data (and optionally non-default options)

    # # define objective function and its derivatives
    # def eval_f(x):
    #     return (x[0] + x[2] + p)**2 + (x[1] + x[2])**2 + np.cos(x[0])
    # def eval_g(x):
    #     return np.array([2.0* ( x[0] + x[2] + p ) - np.sin(x[0]),
    #                     2.0* ( x[1] + x[2] ),
    #                     2.0* ( x[0] + x[2] + p ) + 2.0 * ( x[1] + x[2] )])
    # def eval_h(x):
    #     return np.array([2. - np.cos(x[0]),2.0,2.0,2.0,4.0])

    # # set starting point
    # x = np.array([1.,1.,1.])

    # # find optimum
    # breakpoint()
    # x, g = trb.solve(n, H_ne, x, eval_f, eval_g, eval_h)
    # print(" x:",x)
    # print(" g:",g)

    # # get information