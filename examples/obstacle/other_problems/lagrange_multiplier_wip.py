from mpi4py import MPI
import dolfinx
import ufl
import numpy as np
import scipy.sparse

N = 50
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N, dolfinx.cpp.mesh.CellType.triangle)

Vh = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

u = ufl.TrialFunction(Vh)
v = ufl.TestFunction(Vh)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
a_compiled = dolfinx.fem.form(a)
A = dolfinx.fem.assemble_matrix(a_compiled)
Ah = A.to_scipy()
B = scipy.sparse.identity(Ah.shape[0])

x = ufl.SpatialCoordinate(mesh)
v_ = ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])
f_expr = dolfinx.fem.Expression(
    ufl.div(ufl.grad(v_)), Vh.element.interpolation_points()
)
f_ = dolfinx.fem.Function(Vh)
f_.interpolate(f_expr)
f = dolfinx.fem.assemble_vector(dolfinx.fem.form(ufl.inner(f_,v)*ufl.dx))

def psi(x):
    return -1.0 / 4 + 1.0 / 10 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])

g = dolfinx.fem.Function(Vh, name="g")
g.interpolate(psi)


def estimate_active_set(c, lmbda, u):
    constraint = lmbda.x.array + c * (g.x.array - B @ u.x.array)
    return np.flatnonzero(constraint>=0)


lmbda = dolfinx.fem.Function(Vh, name="lambda")

uh = dolfinx.fem.Function(Vh, name="uh")
uh.interpolate(lambda x: -0.17 + 0.05 * np.sin(4*np.pi*x[0]))
active_set = estimate_active_set(1, lmbda, uh)

ind = dolfinx.fem.Function(Vh, name="active_set")
ind.x.array[active_set] = 1

with dolfinx.io.VTXWriter(mesh.comm, "test.bp", [uh, ind, g]) as bp:
    bp.write(0.0)


# Where active set solve:
# Bu = Iu = u = g
# lmbda = f - Ag
# else
# lmbda = 0
# Au = f

lmbda_h = dolfinx.fem.Function(Vh)
lmbda_h.x.array[active_set] = f.x.array[active_set] - Ah @



# (A  -I) u = (f,v)
# (0   I) lmbda = 0  
#
# (A  -I) u = (f, v)
# (B   0) lmbda = g