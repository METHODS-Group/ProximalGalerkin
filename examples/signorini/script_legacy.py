from dolfin import *
mesh = UnitSquareMesh(10, 10)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 1.0)
    
class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0)

ff = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
Top().mark(ff, 1)
Bottom().mark(ff, 2)


submesh = MeshView.create(ff, 2)

with XDMFFile("output/ff.xdmf") as xdmf:
    xdmf.write(ff)

E, nu = 2.0e3, 0.3
mu = Constant(E / (2.0 * (1.0 + nu)))
lmbda = Constant(E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))

def epsilon(w):
    return sym(grad(w))

def sigma(w, gdim):
    return 2.0 * mu * epsilon(w) + lmbda * tr(grad(w)) * Identity(gdim)


def approximate_facet_normal(V):
        # define the domains for integration
    mesh = V.mesh()
    ds = Measure("ds", domain=mesh)

    n = FacetNormal(mesh)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(u, v)*ds 
    l = inner(n, v)*ds

    A = assemble(a, keep_diagonal=True)
    L = assemble(l)

    A.ident_zeros()
    nh = Function(V)
    solve(A, nh.vector(), L)
    return nh

A = VectorFunctionSpace(mesh, "Lagrange", 1)
B = FunctionSpace(submesh, "Lagrange", 1)

W = MixedFunctionSpace(*[A, B])

v, w = TestFunctions(W)
wh = Function(W)
u = wh.sub(0)
psi = wh.sub(1)
n =approximate_facet_normal(VectorFunctionSpace(mesh, "DG", 2))
n.set_allow_extrapolation(True)
psi_k = Function(B)
psi_k.set_allow_extrapolation(True)




alpha = Constant(1.0)
f = Constant((0.0, 0))
from ufl_legacy import conditional
x = SpatialCoordinate(submesh)
g = 0#0.2*x[0]#conditional(gt(x[0], 0.5), 1.0, 0.0)

dx_s = Measure("dx", domain=submesh, metadata={"quadrature_scheme": "vertex"})
F00 = alpha * inner(sigma(u, mesh.geometry().dim()), grad(v)) * dx(domain=mesh) - alpha * inner(f, v) * dx(domain=mesh)
F01 = -inner(psi-psi_k, dot(v, n)) * dx_s
F10 = inner(dot(u, n), w)  * dx_s
F11 = inner(exp(psi), w)  * dx_s - inner(g, w)  * dx_s
F0 = F00 + F01 
F1 = F10 + F11

u_bc = Function(W.sub_space(0))
u_bc.interpolate(Expression(("0", "-0.2"), degree=2))
bc = DirichletBC(W.sub_space(0), u_bc, ff, 1)
bcs = [bc]
jac00 = derivative(F0, u)
jac01 = derivative(F0, psi)
jac10 = derivative(F1, u)
jac11 = derivative(F1, psi)

J = jac00 + jac01 + jac10 + jac11
L = F0 + F1

xdmf =  XDMFFile("output/legacy.xdmf")
xdmf.write(u,  0)
xdmf.write(psi,0)
for i in range(5):
    alpha.assign(1)
    solve(L==0, wh, bcs, solver_parameters={"newton_solver": {"linear_solver": "mumps"}})
    print(assemble_mixed(abs(dot(u, n) - g + exp(psi))*dx(domain=submesh)))
    psi_k.assign(psi)
    xdmf.write(u,  i+1)
    xdmf.write(psi, i+1)
xdmf.close()