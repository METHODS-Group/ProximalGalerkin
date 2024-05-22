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

E, nu = 2.0e2, 0.3
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

def epsilon(w):
    return sym(grad(w))

def sigma(w, gdim):
    return 2.0 * mu * epsilon(w) + lmbda * tr(grad(w)) * Identity(gdim)

degree = 1


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

A = VectorFunctionSpace(mesh, "Lagrange", degree)
B = FunctionSpace(submesh, "Lagrange", degree)

W = MixedFunctionSpace(*[A, B])

v, w = TestFunctions(W)
wh = Function(W)
u = wh.sub(0)
psi = wh.sub(1)
n =approximate_facet_normal(VectorFunctionSpace(mesh, "DG", 2))

psi_k = Function(B)




alpha = Constant(10.0)
f = Constant((0.0, -10.0))
g = Constant(0.0)


F00 = alpha * inner(sigma(u, mesh.geometry().dim()), grad(v)) * dx(domain=mesh) - alpha * inner(f, v) * dx(domain=mesh)
F01 = inner(psi_k, dot(v, n)) * dx(domain=submesh)
F10 = inner(dot(u, n), w)  * dx(domain=submesh)
F11 = -inner(exp(psi), w)  * dx(domain=submesh) - inner(g, w)  * dx(domain=submesh)
F0 = F00 + F01 
F1 = F10 + F11

u_bc = Function(W.sub_space(0))
u_bc.interpolate(Expression(("0", "-1"), degree=degree))
bc = DirichletBC(W.sub_space(0), u_bc, ff, 1)
bcs = [bc]

jac00 = derivative(F0, u)
jac01 = derivative(F0, psi)
jac10 = derivative(F1, u)
jac11 = derivative(F1, psi)

J = jac00 + jac01 + jac10 + jac11
L = F0 + F1

solve(L==0, wh, bcs, solver_parameters={"newton_solver": {"linear_solver": "mumps"}})
uh = wh.sub(0, deepcopy=True)
uh.rename("displacement", "displacement")

psih = wh.sub(1, deepcopy=True)
psih.rename("aux field", "aux field")
with XDMFFile("output/legacy.xdmf") as xdmf:
    xdmf.write_checkpoint(uh, "uh", 0, append=False)
    xdmf.write_checkpoint(psih, "psi", 0, append=True)
    xdmf.write_checkpoint(n, "nh", 0, append=True)