# docker run -ti -v $(pwd):/root/shared -w /root/shared jpdean/mixed_domain
from mpi4py import MPI
import dolfinx
import dolfinx.fem.petsc
import ufl
import numpy as np
from petsc4py import PETSc
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3, dolfinx.cpp.mesh.CellType.triangle)

def bottom_boundary(x):
    return np.isclose(x[1], 0.0)

def top_boundary(x):
    return np.isclose(x[1], 1.0)


degree = 1
gdim = mesh.geometry.dim
fdim = mesh.topology.dim - 1
top_facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, top_boundary)


contact_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, bottom_boundary)
num_facets_local = mesh.topology.index_map(fdim).size_local + mesh.topology.index_map(fdim).num_ghosts
mt = dolfinx.mesh.meshtags(mesh, mesh.topology.dim-1, contact_facets, np.full_like(contact_facets, 1, dtype=np.int32))


submesh, submesh_to_mesh = dolfinx.mesh.create_submesh(mesh, mesh.topology.dim-1, contact_facets)[0:2]


E, nu = 2.0e2, 0.3
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

def epsilon(w):
    return ufl.sym(ufl.grad(w))

def sigma(w, gdim):
    return 2.0 * mu * epsilon(w) + lmbda * ufl.tr(ufl.grad(w)) * ufl.Identity(gdim)


V = dolfinx.fem.VectorFunctionSpace(mesh, ("Lagrange", degree))
u = dolfinx.fem.Function(V)
v = ufl.TestFunction(V)
W = dolfinx.fem.FunctionSpace(submesh, ("Lagrange", degree))

v = ufl.TestFunction(V)
psi = dolfinx.fem.Function(W)
psi_k = dolfinx.fem.Function(W)
w = ufl.TestFunction(W)

facet_imap = mesh.topology.index_map(fdim)
num_facets = facet_imap.size_local + facet_imap.num_ghosts
mesh_to_submesh = np.full(num_facets, -1)
mesh_to_submesh[submesh_to_mesh] = np.arange(len(submesh_to_mesh))
entity_maps = {submesh: mesh_to_submesh}

ds = ufl.Measure("ds", domain=mesh)
n = ufl.FacetNormal(mesh)
alpha = dolfinx.fem.Constant(mesh, 1.0)
f = dolfinx.fem.Constant(mesh, (0.0, -10.0))
g = dolfinx.fem.Constant(submesh, 0.0)


F00 = alpha * ufl.inner(sigma(u, mesh.geometry.dim), ufl.grad(v)) * ufl.dx - alpha * ufl.inner(f, v) * ufl.dx
F01 = ufl.inner(psi_k, ufl.dot(v, n)) * ds
F10 = ufl.inner(ufl.dot(u, n), w) * ds
F11 = -ufl.inner(ufl.exp(psi), w) * ds - ufl.inner(g, w) * ds
F0 = F00 + F01 
F1 = F10 + F11
residual_0 = dolfinx.fem.form(F0, entity_maps=entity_maps)
residual_1 = dolfinx.fem.form(F1, entity_maps=entity_maps)

u_bc = dolfinx.fem.Function(V)
u_bc.interpolate(lambda x: (np.full(x.shape[1], 0.0), np.full(x.shape[1], -1.0)))
bc = dolfinx.fem.dirichletbc(u_bc, dolfinx.fem.locate_dofs_topological(V, fdim, top_facets))
bcs = [bc]

jac00 = ufl.derivative(F0, u)
jac01 = ufl.derivative(F0, psi)
jac10 = ufl.derivative(F1, u)
jac11 = ufl.derivative(F1, psi)

J00 = dolfinx.fem.form(jac00, entity_maps=entity_maps)
J01 = dolfinx.fem.form(jac01, entity_maps=entity_maps)
J10 = dolfinx.fem.form(jac10, entity_maps=entity_maps)
J11 = dolfinx.fem.form(jac11, entity_maps=entity_maps)

J = [[J00, J01], [J10, J11]]
L = [residual_0, residual_1]

# Assemble matrices

A = dolfinx.fem.petsc.assemble_matrix_block(J, bcs=bcs)
A.assemble()
breakpoint()

b = dolfinx.fem.petsc.create_vector_block(L)
dolfinx.fem.petsc.assemble_vector_block(b, L, J, bcs=bcs)

# Solve
ksp = PETSc.KSP().create(mesh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("superlu_dist")

# Compute solution
x = A.createVecLeft()
ksp.solve(b, x)
print(ksp.getConvergedReason())
print(x.array_r)