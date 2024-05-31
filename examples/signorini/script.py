# docker run -ti -v $(pwd):/root/shared -w /root/shared jpdean/mixed_domain

# SPDX-License-Identifier: MIT

from mpi4py import MPI
import dolfinx
import dolfinx.fem.petsc
import ufl
import numpy as np
from petsc4py import PETSc


class NewtonSolver():
    max_iterations: int
    bcs: list[dolfinx.fem.DirichletBC]
    A: PETSc.Mat
    b: PETSc.Vec
    J: dolfinx.fem.Form
    b: dolfinx.fem.Form
    dx: PETSc.Vec
    def __init__(self, F:list[dolfinx.fem.form], J:list[list[dolfinx.fem.form]], w: list[dolfinx.fem.Function], 
                 bcs: list[dolfinx.fem.DirichletBC]|None=None, max_iterations:int=5,
                 petsc_options: dict[str, str|float|int|None]=None,
                 problem_prefix = "newton"):
        self.max_iterations = max_iterations
        self.bcs = [] if bcs is None else bcs
        self.b = dolfinx.fem.petsc.create_vector_block(F)
        self.F = F
        self.J = J
        self.A = dolfinx.fem.petsc.create_matrix_block(J)
        self.dx = self.A.createVecLeft()
        self.w = w
        self.x = dolfinx.fem.petsc.create_vector_block(F)

        # Set PETSc options
        opts = PETSc.Options()
        opts.prefixPush(problem_prefix)
        if petsc_options is not None:
            for k, v in petsc_options.items():
                opts[k] = v
        opts.prefixPop()

        # Define KSP solver    
        self._solver = PETSc.KSP().create(self.b.getComm().tompi4py())
        self._solver.setOperators(self.A)
        self._solver.setFromOptions()

        # Set matrix and vector PETSc options
        self.A.setOptionsPrefix(problem_prefix)
        self.A.setFromOptions()
        self.b.setOptionsPrefix(problem_prefix)
        self.b.setFromOptions()

   
    def solve(self, tol=1e-6, beta=1.0):
        i = 0


        while i < self.max_iterations:
            dolfinx.cpp.la.petsc.scatter_local_vectors(
                self.x,
                [si.x.petsc_vec.array_r for si in self.w],
                [
                    (si.function_space.dofmap.index_map, si.function_space.dofmap.index_map_bs)
                    for si in self.w
                ])
            self.x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            # Assemble F(u_{i-1}) - J(u_D - u_{i-1}) and set du|_bc= u_D - u_{i-1}
            self.b.zeroEntries()
            dolfinx.fem.petsc.assemble_vector_block(self.b, self.F,self.J, bcs=self.bcs,x0=self.x, scale=-1.0)
            self.b.ghostUpdate(PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD)
            # Assemble Jacobian
            self.A.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix_block(self.A, self.J, bcs=self.bcs)
            self.A.assemble()
            
            
            self._solver.solve(self.b, self.dx)
            assert self._solver.getConvergedReason() > 0, "Linear solver did not converge"
            #breakpoint()
            offset_start = 0
            for s in self.w:
                num_sub_dofs = s.function_space.dofmap.index_map.size_local * s.function_space.dofmap.index_map_bs
                s.x.array[:num_sub_dofs] -= beta*self.dx.array_r[offset_start:offset_start+num_sub_dofs]
                s.x.scatter_forward()
                offset_start += num_sub_dofs
            # Compute norm of update

            correction_norm = self.dx.norm(0)
            print(f"Iteration {i}: Correction norm {correction_norm}")
            if correction_norm < tol:
                break
            i += 1



def bottom_boundary(x):
    return np.isclose(x[2], 0.0)

def top_boundary(x):
    return np.isclose(x[2], 1.0)


mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 10, 10, 10, dolfinx.cpp.mesh.CellType.tetrahedron)
degree = 1
gdim = mesh.geometry.dim
fdim = mesh.topology.dim - 1
top_facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, top_boundary)


contact_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, bottom_boundary)
num_facets_local = mesh.topology.index_map(fdim).size_local + mesh.topology.index_map(fdim).num_ghosts
mt = dolfinx.mesh.meshtags(mesh, mesh.topology.dim-1, contact_facets, np.full_like(contact_facets, 1, dtype=np.int32))


submesh, submesh_to_mesh = dolfinx.mesh.create_submesh(mesh, mesh.topology.dim-1, contact_facets)[0:2]


E, nu = 2.0e4, 0.3
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

def epsilon(w):
    return ufl.sym(ufl.grad(w))

def sigma(w, gdim):
    return 2.0 * mu * epsilon(w) + lmbda * ufl.tr(ufl.grad(w)) * ufl.Identity(gdim)

import basix.ufl
enriched_element = basix.ufl.enriched_element(
    [basix.ufl.element("Lagrange", mesh.topology.cell_name(), degree),
     basix.ufl.element("Bubble", mesh.topology.cell_name(), degree+mesh.geometry.dim)])
# FIXME: We need facet bubbles
#element_u = basix.ufl.blocked_element(enriched_element, shape=(mesh.geometry.dim, ))
#element_p = basix.ufl.element("Lagrange", submesh.topology.cell_name(), degree-1, discontinuous=True)

element_u = basix.ufl.element("Lagrange", mesh.topology.cell_name(), degree, shape=(mesh.geometry.dim, ))
V = dolfinx.fem.functionspace(mesh, element_u)

u = dolfinx.fem.Function(V)
v = ufl.TestFunction(V)


element_p = basix.ufl.element("Lagrange", submesh.topology.cell_name(), degree)
W = dolfinx.fem.functionspace(submesh, element_p)

v = ufl.TestFunction(V)
psi = dolfinx.fem.Function(W)
psi_k = dolfinx.fem.Function(W)
w = ufl.TestFunction(W)
facet_imap = mesh.topology.index_map(fdim)
num_facets = facet_imap.size_local + facet_imap.num_ghosts
mesh_to_submesh = np.full(num_facets, -1)
mesh_to_submesh[submesh_to_mesh] = np.arange(len(submesh_to_mesh))
entity_maps = {submesh: mesh_to_submesh}

ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt, subdomain_id=1)
n = ufl.FacetNormal(mesh)
alpha = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0))
f = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type((0.0, 0.0, -0.0)))
x = ufl.SpatialCoordinate(mesh)
g = dolfinx.fem.Constant(submesh, dolfinx.default_scalar_type(0.0))
#g = ufl.conditional(ufl.lt((x[0]-0.5)**2+(x[1]-0.5)&&, 0.9), 0, 1)

F00 = alpha * ufl.inner(sigma(u, mesh.geometry.dim), ufl.grad(v)) * ufl.dx(domain=mesh) - alpha * ufl.inner(f, v) * ufl.dx(domain=mesh)
F01 = -ufl.inner(psi-psi_k, ufl.dot(v, n)) * ds
F10 = ufl.inner(ufl.dot(u, n), w)  * ds
F11 = ufl.inner(ufl.exp(psi), w)  * ds - ufl.inner(g, w)  * ds
F0 = F00 + F01 
F1 = F10 + F11

F0 = F00 + F01 
F1 = F10 + F11
residual_0 = dolfinx.fem.form(F0, entity_maps=entity_maps)
residual_1 = dolfinx.fem.form(F1, entity_maps=entity_maps)
jac00 = ufl.derivative(F0, u)
jac01 = ufl.derivative(F0, psi)
jac10 = ufl.derivative(F1, u)
jac11 = ufl.derivative(F1, psi)
J00 = dolfinx.fem.form(jac00, entity_maps=entity_maps)
J01 = dolfinx.fem.form(jac01, entity_maps=entity_maps)
J10 = dolfinx.fem.form(jac10, entity_maps=entity_maps)
J11 = dolfinx.fem.form(jac11, entity_maps=entity_maps)

J = [[J00, J01], [J10, J11]]
F = [residual_0, residual_1]

u_bc = dolfinx.fem.Function(V)
disp = -0.1
u_bc.interpolate(lambda x: (np.full(x.shape[1], 0.0), np.full(x.shape[1], 0.0), np.full(x.shape[1], disp)))
V0, V0_to_V = V.sub(2).collapse()
bc = dolfinx.fem.dirichletbc(u_bc, dolfinx.fem.locate_dofs_topological(V, fdim, top_facets))
bcs = [bc]


solver = NewtonSolver(F, J, [u, psi], bcs=bcs, max_iterations=25, petsc_options={"ksp_type":"preonly", "pc_type":"lu", 
"pc_factor_mat_solver_type":"mumps"})

bp = dolfinx.io.VTXWriter(mesh.comm, "uh.bp", [u], engine="BP4")
bp_psi = dolfinx.io.VTXWriter(mesh.comm, "psi.bp", [psi], engine="BP4")

M = 10
for it in range(M):
    u_bc.x.array[V0_to_V] = (it+1)/M * disp
    print((it+1)/M * disp)
    #alpha.value += 0.
    solver.solve(1e-4, 1)
    psi_k.x.array[:] = psi.x.array
    bp.write(it)
    bp_psi.write(it)
bp.close()