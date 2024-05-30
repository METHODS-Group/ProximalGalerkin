# docker run -ti -v $(pwd):/root/shared -w /root/shared jpdean/mixed_domain

# SPDX-License-Identifier: LGPL-3.0-or-later

from mpi4py import MPI
import dolfinx
import dolfinx.fem.petsc
import ufl
import numpy as np
from petsc4py import PETSc
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10, dolfinx.cpp.mesh.CellType.quadrilateral)

def bottom_boundary(x):
    return np.isclose(x[1], 0.0)

def top_boundary(x):
    return np.isclose(x[1], 1.0)


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

    # def assemble_vector(self):
    #     """Bassed of FEniCSx block assembly"""
    #     maps = [
    #     (form.function_spaces[0].dofmap.index_map, form.function_spaces[0].dofmap.index_map_bs)
    #     for form in self.F
    # ]

    #     x0_local = dolfinx.cpp.la.petsc.get_local_vectors(self.dx, maps)
    #     x0_sub = x0_local

    #     # Compute the residual without bcs and scale by -1
    #     bcs1 = dolfinx.fem.bcs_by_block(dolfinx.fem.extract_function_spaces(self.J, 1), bcs)
    #     with self.b.localForm() as b_local:
    #         b_local.set(0.0)

    #     b_local = dolfinx.cpp.la.petsc.get_local_vectors(self.b, maps)
    #     for b_sub, L_sub, a_sub in zip(b_local, self.F, self.J):   
    #         #  Compute  b - J(u_D - u_{i-1})    
    #         dolfinx.fem.assemble_vector(b_sub, L_sub)
    #         dolfinx.fem.apply_lifting(b_sub, a_sub, bcs1, x0_local, -1)

    #     dolfinx.cpp.la.petsc.scatter_local_vectors(self.b, b_local, maps)
    #     self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)          
        
    #     bcs0 = dolfinx.fem.bcs_by_block(dolfinx.fem.extract_function_spaces(self.F), bcs)
    #     offset = 0
    #     b_array = self.b.getArray(readonly=False)
    #     print(b_array)
    #     for submap, bc, _x0 in zip(maps, bcs0, x0_sub):
    #         size = submap[0].size_local * submap[1]
    #         dolfinx.fem.set_bc(b_array[offset : offset + size], bc, _x0, -1)
    #         offset += size
    #         print(_x0)

    #     self.b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

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
            #self.assemble_vector()
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
                print(s.x.array, self.b.array_r[offset_start:offset_start+num_sub_dofs])
            # Compute norm of update
            correction_norm = self.dx.norm(0)
            print(f"Iteration {i}: Correction norm {correction_norm}")
            if correction_norm < tol:
                break
            i += 1

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


V = dolfinx.fem.functionspace(mesh, ("Lagrange", degree, (mesh.geometry.dim, )))
u = dolfinx.fem.Function(V)
v = ufl.TestFunction(V)
W = dolfinx.fem.functionspace(submesh, ("Lagrange", degree))

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
f = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type((0.0, 0.0)))
g = dolfinx.fem.Constant(submesh, dolfinx.default_scalar_type(0.0))


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
u_bc.interpolate(lambda x: (np.full(x.shape[1], 0.0), np.full(x.shape[1], -0.05)))
bc = dolfinx.fem.dirichletbc(u_bc, dolfinx.fem.locate_dofs_topological(V, fdim, top_facets))
bcs = [bc]


solver = NewtonSolver(F, J, [u, psi], bcs=bcs, max_iterations=5)

bp = dolfinx.io.VTXWriter(mesh.comm, "uh.bp", [u], engine="BP4")
solver.solve(1e-6, 1)
bp.write(0.0)
bp.close()