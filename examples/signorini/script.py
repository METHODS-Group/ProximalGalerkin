# Contact example
# SPDX-License-Identifier: MIT

from mpi4py import MPI
import dolfinx
import dolfinx.fem.petsc
import ufl
import numpy as np
from petsc4py import PETSc
import basix.ufl


dst = dolfinx.default_scalar_type


class NewtonSolver:
    max_iterations: int
    bcs: list[dolfinx.fem.DirichletBC]
    A: PETSc.Mat
    b: PETSc.Vec
    J: dolfinx.fem.Form
    b: dolfinx.fem.Form
    dx: PETSc.Vec
    error_on_nonconvergence: bool

    def __init__(
        self,
        F: list[dolfinx.fem.form],
        J: list[list[dolfinx.fem.form]],
        w: list[dolfinx.fem.Function],
        bcs: list[dolfinx.fem.DirichletBC] | None = None,
        max_iterations: int = 5,
        petsc_options: dict[str, str | float | int | None] = None,
        error_on_nonconvergence: bool = True,
    ):
        self.max_iterations = max_iterations
        self.bcs = [] if bcs is None else bcs
        self.b = dolfinx.fem.petsc.create_vector_block(F)
        self.F = F
        self.J = J
        self.A = dolfinx.fem.petsc.create_matrix_block(J)
        self.dx = self.A.createVecLeft()
        self.w = w
        self.x = dolfinx.fem.petsc.create_vector_block(F)
        self.norm_array = dolfinx.fem.Function(w[0].function_space)
        self.error_on_nonconvergence = error_on_nonconvergence
        # Set PETSc options
        opts = PETSc.Options()
        if petsc_options is not None:
            for k, v in petsc_options.items():
                opts[k] = v

        # Define KSP solver
        self._solver = PETSc.KSP().create(self.b.getComm().tompi4py())
        self._solver.setOperators(self.A)
        self._solver.setFromOptions()

        # Set matrix and vector PETSc options
        self.A.setFromOptions()
        self.b.setFromOptions()

    def _update_solution(self, beta):
        offset_start = 0
        for s in self.w:
            num_sub_dofs = (
                s.function_space.dofmap.index_map.size_local
                * s.function_space.dofmap.index_map_bs
            )
            s.x.array[:num_sub_dofs] -= (
                beta * self.dx.array_r[offset_start : offset_start + num_sub_dofs]
            )
            s.x.scatter_forward()
            offset_start += num_sub_dofs

    def solve(self, tol=1e-6, beta=1.0):
        i = 0

        while i < self.max_iterations:
            # Assemble F(u_{i-1}) - J(u_D - u_{i-1}) and set du|_bc= u_D - u_{i-1}
            self.b.zeroEntries()
            dolfinx.fem.petsc.assemble_vector_block(
                self.b, self.F, self.J, bcs=self.bcs, x0=self.x, scale=-1.0
            )
            self.b.ghostUpdate(
                PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD
            )
            # Assemble Jacobian
            self.A.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix_block(self.A, self.J, bcs=self.bcs)
            self.A.assemble()

            self._solver.solve(self.b, self.dx)
            self._update_solution(beta)

            if self.error_on_nonconvergence:
                assert (
                    self._solver.getConvergedReason() > 0
                ), "Linear solver did not converge"
            else:
                converged = self._solver.getConvergedReason()
                import warnings
                if converged < 0:
                    warnings.warn("Linear solver did not converge, exiting", RuntimeWarning)
                    return 0
            # Compute norm of primal space diff
            norm = self.b.copy()
            norm.zeroEntries()

            dolfinx.cpp.la.petsc.scatter_local_vectors(
                self.x,
                [si.x.petsc_vec.array_r for si in self.w],
                [
                    (
                        si.function_space.dofmap.index_map,
                        si.function_space.dofmap.index_map_bs,
                    )
                    for si in self.w
                ],
            )
            self.x.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )

            local_du, _ = dolfinx.cpp.la.petsc.get_local_vectors(self.dx,                  [
                    (
                        si.function_space.dofmap.index_map,
                        si.function_space.dofmap.index_map_bs,
                    )
                    for si in self.w
                ])
            self.norm_array.x.petsc_vec.array_w[:] = local_du
            self.norm_array.x.petsc_vec.ghostUpdate(
                PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD
            )
            correction_norm = self.norm_array.x.petsc_vec.norm(1)

            print(f"Iteration {i}: Correction norm {correction_norm}")
            if correction_norm < tol:
                break
            i += 1
        return 1

def bottom_boundary(x):
    return np.isclose(x[2], 0.0)


def top_boundary(x):
    return np.isclose(x[2], 1.0)


def epsilon(w):
    return ufl.sym(ufl.grad(w))


def sigma(w, mu, lmbda):
    ew = epsilon(w)
    gdim = ew.ufl_shape[0]
    return 2.0 * mu * epsilon(w) + lmbda * ufl.tr(ufl.grad(w)) * ufl.Identity(gdim)



def solve_contact_problem(degree:int, E:float, nu:float):
    """
    Solve a contact problem with Signorini contact conditions using the Latent Variable Proximal Point algorithm

    :param degree: Degree of primal and latent space

    """
    mesh = dolfinx.mesh.create_unit_cube(
        MPI.COMM_WORLD, 16, 7, 5, dolfinx.cpp.mesh.CellType.hexahedron
    )
    fdim = mesh.topology.dim - 1
    top_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim - 1, top_boundary
    )


    contact_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, bottom_boundary)
    mt = dolfinx.mesh.meshtags(
        mesh,
        mesh.topology.dim - 1,
        contact_facets,
        np.full_like(contact_facets, 1, dtype=np.int32),
    )


    submesh, submesh_to_mesh = dolfinx.mesh.create_submesh(
        mesh, mesh.topology.dim - 1, contact_facets
    )[0:2]


    E, nu = 2.0e4, 0.3
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))




    # NOTE: If we get facet-bubble spaces in DOLFINx we could use an alternative pair here
    element_u = basix.ufl.element(
        "Lagrange", mesh.topology.cell_name(), degree, shape=(mesh.geometry.dim,)
    )
    V = dolfinx.fem.functionspace(mesh, element_u)

    u = dolfinx.fem.Function(V, name="displacement")
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


    metadata = {"quadrature_degree": 4}
    ds = ufl.Measure(
        "ds", domain=mesh, subdomain_data=mt, subdomain_id=1, metadata=metadata
    )
    n = ufl.FacetNormal(mesh)
    n_g = dolfinx.fem.Constant(mesh, dst((0.0, 0.0, -1.0)))
    alpha = dolfinx.fem.Constant(mesh, dst(1.0))
    f = dolfinx.fem.Constant(mesh, dst((0.0, 0.0, 0.0)))
    x = ufl.SpatialCoordinate(mesh)
    g = x[2] + dolfinx.fem.Constant(mesh, dst(0.05))

    F00 = alpha * ufl.inner(sigma(u, mu, lmbda), epsilon(v)) * ufl.dx(
        domain=mesh
    ) - alpha * ufl.inner(f, v) * ufl.dx(domain=mesh)
    F01 = -ufl.inner(psi - psi_k, ufl.dot(v, n)) * ds
    F10 = ufl.inner(ufl.dot(u, n_g), w) * ds
    F11 = ufl.inner(ufl.exp(psi), w) * ds - ufl.inner(g, w) * ds

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
    disp = -0.2
    u_bc.interpolate(
        lambda x: (
            np.full(x.shape[1], 0.0),
            np.full(x.shape[1], 0.0),
            np.full(x.shape[1], disp),
        )
    )
    V0, V0_to_V = V.sub(2).collapse()
    bc = dolfinx.fem.dirichletbc(
        u_bc, dolfinx.fem.locate_dofs_topological(V, fdim, top_facets)
    )
    bcs = [bc]


    solver = NewtonSolver(
        F,
        J,
        [u, psi],
        bcs=bcs,
        max_iterations=25,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        error_on_nonconvergence=False,
    )

    bp = dolfinx.io.VTXWriter(mesh.comm, "uh.bp", [u])
    bp_psi = dolfinx.io.VTXWriter(mesh.comm, "psi.bp", [psi])

    M = 10
    for it in range(M):
        u_bc.x.array[V0_to_V] = disp  # (it+1)/M * disp

        # print((it+1)/M * disp)
        alpha.value += 1
        converged = solver.solve(1e-6, 1)
        psi_k.x.array[:] = psi.x.array
        bp.write(it)
        bp_psi.write(it)
        if not converged:
            break

    bp.close()


if __name__ == "__main__":
    solve_contact_problem(2, E=2.0e4, nu=0.3)