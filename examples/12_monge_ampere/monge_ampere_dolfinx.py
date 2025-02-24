from mpi4py import MPI

import basix.ufl
import dolfinx
import numpy as np
import scipy as sc
import ufl
from expm import expm

from lvpp import SNESProblem, SNESSolver

nonsmooth = False
z_prev = None
mesh = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [[-1, -1], [1, 1]], [2, 2])
mesh.geometry

sp = {
    "snes_monitor": None,
    "snes_linesearch_type": "l2",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "pc_svd_monitor": None,
    "mat_mumps_icntl_14": 10000,
}


def extract_num_dofs(V: dolfinx.fem.FunctionSpace):
    return V.dofmap.index_map.size_global * V.dofmap.index_map_bs


errors = []
for j in range(3, 15):
    X = (x, y) = ufl.SpatialCoordinate(mesh)
    gdim = mesh.geometry.dim

    if nonsmooth:
        u_exact = ufl.conditional(
            ufl.le(ufl.dot(X, X), 1 / 4),
            2 * ufl.dot(X, X),
            2 * ufl.dot(X, X) + 2 * (ufl.sqrt(ufl.dot(X, X)) - 1 / 2) ** 2,
        )
    else:
        u_exact = ufl.exp(ufl.dot(X, X) / 2)

    def hessian(u):
        return ufl.as_tensor([[u.dx(0).dx(0), u.dx(0).dx(1)], [u.dx(1).dx(0), u.dx(1).dx(1)]])

    rho = ufl.det(hessian(u_exact))  # forcing term
    g = u_exact  # boundary data

    k = j
    el_V = basix.ufl.element("Lagrange", mesh.basix_cell(), k)
    el_U = basix.ufl.element("Lagrange", mesh.basix_cell(), k + 1, shape=(gdim,))
    el_W = basix.ufl.element("Lagrange", mesh.basix_cell(), k, shape=(3,))
    me = basix.ufl.mixed_element([el_V, el_U, el_W])
    Z = dolfinx.fem.functionspace(mesh, me)

    C = dolfinx.fem.functionspace(mesh, ("DG", el_U.degree))
    rho_c = dolfinx.fem.Function(C, name="Density")
    rho_expr = dolfinx.fem.Expression(rho, C.element.interpolation_points())
    rho_c.interpolate(rho_expr)
    with dolfinx.io.VTXWriter(mesh.comm, "output/density.bp", [rho_c]) as bp:
        bp.write(0.0)
    assert all(rho_c.x.array > 0)

    z = dolfinx.fem.Function(Z)

    print(
        f"# degrees of freedom: {extract_num_dofs(Z)} {
            tuple(extract_num_dofs(Z.sub(i).collapse()[0]) for i in range(Z.num_sub_spaces))
        }",
        flush=True,
    )

    (u, p, Psi) = ufl.split(z)
    psi = ufl.as_tensor([[Psi[0], Psi[1]], [Psi[1], Psi[2]]])
    (v, q, Phi) = ufl.split(ufl.TestFunction(Z))
    phi = ufl.as_tensor([[Phi[0], Phi[1]], [Phi[1], Phi[2]]])
    n = ufl.FacetNormal(mesh)

    dx = ufl.dx
    F = (
        ufl.inner(ufl.tr(psi) - ufl.ln(rho), v) * dx
        + ufl.inner(p, q) * dx
        - ufl.inner(ufl.grad(u), q) * dx
        + ufl.inner(ufl.grad(p), phi) * dx
        - ufl.inner(expm(psi), phi) * dx
    )
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    bndry_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    V, _ = Z.sub(0).collapse()
    u_bc = dolfinx.fem.Function(V)
    bc_expr = dolfinx.fem.Expression(g, V.element.interpolation_points())
    u_bc.interpolate(bc_expr)
    bndry_dofs = dolfinx.fem.locate_dofs_topological(
        (Z.sub(0), V), mesh.topology.dim - 1, bndry_facets
    )
    bc = dolfinx.fem.dirichletbc(u_bc, bndry_dofs, Z.sub(0))

    # Construct initial guesses
    if z_prev is not None:
        cell_map = mesh.topology.index_map(mesh.topology.dim)
        num_cells = cell_map.size_local + cell_map.num_ghosts
        cells = np.arange(num_cells, dtype=np.int32)
        for i in range(Z.num_sub_spaces):
            interpolation_data = dolfinx.fem.create_interpolation_data(
                z.function_space.sub(i), z_prev.function_space.sub(i), cells, padding=1e-6
            )
            z.sub(i).interpolate_nonmatching(
                z_prev.sub(i), cells, interpolation_data=interpolation_data
            )
    else:
        if nonsmooth:
            u_guess = 2 * ufl.dot(X, X)
        else:
            u_guess = x**2 + y**2
        B = dolfinx.fem.functionspace(mesh, ("DG", el_W.degree, (gdim, gdim)))
        z.sub(0).interpolate(
            dolfinx.fem.Expression(u_guess, Z.sub(0).element.interpolation_points())
        )
        z.sub(1).interpolate(
            dolfinx.fem.Expression(ufl.grad(u_guess), Z.sub(1).element.interpolation_points())
        )
        psi_init = dolfinx.fem.Function(B)
        psi_init.interpolate(
            dolfinx.fem.Expression(hessian(z.sub(0)), B.element.interpolation_points())
        )
        loghessian = dolfinx.fem.Function(B, name="LogHessian")
        tmp_arr = psi_init.x.array.reshape(-1, gdim, gdim)
        for i in range(len(psi_init.x.array) // (gdim * gdim)):
            loghessian.x.array[i * gdim**2 : (i + 1) * gdim**2] = sc.linalg.logm(
                tmp_arr[i]
            ).flatten()
        z.sub(2).interpolate(
            dolfinx.fem.Expression(
                ufl.as_vector([loghessian[0, 0], loghessian[0, 1], loghessian[1, 1]]),
                Z.sub(2).element.interpolation_points(),
            )
        )

        J = ufl.derivative(F, z)
        pulled_back_a = ufl.algorithms.compute_form_data(
            J,
            do_apply_function_pullbacks=True,
            do_apply_integral_scaling=True,
            do_apply_geometry_lowering=True,
            preserve_geometry_types=(ufl.classes.Jacobian,),
        )
        print(list(itg.metadata() for itg in pulled_back_a.integral_data[0].integrals))
        with dolfinx.io.VTXWriter(mesh.comm, "output/initialguess.bp", [z.sub(2).collapse()]) as bp:
            bp.write(0.0)
        #

    problem = SNESProblem(F, z, bcs=[bc])
    solver = SNESSolver(problem, sp)
    converged_reason, num_iterations = solver.solve()
    print(f"Converged reason: {converged_reason}, iterations: {num_iterations}", flush=True)
    assert converged_reason > 0, "Solver did not converge"
    z_prev = z

    L2_error = ufl.inner(u - u_exact, u - u_exact) * dx
    local_error = dolfinx.fem.assemble_scalar(dolfinx.fem.form(L2_error))
    global_error = np.sqrt(mesh.comm.allreduce(local_error, op=MPI.SUM))
    errors.append(global_error)

    with dolfinx.io.VTXWriter(mesh.comm, f"output/solution_{j}.bp", [z.sub(0).collapse()]) as vtx:
        vtx.write(0.0)

    # mesh, _, _ = dolfinx.mesh.refine(mesh)


# Not relevant for p-refinement
# def convergence_orders(x):
#     return np.log2(np.array(x)[:-1] / np.array(x)[1:])

print("Errors", errors, flush=True)
# print("Convergence orders: ", convergence_orders(errors), flush=True)
