from mpi4py import MPI

import basix.ufl
import dolfinx
import numpy as np
import ufl

from lvpp import SNESProblem, SNESSolver


class NotConvergedError(Exception):
    pass


mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 1001)

p = 1
el_s = basix.ufl.element("Lagrange", mesh.basix_cell(), p)
el_v = basix.ufl.element("Lagrange", mesh.basix_cell(), p, shape=(mesh.geometry.dim,))
me = basix.ufl.mixed_element([el_s, el_s, el_v])
Z = dolfinx.fem.functionspace(mesh, me)
z = dolfinx.fem.Function(Z, name="Solution")
(u, psi0, psi) = ufl.split(z)
z_test = ufl.TestFunction(Z)
(v, w0, w) = ufl.split(z_test)

z_prev = dolfinx.fem.Function(Z, name="PreviousContinuationSolution")
(_, psi0_prev, psi_prev) = ufl.split(z_prev)
z_iter = dolfinx.fem.Function(Z, name="PreviousLVPPSolution")
(u_iter, psi0_iter, psi_iter) = ufl.split(z_iter)

c = dolfinx.fem.Constant(mesh, 0.0)
dx = ufl.dx(domain=mesh)
E = 0.5 * ufl.inner(ufl.grad(u), ufl.grad(u)) * dx + c * u * dx

x = ufl.SpatialCoordinate(mesh)[0]

# Nonsmooth obstacle
phi0 = ufl.conditional(ufl.le(x, 0.4), 0, ufl.conditional(ufl.ge(x, 0.6), 0, 1))
# Smooth obstacle
(l, r) = (0.2, 0.8)
bump = ufl.exp(-1 / (10 * (x - l) * (r - x))) / ufl.exp(-1 / (10 * (0.5 - l) * (r - 0.5)))
phi0 = ufl.conditional(ufl.le(x, l), 0, ufl.conditional(ufl.ge(x, r), 0, bump))

phic = dolfinx.fem.Constant(mesh, 100.0)
phi = ufl.conditional(ufl.le(x, 0.2), phic, ufl.conditional(ufl.gt(x, 0.8), phic, 100))

alpha = dolfinx.fem.Constant(mesh, 1.0)
F = (
    alpha * ufl.derivative(E, z, z_test)
    + ufl.inner(psi0, v) * dx
    + ufl.inner(psi, ufl.grad(v)) * dx
    - ufl.inner(psi0_iter, v) * dx
    - ufl.inner(psi_iter, ufl.grad(v)) * dx
    + ufl.inner(u, w0) * dx
    - ufl.inner(ufl.exp(psi0), w0) * dx
    - ufl.inner(phi0, w0) * dx
    + ufl.inner(ufl.grad(u), w) * dx
    - ufl.inner(phi * psi / ufl.sqrt(1 + ufl.dot(psi, psi)), w) * dx
)
mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
bc_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
bc_dofs = dolfinx.fem.locate_dofs_topological(Z.sub(0), mesh.topology.dim - 1, bc_facets)
bcs = [dolfinx.fem.dirichletbc(0.0, bc_dofs, Z.sub(0))]

problem = SNESProblem(F, z, bcs=bcs)

sp = {
    "snes_monitor": None,
    "snes_converged_reason": None,
    "snes_linesearch_type": "l2",
    "snes_linesearch_maxstep": 1,
    "snes_atol": 1.0e-6,
    "snes_rtol": 1.0e-6,
    "snes_stol": 1e-14,
    "snes_linesearch_monitor": None,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_14": 500,
}


L2_u = dolfinx.fem.form(ufl.inner(u - u_iter, u - u_iter) * dx)
L2_z = dolfinx.fem.form(ufl.inner(z - z_prev, z - z_iter) * dx)

V_primal, primal_to_Z = Z.sub(0).collapse()
primal_points = V_primal.element.interpolation_points()
u_out = dolfinx.fem.Function(V_primal, name="u")
u_conform_out = dolfinx.fem.Function(V_primal, name="conform_u")
conform_expr = dolfinx.fem.Expression(ufl.exp(psi0) + phi0, primal_points)
obstacle = dolfinx.fem.Function(V_primal, name="obstacle")
obstacle_expr = dolfinx.fem.Expression(phi0, primal_points)
vtx = dolfinx.io.VTXWriter(mesh.comm, "output.bp", [u_out, u_conform_out, obstacle])

G = dolfinx.fem.functionspace(mesh, ("DG", 0, (mesh.geometry.dim,)))
G_points = G.element.interpolation_points()
grad_u = dolfinx.fem.Expression(ufl.grad(u), G_points)
grad_func = dolfinx.fem.Function(G, name="grad_u")
phi_func = dolfinx.fem.Function(G, name="phi")
phi_expr = dolfinx.fem.Expression(phi, G_points)
conform_grad = dolfinx.fem.Expression(ufl.grad(ufl.exp(psi0) + phi0), G_points)
conform_grad_func = dolfinx.fem.Function(G, name="grad_u_conform")
vtx_grad = dolfinx.io.VTXWriter(
    mesh.comm, "output_grad.bp", [grad_func, phi_func, conform_grad_func]
)
NFAIL_MAX = 50
phis = [3, 2, 1, 0.5, 0.1, 0.01]
num_newton_iterations = np.zeros_like(phis, dtype=np.int32)
num_lvpp_iterations = np.zeros_like(phis, dtype=np.int32)
problem = SNESProblem(F, z, bcs=bcs)
for i, phi_ in enumerate(phis):
    phic.value = phi_
    if mesh.comm.rank == 0:
        print(f"Solving for phi = {float(phic)}", flush=True)
    alpha.value = 1
    z_iter.interpolate(z)
    k = 1
    r = 2
    nfail = 0
    while nfail <= NFAIL_MAX:
        try:
            if mesh.comm.rank == 0:
                print(f"Attempting {k=} alpha={float(alpha)}", flush=True)
            solver = SNESSolver(problem, sp)
            converged_reason, num_iterations = solver.solve()
            num_newton_iterations[i] += num_iterations
            if num_iterations == 0 and converged_reason > 0:
                # solver didn't actually get to do any work,
                # we've just reduced alpha so much that the initial guess
                # satisfies the PDE
                raise NotConvergedError("Not converged")
            if converged_reason < 0:
                raise NotConvergedError("Not converged")
        except NotConvergedError:
            nfail += 1
            if mesh.comm.rank == 0:
                print(f"Failed to converge, {k=} alpha={float(alpha)}", flush=True)
            alpha.value /= 2
            if k == 1:
                z.interpolate(z_prev)
            else:
                z.interpolate(z_iter)

            if nfail >= NFAIL_MAX:
                if mesh.comm.rank == 0:
                    print(f"Giving up. phic={float(phic)} alpha={float(alpha)} {k=}", flush=True)
                break
            else:
                continue

        # Termination
        nrm = np.sqrt(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(L2_u), op=MPI.SUM))
        if mesh.comm.rank == 0:
            print(
                f"Solved {k=} phi={float(phic)} alpha={float(alpha)} ||u_{k} - u_{k - 1}|| = {nrm}",
                flush=True,
            )
        num_lvpp_iterations[i] += 1
        if nrm < 1.0e-4:
            break

        # Update alpha
        if num_iterations <= 4:
            alpha.value *= r
        elif num_iterations >= 10:
            alpha.value /= r

        # Update z_iter
        z_iter.interpolate(z)
        k += 1

    # Update output
    u_out.x.array[:] = z.x.array[primal_to_Z]
    u_conform_out.interpolate(conform_expr)
    obstacle.interpolate(obstacle_expr)
    vtx.write(phi_)
    phi_func.interpolate(phi_expr)
    grad_func.interpolate(grad_u)
    conform_grad_func.interpolate(conform_grad)
    vtx_grad.write(phi_)

vtx_grad.close()

vtx.close()
print(f"{num_lvpp_iterations=}")
print(f"{num_newton_iterations=}")
