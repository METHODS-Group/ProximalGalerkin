import dolfinx.fem.petsc
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
import basix.ufl
import ufl
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem.petsc import NonlinearProblem
# from read_mobius_dolfinx import read_mobius_strip

# mesh = read_mobius_strip("./mobius-strip.mesh/Cycle000000/proc000000.vtu")
from mob_create import create_mobius_mesh

M = 10
degree = 2
mesh = create_mobius_mesh(M, degree=degree)
import dolfinx.io

# x = mesh.geometry.x
# pos = np.isclose(x[:, 1], 0.0) & np.isclose(x[:, 2], 0.0)
# x[pos, :] += 0.1

el_0 = basix.ufl.element("DG", mesh.topology.cell_name(), 3)
el_1 = basix.ufl.element("RT", mesh.topology.cell_name(), 4)
trial_el = basix.ufl.mixed_element([el_0, el_1])
V = dolfinx.fem.functionspace(mesh, trial_el)

w = dolfinx.fem.Function(V)
u, psi = ufl.split(w)

v, tau = ufl.TestFunctions(V)

dx = ufl.Measure("dx", domain=mesh)

uD = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0))
U, U_to_W = V.sub(0).collapse()
Q, Q_to_W = V.sub(1).collapse()
f = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1))


alpha = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1))
phi = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1))
w0 = dolfinx.fem.Function(V)
u0, psi0 = ufl.split(w0)

F = ufl.inner(ufl.div(psi), v) * dx
F -= ufl.inner(ufl.div(psi0), v) * dx
F += alpha * ufl.inner(f, v) * dx

non_lin_term = 1 / (ufl.sqrt(1 + ufl.dot(psi, psi)))
F += ufl.inner(u, ufl.div(tau)) * dx
F += phi * non_lin_term * ufl.dot(psi, tau) * dx


J = ufl.derivative(F, w)

tol = 1e-5

problem = NonlinearProblem(F, w, bcs=[], J=J)
solver = NewtonSolver(mesh.comm, problem)
solver.convergence_criterion = "residual"
solver.rtol = tol
solver.atol = tol
solver.max_it = 100
solver.error_on_nonconvergence = True


ksp = solver.krylov_solver
opts = PETSc.Options()  # type: ignore
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
opts[f"{option_prefix}ksp_error_if_not_converged"] = True
ksp.setFromOptions()

dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
V_out = dolfinx.fem.functionspace(mesh, ("DG", mesh.geometry.cmap.degree))
u_out = dolfinx.fem.Function(V_out)
u_out.name = "u"
bp_u = dolfinx.io.VTXWriter(mesh.comm, "u.bp", [u_out], engine="BP4")
diff = w.sub(0) - w0.sub(0)
L2_squared = ufl.dot(diff, diff) * dx
compiled_diff = dolfinx.fem.form(L2_squared)

Q_out = dolfinx.fem.functionspace(
    mesh, ("DG", mesh.geometry.cmap.degree, (mesh.geometry.dim,))
)
q_out = dolfinx.fem.Function(Q_out)


q_expr = dolfinx.fem.Expression(
    ufl.grad(w.sub(0)), Q_out.element.interpolation_points()
)

psi_out = dolfinx.fem.Function(Q_out)
psi_out.name = "psi"

vtx_psi = dolfinx.io.VTXWriter(
    mesh.comm,
    "psi.bp",
    [psi_out],
    engine="BP5",
)

vtx_gu = dolfinx.io.VTXWriter(
    mesh.comm,
    "gradu.bp",
    [q_out],
    engine="BP5",
)
try:
    newton_iterations = []
    for i in range(1, 100):
        alpha.value = min(2**i, 10)

        num_newton_iterations, converged = solver.solve(w)
        newton_iterations.append(num_newton_iterations)
        print(
            f"Iteration {i}: {converged=} {num_newton_iterations=} {ksp.getConvergedReason()=}"
        )
        local_diff = dolfinx.fem.assemble_scalar(compiled_diff)
        global_diff = np.sqrt(mesh.comm.allreduce(local_diff, op=MPI.SUM))
        print(f"|delta u |= {global_diff}")
        w0.x.array[:] = w.x.array

        dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)

        u_out.interpolate(w.sub(0))
        bp_u.write(i)
        q_out.interpolate(q_expr)
        psi_out.interpolate(w.sub(1))
        vtx_gu.write(i)
        vtx_psi.write(i)

        if global_diff < 5 * tol:
            break
finally:
    bp_u.close()
    vtx_psi.close()

print(
    f"Num LVPP iterations {i}, Total number of newton iterations {sum(newton_iterations)}"
)
print(f"{min(newton_iterations)=} and {max(newton_iterations)=}")
