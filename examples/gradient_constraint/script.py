import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
import basix.ufl
import ufl
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem

N = 80
M = 77
mesh = dolfinx.mesh.create_unit_square(
    MPI.COMM_WORLD, N, M)
# , dolfinx.mesh.CellType.quadrilateral)

el_0 = basix.ufl.element("Lagrange", mesh.topology.cell_name(), 2)
# el_1 = basix.ufl.element(
#     "DG", mesh.topology.cell_name(), 1, shape=(mesh.geometry.dim,))
el_1 = basix.ufl.element(
    "RT", mesh.topology.cell_name(), 1)
trial_el = basix.ufl.mixed_element([el_0, el_1])
V_trial = dolfinx.fem.functionspace(mesh, trial_el)
# test_el = basix.ufl.mixed_element([el_0, el_1])
# V_test = dolfinx.fem.functionspace(mesh, test_el)
V_test = V_trial

w = dolfinx.fem.Function(V_trial)
u, psi = ufl.split(w)

v, tau = ufl.TestFunctions(V_test)

dx = ufl.Measure("dx",  domain=mesh)
alpha = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1))
x = ufl.SpatialCoordinate(mesh)
phi = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1))
# phi = 1 + x[0]+2*x[1]
w0 = dolfinx.fem.Function(V_trial)
u0, psi0 = ufl.split(w0)

amp = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1))
f = amp*10.
# f = amp*ufl.sin(2*x[0]*ufl.pi)*ufl.sin(x[1]*ufl.pi)
# f = amp*(x[0]*x[1]*(x[0]-1.)*(x[1]-1.))**2

mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)

U, U_to_W = V_trial.sub(0).collapse()
Q, Q_to_W = V_trial.sub(1).collapse()
boundary_dofs = dolfinx.fem.locate_dofs_topological(
    (V_trial.sub(0), U), mesh.topology.dim-1, boundary_facets)

# Create initial condition
p = ufl.TrialFunction(U)
q = ufl.TestFunction(U)
a = ufl.inner(ufl.grad(p), ufl.grad(q))*dx
L = f * q * dx
u_bc_init = dolfinx.fem.Function(U)
bc_init = dolfinx.fem.dirichletbc(u_bc_init, boundary_dofs[1])
lin_prob = LinearProblem(a, L, bcs=[bc_init])
u_init_out = lin_prob.solve()
u_init_out.name = "InitialU"
# u init is equal to the solution of the linear problem
w.x.array[U_to_W] = u_init_out.x.array
w.x.array[Q_to_W] = 0.


F = alpha * ufl.inner(ufl.grad(u), ufl.grad(v))*dx
F -= ufl.inner(ufl.div(psi), v)*dx
F -= alpha * ufl.inner(f, v) * dx
F += ufl.inner(ufl.div(psi0), v)*dx
F += ufl.inner(ufl.grad(u), tau)*dx
non_lin_term = 1/(ufl.sqrt(1 + ufl.dot(psi, psi)))
F -= phi * non_lin_term * ufl.dot(psi, tau)*dx


u_bc = dolfinx.fem.Function(U)
u_bc.x.array[:] = 0
bcs = [dolfinx.fem.dirichletbc(u_bc, boundary_dofs, V_trial.sub(0))]


problem = NonlinearProblem(F, w, bcs=bcs)
solver = NewtonSolver(mesh.comm, problem)
solver.convergence_criterion = "residual"
solver.rtol = 1e-9
solver.atol = 1e-9
solver.max_it = 20
# solver.relaxation_factor = 0.7
solver.error_on_nonconvergence = False


ksp = solver.krylov_solver
opts = PETSc.Options()  # type: ignore
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "superlu"

# For factorisation prefer MUMPS, then superlu_dist, then default.
# sys = PETSc.Sys()  # type: ignore
# if sys.hasExternalPackage("mumps"):
#     opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
# elif sys.hasExternalPackage("superlu_dist"):
#     opts[f"{option_prefix}pc_factor_mat_solver_type"] = "superlu_dist"
ksp.setFromOptions()


# T = dolfinx.fem.functionspace(mesh, ("DG", 0, (mesh.geometry.dim,)))
# grad_psi = dolfinx.fem.Expression(
#     ufl.grad(w.sub(1)), T.element.interpolation_points())
# t = dolfinx.fem.Function(T)
# t.name = "grad_psi"

Q = dolfinx.fem.functionspace(mesh, ("DG", 1, (mesh.geometry.dim, )))
grad_u = dolfinx.fem.Expression(
    ufl.grad(w.sub(0)), Q.element.interpolation_points())
q_out = dolfinx.fem.Function(Q)
q_out.name = "gradu"


dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
# xdmf = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "output.xdmf", "w")
# xdmf.write_mesh(mesh)
u_out = w.sub(0).collapse()
u_out.name = "u"
bp_u = dolfinx.io.VTXWriter(mesh.comm, "u.bp", [u_out], engine="BP4")

bp_grad_u = dolfinx.io.VTXWriter(mesh.comm, "grad_u.bp", [q_out], engine="bp4")


diff = w.sub(0)-w0.sub(0)
L2_squared = ufl.dot(diff, diff)*dx
compiled_diff = dolfinx.fem.form(L2_squared)

for i in range(13):
    alpha.value = 2**i
    num_newton_iterations, converged = solver.solve(w)
    # ksp.view()
    print(
        f"Iteration {i}: {converged=} {num_newton_iterations=} {ksp.getConvergedReason()=}")
    local_diff = dolfinx.fem.assemble_scalar(compiled_diff)
    global_diff = np.sqrt(mesh.comm.allreduce(local_diff, op=MPI.SUM))
    print(f"|delta u |= {global_diff}")
    w0.x.array[:] = w.x.array

    dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)

    u_out.x.array[:] = w.sub(0).x.array[U_to_W]
    bp_u.write(i)
    q_out.interpolate(grad_u)

    bp_grad_u.write(i)

    # t.interpolate(grad_psi)

    # xdmf.write_function(u_out, i)
    # xdmf.write_function(psi_out, i)
    # xdmf.write_function(q_out, i)
    # xdmf.write_function(t, i)
# xdmf.close()
bp_u.close()
bp_grad_u.close()