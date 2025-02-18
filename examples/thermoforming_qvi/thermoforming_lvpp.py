from mpi4py import MPI

import basix.ufl
import dolfinx.fem.petsc
import dolfinx.nls.petsc
import numpy as np
from ufl import (
    SpatialCoordinate,
    TestFunctions,
    conditional,
    derivative,
    dx,
    exp,
    grad,
    inner,
    lt,
    max_value,
    pi,
    sin,
    split,
)

from lvpp import SNESProblem, SNESSolver

# Define domain
M = 150
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, M, M)

# Define finite element spaces, the unknown solution and the test functions
el = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
me = basix.ufl.mixed_element([el, el, el])
V = dolfinx.fem.functionspace(mesh, me)
s = dolfinx.fem.Function(V)
u, T, psi = split(s)
v, q, w = TestFunctions(V)

# Set up function for `g`
_q = 0.01
bound0 = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0))
bound1 = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(_q))
assert float(bound0) < float(bound1)


def g(s):
    cond0 = lt(s, bound0)
    cond1 = lt(s, bound1)
    f0 = 1
    f1 = 1 - s / bound1
    f2 = 0
    return conditional(cond0, f0, conditional(cond1, f1, f2))


# Set problem specific parameters
x, y = SpatialCoordinate(mesh)
s_prev = dolfinx.fem.Function(V)
u_prev, _, psi_prev = split(s_prev)
beta = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0))
alpha = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(2 ** (-6)))
f = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(25))
Phi0 = 1 - 2 * max_value(abs(x - 0.5), abs(y - 0.5))
xi = sin(pi * x) * sin(pi * y)

# Create residual
F = alpha * inner(grad(u), grad(v)) * dx + inner(psi, v) * dx
F += -alpha * inner(f, v) * dx - inner(psi_prev, v) * dx
F += inner(grad(T), grad(q)) * dx + beta * inner(T, q) * dx
F += -inner(g(exp(-psi)), q) * dx
F += inner(u, w) * dx + inner(exp(-psi), w) * dx
F += -inner(Phi0 + xi * T, w) * dx

# Create modified Jacobian
eps = dolfinx.fem.Constant(mesh, 1e-10)
F_modified = F + eps / alpha * inner(grad(psi), grad(w)) * dx
J = derivative(F_modified, s)

# Create boundary condition
mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
boundary_dofs = dolfinx.fem.locate_dofs_topological(
    V.sub(0), mesh.topology.dim - 1, boundary_facets
)
bc = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0.0), boundary_dofs, V.sub(0))

# Create data structures for computing H1 norm
u_diff = u - u_prev
u_diff_H1 = dolfinx.fem.form(inner(u_diff, u_diff) * dx + inner(grad(u_diff), grad(u_diff)) * dx)

# Create variables for outputting and files to output to
V0, sub0_to_parent = V.sub(0).collapse()
u_out = dolfinx.fem.Function(V0)
u_out.name = "u"
V1, sub1_to_parent = V.sub(1).collapse()
T_out = dolfinx.fem.Function(V1)
T_out.name = "T"
vtx_u = dolfinx.io.VTXWriter(mesh.comm, "u.bp", [u_out])
vtx_T = dolfinx.io.VTXWriter(mesh.comm, "T.bp", [T_out])


# Create nonlinear solver with PETSc SNES
termination_tol = 1e-9
max_lvpp_iterations = 100
sp = {
    "snes_type": "newtonls",
    "snes_monitor": None,
    "snes_linesearch_type": "bt",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "pc_svd_monitor": None,
    "mat_mumps_icntl_14": 1000,
    "snes_atol": 1e-6,
    "snes_rtol": 1e-6,
    "snes_stol": 10 * np.finfo(dolfinx.default_scalar_type).eps,
    "snes_linesearch_damping": 1e4,
    "snes_linesearch_order": 2,
    "snes_linesearch_monitor": None,
}
problem = SNESProblem(F, s, bcs=[bc])
solver = SNESSolver(problem, sp)

# Set initial guess for T
s.sub(1).interpolate(lambda x: np.ones_like(x[1]))

# LVPP Loop
num_iterations = []
for i in range(1, max_lvpp_iterations + 1):
    if mesh.comm.rank == 0:
        print(f"LVPP iteration: {i} Alpha: {float(alpha)}", flush=True)
    # Solve non-linear problem
    dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
    converged_reason, num_its = solver.solve()
    error_msg = f"Solver did not converge with {converged_reason}"
    assert converged_reason > 0, error_msg

    # Output
    u_out.x.array[:] = s.x.array[sub0_to_parent]
    T_out.x.array[:] = s.x.array[sub1_to_parent]
    vtx_u.write(i)
    vtx_T.write(i)

    # Check for convergence
    diff_local = dolfinx.fem.assemble_scalar(u_diff_H1)
    normed_diff = np.sqrt(mesh.comm.allreduce(diff_local, op=MPI.SUM))

    if mesh.comm.rank == 0:
        print(
            f"LVPP iteration {i}, Converged reason {converged_reason}",
            f" Newton iterations {num_its} ||u-u_prev||_L2={normed_diff}",
            flush=True,
        )
    num_iterations.append(num_its)
    if normed_diff < termination_tol:
        if mesh.comm.rank == 0:
            print(f"Solver converged after {i} iterations", flush=True)
        break
    # Update previous solution and alpha
    s_prev.x.array[:] = s.x.array
    alpha.value *= 4

if mesh.comm.rank == 0:
    print(f"Total number of LVPP iterations: {i}", flush=True)
    print(f"Total number of Newton iterations: {sum(num_iterations)}", flush=True)

vtx_u.close()
vtx_T.close()

# Store original mould
interpolation_points = V0.element.interpolation_points()
original_mould = dolfinx.fem.Function(V0)
original_mould.interpolate(dolfinx.fem.Expression(Phi0, interpolation_points))
original_mould.name = "OriginalMould"
mould = dolfinx.fem.Function(V0)
mould.interpolate(dolfinx.fem.Expression(Phi0 + xi * T, interpolation_points))
mould.name = "Mould"
with dolfinx.io.VTXWriter(mesh.comm, "mould.bp", [original_mould, mould]) as bp:
    bp.write(0.0)
