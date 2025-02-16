from mpi4py import MPI
import numpy as np
import dolfinx.fem.petsc, dolfinx.nls.petsc
import basix.ufl
from ufl import (
    inner,
    grad,
    dx,
    exp,
    split,
    TestFunctions,
    conditional,
    lt,
    sin,
    pi,
    SpatialCoordinate,
    max_value,
    derivative,
)
from lvpp import SNESProblem, SNESSolver

M = 150
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, M, M)
el = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
me = basix.ufl.mixed_element([el, el, el])
V = dolfinx.fem.functionspace(mesh, me)
s = dolfinx.fem.Function(V)
u, T, psi = split(s)
v, q, w = TestFunctions(V)


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


# Setup problem formulation
x, y = SpatialCoordinate(mesh)
s_prev = dolfinx.fem.Function(V)
u_prev, _, psi_prev = split(s_prev)
beta = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0))
alpha = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(2e-6))
f = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(25))
Phi0 = 1 - 2 * max_value(abs(x - 0.5), abs(y - 0.5))
xi = sin(pi * x) * sin(pi * y)

F = alpha * inner(grad(u), grad(v)) * dx + inner(psi, v) * dx
F += -alpha * inner(f, v) * dx - inner(psi_prev, v) * dx
F += inner(grad(T), grad(q)) * dx + beta * inner(T, q) * dx
F += -inner(g(exp(-psi)), q) * dx
F += inner(u, w) * dx + inner(exp(-psi), w) * dx
F += -inner(Phi0 + xi * T, w) * dx

# out = dolfinx.fem.Function(V.sub(0).collapse()[0])
# out.interpolate(dolfinx.fem.Expression(Phi0, V.sub(0).collapse()[0].element.interpolation_points()))
# with dolfinx.io.VTXWriter(mesh.comm, "phi0.bp", [out]) as bp:
#     bp.write(0.0)
# exit()

# Create modified Jacobian
eps = dolfinx.fem.Constant(mesh, 1e-10)
F_modified = F + eps / alpha * inner(grad(psi), grad(w)) * dx
J = derivative(F_modified, s)

mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
boundary_dofs = dolfinx.fem.locate_dofs_topological(
    V.sub(0), mesh.topology.dim - 1, boundary_facets
)
bc = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0.0), boundary_dofs, V.sub(0))


u_diff_L2 = dolfinx.fem.form(inner(u - u_prev, u - u_prev) * dx)
termination_tol = 1e-9
s.sub(1).interpolate(lambda x: np.ones_like(x[1]))

sp = {
    "snes_monitor": None,
    "snes_linesearch_type": "l2",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "pc_svd_monitor": None,
    "mat_mumps_icntl_14": 1000,
}
problem = SNESProblem(F, s, bcs=[bc])
solver = SNESSolver(problem, sp)

max_lvpp_iterations = 100
V0, sub_to_parent = V.sub(0).collapse()
u_out = dolfinx.fem.Function(V0)
vtx = dolfinx.io.VTXWriter(MPI.COMM_WORLD, "output.bp", [u_out])
for i in range(1, max_lvpp_iterations + 1):
    dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
    converged, num_iterations = solver.solve()
    u_out.x.array[:] = s.x.array[sub_to_parent]
    vtx.write(i)
    if i == 3:
        break
    diff_local = dolfinx.fem.assemble_scalar(u_diff_L2)
    normed_diff = np.sqrt(mesh.comm.allreduce(diff_local, op=MPI.SUM))
    print(i, converged, num_iterations, normed_diff)
    if normed_diff < termination_tol:
        break
    s_prev.x.array[:] = s.x.array
    alpha.value *= 4
