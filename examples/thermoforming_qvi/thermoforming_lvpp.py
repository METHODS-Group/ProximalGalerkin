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
alpha = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(2 ** (-6)))
f = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(25))
Phi0 = 1 - 2 * max_value(abs(x - 0.5), abs(y - 0.5))
xi = sin(pi * x) * sin(pi * y)

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

mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
boundary_dofs = dolfinx.fem.locate_dofs_topological(
    V.sub(0), mesh.topology.dim - 1, boundary_facets
)
bc = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0.0), boundary_dofs, V.sub(0))


u_diff = u - u_prev
u_diff_H1 = dolfinx.fem.form(inner(u_diff, u_diff) * dx + inner(grad(u_diff), grad(u_diff)) * dx)
termination_tol = 1e-8
s.sub(1).interpolate(lambda x: np.ones_like(x[1]))

sp = {
    "snes_monitor": None,
    "snes_linesearch_type": "bt",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "pc_svd_monitor": None,
    "mat_mumps_icntl_14": 1000,
    "snes_atol": 1e-8,
    "snes_rtol": 1e-6,
    "snes_linesearch_damping": 0.5,
}

max_lvpp_iterations = 100
V0, sub0_to_parent = V.sub(0).collapse()
u_out = dolfinx.fem.Function(V0)
u_out.name = "u"
V1, sub1_to_parent = V.sub(1).collapse()
T_out = dolfinx.fem.Function(V1)
T_out.name = "T"

vtx_u = dolfinx.io.VTXWriter(mesh.comm, "u.bp", [u_out])
vtx_T = dolfinx.io.VTXWriter(mesh.comm, "T.bp", [T_out])

problem = SNESProblem(F, s, bcs=[bc])
solver = SNESSolver(problem, sp)

num_iterations = []
for i in range(1, max_lvpp_iterations + 1):
    print(f"LVPP iteration: {i} Alpha: {float(alpha)}")

    dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
    converged_reason, num_its = solver.solve()
    assert converged_reason > 0, f"Solver did not converge with {converged_reason}"

    # Output
    u_out.x.array[:] = s.x.array[sub0_to_parent]
    T_out.x.array[:] = s.x.array[sub1_to_parent]
    vtx_u.write(i)
    vtx_T.write(i)
    diff_local = dolfinx.fem.assemble_scalar(u_diff_H1)
    normed_diff = np.sqrt(mesh.comm.allreduce(diff_local, op=MPI.SUM))
    print(
        f"LVPP iteration {i}, Converged reason {converged_reason}",
        f" Newton iterations {num_its} ||u-u_prev||_L2={normed_diff}",
    )
    num_iterations.append(num_its)
    if normed_diff < termination_tol:
        print("Solver converged after {i} iterations")
        break
    s_prev.x.array[:] = s.x.array
    alpha.value *= 4

print("Total number of LVPP iterations:", i)
print("Total number of Newton iterations:", sum(num_iterations))

vtx_u.close()
vtx_T.close()

original_mould = dolfinx.fem.Function(V0)
original_mould.interpolate(dolfinx.fem.Expression(Phi0, V0.element.interpolation_points()))
original_mould.name = "OriginalMould"
mould = dolfinx.fem.Function(V0)
mould.interpolate(dolfinx.fem.Expression(Phi0 + xi * T, V0.element.interpolation_points()))
mould.name = "Mould"


with dolfinx.io.VTXWriter(mesh.comm, "original_mould.bp", [original_mould, mould]) as bp:
    bp.write(0.0)
exit()
