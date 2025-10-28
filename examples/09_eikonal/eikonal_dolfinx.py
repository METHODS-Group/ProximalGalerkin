import argparse
from pathlib import Path

from mpi4py import MPI

import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import numpy as np
import scifem
import ufl
from dolfinx.fem.petsc import NonlinearProblem
from read_mobius_dolfinx import read_mobius_strip

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mesh-dir",
    dest="mesh_dir",
    type=Path,
    default="mobius-strip.mesh",
    help="Path to input MFEM mesh directory",
)

args = parser.parse_args()
mesh_dir = args.mesh_dir
# Mesh generated with MFEM, see: convert_mesh.cpp for instructions
mesh = read_mobius_strip(mesh_dir / "Cycle000000/proc000000.vtu")

el_0 = basix.ufl.element("P", mesh.topology.cell_name(), 1)
el_1 = basix.ufl.element("P", mesh.topology.cell_name(), 2, shape=(3,))
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

sp = {
    "snes_monitor": None,
    "snes_linesearch_type": "l2",
    "snes_rtol": tol,
    "snes_atol": tol,
    "snes_stol": tol,
    "snes_max_it": 100,
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_14": 500,
    "snes_error_if_not_converged": True,
    "ksp_type": "preonly",
    "pc_type": "lu",
}
problem = NonlinearProblem(
    F, u=w, bcs=[], J=J, petsc_options=sp, petsc_options_prefix="snes_"
)


dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
V_out = dolfinx.fem.functionspace(mesh, ("DG", mesh.geometry.cmap.degree))
u_out = dolfinx.fem.Function(V_out)
u_out.name = "u"
bp_u = dolfinx.io.VTXWriter(mesh.comm, "u.bp", [u_out], engine="BP4")
diff = w.sub(0) - w0.sub(0)
L2_squared = ufl.dot(diff, diff) * dx
compiled_diff = dolfinx.fem.form(L2_squared)


nh = ufl.FacetNormal(mesh)
mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
num_facets = mesh.topology.index_map(mesh.topology.dim - 1).size_local
submesh, entity_map, _, _ = dolfinx.mesh.create_submesh(
    mesh, mesh.topology.dim - 1, np.arange(num_facets, dtype=np.int32)
)
q_el = basix.ufl.quadrature_element(submesh.basix_cell(), nh.ufl_shape, "default", 1)
Q = dolfinx.fem.functionspace(submesh, q_el)
expr = dolfinx.fem.Expression(
    nh, Q.element.interpolation_points, dtype=dolfinx.default_scalar_type
)
mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
f_to_c = mesh.topology.connectivity(mesh.topology.dim - 1, mesh.topology.dim)
mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
c_to_f = mesh.topology.connectivity(mesh.topology.dim, mesh.topology.dim - 1)
ie = []
parent_facets = entity_map.sub_topology_to_topology(
    np.arange(
        submesh.topology.index_map(submesh.topology.dim).size_local
        + submesh.topology.index_map(submesh.topology.dim).num_ghosts,
        dtype=np.int32,
    ),
    inverse=False,
)
ie = np.full((len(parent_facets), 2), -1, dtype=np.int32)
for i, facet in enumerate(parent_facets):
    cells = f_to_c.links(facet)
    if len(cells) > 1:
        cell = f_to_c.links(facet)[1]
    else:
        cell = f_to_c.links(facet)[0]
    facets = c_to_f.links(cell)
    local_index = np.flatnonzero(facets == facet)[0]
    ie[i, 0] = cell
    ie[i, 1] = local_index

values = expr.eval(mesh, np.asarray(ie, dtype=np.int32))
qq = dolfinx.fem.Function(Q)
qq.x.array[:] = values.flatten()

scifem.xdmf.create_pointcloud("data.xdmf", [qq])
Q_out = dolfinx.fem.functionspace(
    mesh, ("DG", mesh.geometry.cmap.degree, (mesh.geometry.dim,))
)
q_out = dolfinx.fem.Function(Q_out)
q_out.name = "grad(u)"
q_expr = dolfinx.fem.Expression(ufl.grad(w.sub(0)), Q_out.element.interpolation_points)

psi_out = dolfinx.fem.Function(Q_out)
psi_out.name = "psi"
vtx_psi = dolfinx.io.VTXWriter(
    mesh.comm,
    "psi.bp",
    [q_out, psi_out],
    engine="BP5",
)
try:
    newton_iterations = []
    for i in range(1, 100):
        alpha.value = min(2**i, 10)

        problem.solve()
        num_newton_iterations = problem.solver.getIterationNumber()
        converged = problem.solver.getConvergedReason() > 0
        ksp_converged = problem.solver.ksp.getConvergedReason()
        newton_iterations.append(num_newton_iterations)
        print(f"Iteration {i}: {converged=} {num_newton_iterations=} {ksp_converged=}")
        local_diff = dolfinx.fem.assemble_scalar(compiled_diff)
        global_diff = np.sqrt(mesh.comm.allreduce(local_diff, op=MPI.SUM))
        print(f"|delta u |= {global_diff}")
        w0.x.array[:] = w.x.array

        dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)

        u_out.interpolate(w.sub(0))
        bp_u.write(i)
        q_out.interpolate(q_expr)
        psi_out.interpolate(w.sub(1))
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
