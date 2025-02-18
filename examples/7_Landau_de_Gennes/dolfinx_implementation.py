from mpi4py import MPI
from ufl_expressions import expm2
from lvpp import SNESProblem, SNESSolver
import numpy as np
import dolfinx
from ufl import (
    inv,
    Identity,
    split,
    as_tensor,
    TestFunction,
    inner,
    grad,
    Measure,
    tr,
    derivative,
)
import basix.ufl

import numpy.linalg


class NotConvergedError(Exception):
    pass


def tanh(M):
    return 2 * inv(expm2(2 * M) + I) * (expm2(2 * M) - I)


class Constant(dolfinx.fem.Constant):
    def __init__(self, mesh, value):
        super().__init__(mesh, dolfinx.default_scalar_type(value))


I = Identity(2)

N = 100
mesh = dolfinx.mesh.create_unit_square(
    MPI.COMM_WORLD, N, N, cell_type=dolfinx.mesh.CellType.quadrilateral
)

p = 3
el_V = basix.ufl.element("Lagrange", mesh.basix_cell(), p)
el_L = basix.ufl.element("Lagrange", mesh.basix_cell(), p)
me = basix.ufl.mixed_element([el_V, el_V, el_L, el_L])
Z = dolfinx.fem.functionspace(mesh, me)
z = dolfinx.fem.Function(Z, name="Solution")
(q1, q2, psi1, psi2) = split(z)
Q = as_tensor([[q1, q2], [q2, -q1]])
Psi = as_tensor([[psi1, psi2], [psi2, -psi1]])

z_test = TestFunction(Z)
(w1, w2, phi1, phi2) = split(z_test)
W = as_tensor([[w1, w2], [w2, -w1]])
Phi = as_tensor([[phi1, phi2], [phi2, -phi1]])
z_prev = dolfinx.fem.Function(Z, name="PreviousContinuationSolution")
(q1_prev, q2_prev, _, _) = split(z_prev)
z_iter = dolfinx.fem.Function(Z, name="PreviousLVPPSolution")
(q1_iter, q2_iter, psi1_iter, psi2_iter) = split(z_iter)
Q_iter = as_tensor([[q1_iter, q2_iter], [q2_iter, -q1_iter]])
Psi_iter = as_tensor([[psi1_iter, psi2_iter], [psi2_iter, -psi1_iter]])

A = Constant(mesh, 1.0)
B = Constant(mesh, 11.0)
C = Constant(mesh, 4.0)
dx = Measure("dx", domain=mesh, metadata={"quadrature_degree": 20})
E = (
    1 / 2 * inner(grad(Q), grad(Q)) * dx
    + 1 / 2 * A * tr(Q * Q) * dx
    + 1 / 4 * C * (tr(Q * Q) ** 2) * dx
)

alpha = Constant(mesh, 1)
F = (
    alpha * derivative(E, z, z_test)
    + inner(Psi, W) * dx
    - inner(Psi_iter, W) * dx
    + inner(Q, Phi) * dx
    - inner(0.5 * tanh(Psi / 2), Phi) * dx
)
cffi_options = ["-Ofast", "-march=native"]
jit_options = {
    "cffi_extra_compile_args": cffi_options,
    "cffi_libraries": ["m"],
}
F_compiled = dolfinx.fem.form(F, jit_options=jit_options)
J_compiled = dolfinx.fem.form(derivative(F, z), jit_options=jit_options)

# Set up for boundary conditions from Robinson et al. (2017)
d = 0.06
theta_tb = 0
theta_lr = np.pi / 2


def s(x, y):
    top_bottom = np.isclose(y, 0) | np.isclose(y, 1)
    left_right = np.isclose(x, 0) | np.isclose(x, 1)
    return T(y) * left_right + T(x) * top_bottom


def T(z, eps=np.finfo(dolfinx.default_scalar_type).eps):
    """Compute ramp function for boundary conditions in a vectorized fashion"""
    assert ((0 <= z + eps) & (z - eps <= 1)).all(), (
        "Invalid range on variable expected to be in [0, 1]"
    )
    interval1 = (0 <= z + eps) & (z - eps < d)
    interval3 = (1 - d <= z + eps) & (z - eps <= 1)
    interval2 = np.invert(interval1) & np.invert(interval3)
    return interval1 * z / d + 1 * interval2 + (1 - z) / d * interval3


def theta(x):
    top_bottom = np.isclose(x[1], 0) | np.isclose(x[1], 1)
    left_right = np.isclose(x[0], 0) | np.isclose(x[0], 1)
    return theta_lr * left_right + theta_tb * top_bottom


def g_xx(x):
    tht = theta(x)
    return 1 / 2 * s(x[0], x[1]) * np.cos(2 * tht)


def g_xy(x):
    tht = theta(x)
    return 1 / 2 * s(x[0], x[1]) * np.sin(2 * tht)


tdim = mesh.topology.dim
mesh.topology.create_connectivity(tdim - 1, tdim)
exterior_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
V0, V0_to_Z = Z.sub(0).collapse()
u_bc0 = dolfinx.fem.Function(V0)
u_bc0.interpolate(g_xx)

dofs0 = dolfinx.fem.locate_dofs_topological((Z.sub(0), V0), tdim - 1, exterior_facets)
bc0 = dolfinx.fem.dirichletbc(u_bc0, dofs0, Z.sub(0))


V1, _ = Z.sub(1).collapse()
dofs1 = dolfinx.fem.locate_dofs_topological((Z.sub(1), V1), tdim - 1, exterior_facets)
u_bc1 = dolfinx.fem.Function(V1)
u_bc1.interpolate(g_xy)
bc1 = dolfinx.fem.dirichletbc(u_bc1, dofs1, Z.sub(1))
bcs = [bc0, bc1]

sp = {"snes_monitor": None, "snes_linesearch_type": "l2", "snes_linesearch_monitor": None}


Q_space = dolfinx.fem.functionspace(mesh, ("Lagrange", p, Q.ufl_shape))
Q_points = Q_space.element.interpolation_points()
q_out = dolfinx.fem.Function(Q_space, name="Q-tensor")
expr = dolfinx.fem.Expression(Q, Q_points)
vtx = dolfinx.io.VTXWriter(mesh.comm, "Q.bp", [q_out])

L2_Q = dolfinx.fem.form(inner(Q - Q_iter, Q - Q_iter) * dx, jit_options=jit_options)
problem = SNESProblem(F_compiled, z, bcs=bcs, J=J_compiled)

NFAIL_MAX = 50
NLVVP_MAX = 100
r = 2
nfail = 0
nlvpp = 0
num_newton_iterations = []
while nfail < NFAIL_MAX and nlvpp < NLVVP_MAX:
    if mesh.comm.rank == 0:
        print(f"Attempting {nlvpp=} alpha={float(alpha)}", flush=True)
    try:
        solver = SNESSolver(problem, sp)
        converged_reason, num_iterations = solver.solve()
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
            print(f"Failed to converge, {nlvpp=} alpha={float(alpha)}", flush=True)
        alpha.value /= 2
        if nlvpp == 0:
            z.interpolate(z_prev)
        else:
            z.interpolate(z_iter)

        if nfail >= NFAIL_MAX:
            if mesh.comm.rank == 0:
                print(f"Giving up. {T=} alpha={float(alpha)} {nlvpp=}", flush=True)
                break
        else:
            continue
    num_newton_iterations.append(num_iterations)
    # Termination
    nrm = np.sqrt(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(L2_Q), op=MPI.SUM))
    if mesh.comm.rank == 0:
        print(
            f"Solved {nlvpp=} alpha={float(alpha)} ||Q_{nlvpp} - Q_{nlvpp - 1}|| = {nrm}",
            flush=True,
        )
    if nrm < 1.0e-10:
        break

    # Update alpha
    if num_iterations <= 4:
        alpha.value *= r
    elif num_iterations >= 10:
        alpha.value /= r

    # Update z_iter
    z_iter.interpolate(z)
    q_out.interpolate(expr)
    vtx.write(nlvpp)
    nlvpp += 1

vtx.close()
if mesh.comm.rank == 0:
    assert nlvpp == len(num_newton_iterations)
    print(
        f"#LVPP iterations {len(num_newton_iterations)}, #Newton iterations {sum(num_newton_iterations)}",
        flush=True,
    )
    print(
        f"Min/Max Newton iterations {min(num_newton_iterations)}/{max(num_newton_iterations)}",
        flush=True,
    )

latent = dolfinx.fem.Function(Q_space, name="LatentVariable")
latent.interpolate(dolfinx.fem.Expression(Psi, Q_points))

conforming = dolfinx.fem.Function(Q_space, name="ConformingApproximation")
conforming.interpolate(dolfinx.fem.Expression(0.5 * tanh(Psi / 2), Q_points))

diff = dolfinx.fem.Function(Q_space, name="DifferenceBetweenPrimalAndConforming")
diff.x.array[:] = q_out.x.array - conforming.x.array

m_plus = dolfinx.fem.Function(V0, name="MaximumEigenvalue")
m_minus = dolfinx.fem.Function(V0, name="MinimumEigenvalue")
value_size = np.prod(Q.ufl_shape)
for i in range(len(m_plus.x.array)):
    eigenvalues = numpy.linalg.eigvals(
        q_out.x.array[value_size * i : value_size * (i + 1)].reshape(Q.ufl_shape)
    )
    m_plus.x.array[i] = eigenvalues.max()
    m_minus.x.array[i] = eigenvalues.min()

with dolfinx.io.VTXWriter(mesh.comm, "t.bp", [q_out, latent, conforming]) as vtx:
    vtx.write(0.0)

with dolfinx.io.VTXWriter(mesh.comm, "m_plus.bp", [m_plus, m_minus]) as vtx:
    vtx.write(0.0)
