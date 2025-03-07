# Solving a problem with constrained eigenvalues using the LVPP algorithm.
# Author: Patrick E. Farrell
# SPDX-License-Identifier: MIT

from firedrake import *
from ufl_expressions import expm2 as expm
import numpy.linalg
from pathlib import Path

I = Identity(2)
tanh = lambda M: 2 * inv(expm(2 * M) + I) * (expm(2 * M) - I)

L = 1
mesh = SquareMesh(100, 100, L, quadrilateral=True)

p = 3
V = FunctionSpace(mesh, "CG", p)
L = FunctionSpace(mesh, "CG", p)
Z = MixedFunctionSpace([V, V, L, L])

z = Function(Z, name="Solution")
(q1, q2, psi1, psi2) = split(z)
Q = as_tensor([[q1, q2], [q2, -q1]])
Psi = as_tensor([[psi1, psi2], [psi2, -psi1]])

z_test = TestFunction(Z)
(w1, w2, phi1, phi2) = split(z_test)
W = as_tensor([[w1, w2], [w2, -w1]])
Phi = as_tensor([[phi1, phi2], [phi2, -phi1]])
z_prev = Function(Z, name="PreviousContinuationSolution")
(q1_prev, q2_prev, _, _) = split(z_prev)
z_iter = Function(Z, name="PreviousLVPPSolution")
(q1_iter, q2_iter, psi1_iter, psi2_iter) = split(z_iter)
Q_iter = as_tensor([[q1_iter, q2_iter], [q2_iter, -q1_iter]])
Psi_iter = as_tensor([[psi1_iter, psi2_iter], [psi2_iter, -psi1_iter]])

A = Constant(1.0)
B = Constant(11.0)
C = Constant(4.0)

E = (
    1 / 2 * inner(grad(Q), grad(Q)) * dx
    + 1 / 2 * A * tr(Q * Q) * dx
    + 1 / 4 * C * (tr(Q * Q) ** 2) * dx
)

alpha = Constant(1)
F = (
    alpha * derivative(E, z, z_test)
    # + inner(psi1, w1)*dx
    # - inner(psi1_iter, w1)*dx
    # + inner(psi2, w2)*dx
    # - inner(psi2_iter, w2)*dx
    + inner(Psi, W) * dx
    - inner(Psi_iter, W) * dx
    + inner(Q, Phi) * dx
    - inner(0.5 * tanh(Psi / 2), Phi) * dx
)

# Set up for boundary conditions from Robinson et al. (2017)
(x, y) = SpatialCoordinate(mesh)
d = Constant(0.06)
trap = lambda t: conditional(
    lt(t, d), t / d, conditional(lt(t, 1 - d), 1, conditional(le(t, 1), (1 - t) / d, 0))
)
s = conditional(Or(eq(x, 0), eq(x, 1)), trap(y), conditional(Or(eq(y, 0), eq(y, 1)), trap(x), 0))

# top/bottom and left/right
theta_tb = 0
theta_lr = pi / 2
tb = (1, 2)
lr = (3, 4)
bcs = [
    DirichletBC(Z.sub(0), (1 / 2) * s * cos(2 * theta_tb), tb),
    DirichletBC(Z.sub(1), (1 / 2) * s * sin(2 * theta_tb), tb),
    DirichletBC(Z.sub(0), (1 / 2) * s * cos(2 * theta_lr), lr),
    DirichletBC(Z.sub(1), (1 / 2) * s * sin(2 * theta_lr), lr),
]

sp = {"snes_monitor": None, "snes_linesearch_type": "l2", "snes_linesearch_monitor": None}

k = 1
r = 2
nfail = 0
while True:
    try:
        print(BLUE % f"  Attempting {k=} α = {float(alpha)}", flush=True)
        problem = NonlinearVariationalProblem(F, z, bcs)
        solver = NonlinearVariationalSolver(problem, solver_parameters=sp)
        solver.solve()

        if solver.snes.getIterationNumber() == 0:
            # solver didn't actually get to do any work,
            # we've just reduced alpha so much that the initial guess
            # satisfies the PDE
            raise ConvergenceError
    except ConvergenceError:
        nfail += 1
        print(RED % f"  Failed {k=} α = {float(alpha)}.", flush=True)
        alpha.assign(alpha / 2)

        if k == 1:
            z.assign(z_prev)
        else:
            z.assign(z_iter)

        if nfail >= NFAIL_MAX:
            print(RED % f"  Giving up.", flush=True)
            break
        else:
            continue

    # Termination
    nrm = norm(Q - Q_iter, "L2")
    print(GREEN % f"  Solved {k=} α = {float(alpha)}. ||Q_{k} - Q_{k - 1}|| = {nrm}", flush=True)
    if nrm < 1.0e-10:
        break

    # Update alpha
    if solver.snes.getIterationNumber() <= 4:
        alpha.assign(alpha * r)
    elif solver.snes.getIterationNumber() >= 10:
        alpha.assign(alpha / r)

    # Update z_iter
    z_iter.assign(z)

    k += 1

T = TensorFunctionSpace(mesh, "CG", p)
t = Function(T, name="Q-tensor")
t.interpolate(Q)

latent = Function(T, name="LatentVariable")
latent.interpolate(Psi)

conforming = Function(T, name="ConformingApproximation")
conforming.interpolate(0.5 * tanh(Psi / 2))

diff = Function(T, name="DifferenceBetweenPrimalAndConforming")
diff.interpolate(t - conforming)

m_plus = Function(V, name="MaximumEigenvalue")
for i in range(t.dat.data.shape[0]):
    m_plus.dat.data[i] = numpy.linalg.eigvals(t.dat.data[i, :, :]).max()
m_minus = Function(V, name="MinimumEigenvalue")
for i in range(t.dat.data.shape[0]):
    m_minus.dat.data[i] = numpy.linalg.eigvals(t.dat.data[i, :, :]).min()

folder = Path("output")
folder.mkdir(exist_ok=True)
VTKFile(folder / "solution.pvd").write(t, latent, conforming, diff, m_plus, m_minus)
