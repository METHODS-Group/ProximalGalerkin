# Solving a problem with multiple intersecting inequality constraints using the LVPP algorithm.
# Author: Patrick E. Farrell
# SPDX-License-Identifier: MIT
from firedrake import *
import matplotlib.pyplot as plt
import firedrake.pyplot as fplt
from firedrake import PETSc
import numpy as np
from pathlib import Path

print = PETSc.Sys.Print

base = UnitIntervalMesh(1001)
mh = MeshHierarchy(base, 0)
mesh = mh[-1]
V = FunctionSpace(mesh, "CG", 1)
C = FunctionSpace(mesh, "CG", 1)
L = VectorFunctionSpace(mesh, "CG", 1)
Z = MixedFunctionSpace([V, C, L])

z = Function(Z, name="Solution")
(u, psi0, psi) = split(z)
z.subfunctions[0].rename("Displacement")
z.subfunctions[1].rename("LatentObstacle")
z.subfunctions[2].rename("LatentGradient")
z_test = TestFunction(Z)
(v, w0, w) = split(z_test)

z_prev = Function(Z, name="PreviousContinuationSolution")
(_, psi0_prev, psi_prev) = split(z_prev)
z_iter = Function(Z, name="PreviousLVPPSolution")
(u_iter, psi0_iter, psi_iter) = split(z_iter)

c = Constant(0)
E = 0.5 * inner(grad(u), grad(u)) * dx + c * u * dx

x = SpatialCoordinate(mesh)[0]

# Nonsmooth obstacle
phi0 = conditional(le(x, 0.4), 0, conditional(ge(x, 0.6), 0, 1))
# Smooth obstacle
(l, r) = (0.2, 0.8)
bump = exp(-1 / (10 * (x - l) * (r - x))) / exp(-1 / (10 * (0.5 - l) * (r - 0.5)))
phi0 = conditional(le(x, l), 0, conditional(ge(x, r), 0, bump))

phic = Constant(100)
phi = conditional(le(x, 0.2), phic, conditional(gt(x, 0.8), phic, 100))

alpha = Constant(1)
F = (
    alpha * derivative(E, z, z_test)
    + inner(psi0, v) * dx
    + inner(psi, grad(v)) * dx
    - inner(psi0_iter, v) * dx
    - inner(psi_iter, grad(v)) * dx
    + inner(u, w0) * dx
    - inner(exp(psi0), w0) * dx
    - inner(phi0, w0) * dx
    + inner(grad(u), w) * dx
    - inner(phi * psi / sqrt(1 + dot(psi, psi)), w) * dx
)

bcs = [DirichletBC(Z.sub(0), 0, "on_boundary")]
# bcs = None

sp = {
    "snes_monitor": None,
    "snes_converged_reason": None,
    "snes_linesearch_type": "l2",
    "snes_linesearch_maxstep": 1,
    "snes_atol": 1.0e-6,
    "snes_linesearch_monitor": None,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_14": 500,
}


plt.figure(figsize=(16, 8), dpi=300)
ax = plt.gca()
NFAIL_MAX = 50

for phi_ in [3, 2, 1, 0.5, 0.1, 0.01]:
    phic.assign(phi_)
    print(BLUE % f"Solving for phi = {float(phic)}", flush=True)

    alpha.assign(1)
    z_iter.assign(z)
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
        nrm = norm(u - u_iter, "L2")
        print(
            GREEN % f"  Solved {k=} α = {float(alpha)}. ||u_{k} - u_{k - 1}|| = {nrm}", flush=True
        )
        if nrm < 1.0e-12:
            break

        # Update alpha
        if solver.snes.getIterationNumber() <= 4:
            alpha.assign(alpha * r)
        elif solver.snes.getIterationNumber() >= 10:
            alpha.assign(alpha / r)

        # Update z_iter
        z_iter.assign(z)

        k += 1

    # Check for failure
    if k == 1 and norm(z_prev - z) == 0.0:
        break

    if nfail == NFAIL_MAX:
        break

    # fplt.plot(z.subfunctions[0], label=f"φ = {phi_}", axes=ax)
    u_conform = Function(V)
    u_conform.interpolate(exp(z.subfunctions[1]) + phi0)
    fplt.plot(u_conform, label=f"φ = {phi_}", axes=ax)
    u = z.subfunctions[0]
    with open(f"output/phi-{phi_}.txt", "w") as f:
        for x_ in np.linspace(0, 1, 1001):
            f.write(f"{x_} {u_conform.at(x_)}\n")
            # f.write(f"{x_} {u.at(x_)}\n")
    z_prev.assign(z)

phi0_plot = Function(V)
phi0_plot.interpolate(phi0)
folder = Path("output")
folder.mkdir(exist_ok=True)

with open(folder / "obstacle.txt", "w") as f:
    for x_ in np.linspace(0, 1, 1001):
        f.write(f"{x_} {phi0_plot.at(x_)}\n")
fplt.plot(phi0_plot, label="φ₀", axes=ax)
plt.legend()
# plt.show()
plt.savefig(folder / "intersecting-constraints.png")
