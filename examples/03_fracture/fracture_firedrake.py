# Fracture problem solved with the LVPP algorithm
# Author: Patrick E. Farrell
# SPDX-License-Identifier: MIT

from firedrake import *
from numpy import linspace
from firedrake import PETSc
from netgen.geom2d import CSG2d, Solid2d, EdgeInfo, Circle
import argparse

print = PETSc.Sys.Print

geo = CSG2d()
poly = Solid2d(
    [
        (0, 0),
        (2, 0),
        (2, 2),
        EdgeInfo(bc="plus"),
        (1.01, 2),
        (1, 1.5),
        (0.99, 2),
        EdgeInfo(bc="minus"),
        (0, 2),
    ]
)


disk = Circle((0.3, 0.3), 0.2)
geo.Add(poly - disk)
ngmesh = geo.GenerateMesh(maxh=0.05)

parser = argparse.ArgumentParser()
parser.add_argument("--nref", type=int, default=1)
args = parser.parse_args()

base = Mesh(ngmesh)
mh = MeshHierarchy(base, args.nref, netgen_flags={"degree": 1})
mesh = mh[-1]
V = FunctionSpace(mesh, "CG", 1)
C = FunctionSpace(mesh, "CG", 1)
L = FunctionSpace(mesh, "CG", 1)
Z = MixedFunctionSpace([V, C, L])

Gc = Constant(1)
G = Constant(1)

W = FunctionSpace(mesh, "DG", 0)
diams = project(CellDiameter(mesh), W)
with diams.dat.vec_ro as d:
    maxdiam = d.max()[1]
l = Constant(2 * maxdiam)
print(BLUE % f"Using l = {float(l)}")

z = Function(Z, name="Solution")
(u, c, psi) = split(z)
z.subfunctions[0].rename("Displacement")
z.subfunctions[1].rename("Damage")
z.subfunctions[2].rename("Latent")
z_test = TestFunction(Z)
(v, d, phi) = split(z_test)
z_trial = TrialFunction(Z)
(v_trial, d_trial, phi_trial) = split(z_trial)
z_prev = Function(Z, name="PreviousContinuationSolution")
(_, c_prev, _) = split(z_prev)
z_iter = Function(Z, name="PreviousLVPPSolution")
(_, c_iter, psi_iter) = split(z_iter)

c_conform = (c_prev + exp(psi)) / (exp(psi) + 1)
c_conform_ = Function(FunctionSpace(mesh, "CG", 3), name="ConformingDamage")

eps = Constant(1.0e-5)
E = (
    0.5 * G * ((1 - eps) * (1 - c) ** 2 + eps) * inner(grad(u), grad(u)) * dx
    + 0.5 * Gc / l * inner(c, c) * dx
    + 0.5 * Gc * l * inner(grad(c), grad(c)) * dx
)


alpha = Constant(1)
F = (
    alpha * derivative(E, z, z_test)
    + inner(psi, d) * dx
    - inner(psi_iter, d) * dx
    + inner(c, phi) * dx
    - inner(c_conform, phi) * dx
)

reps = Constant(1.0e-3)
J_reg = (
    derivative(F, z, z_trial)
    + reps * inner(v, v_trial) * dx
    + reps * inner(d, d_trial) * dx
    - reps * inner(phi, phi_trial) * dx
)

plus = [i + 1 for i, name in enumerate(ngmesh.GetRegionNames(codim=1)) if name in ["plus"]]
minus = [i + 1 for i, name in enumerate(ngmesh.GetRegionNames(codim=1)) if name in ["minus"]]
bcminus = Constant(0)
bcplus = Constant(0)
bcs = [DirichletBC(Z.sub(0), bcplus, plus), DirichletBC(Z.sub(0), bcminus, minus)]

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

pvd = VTKFile(f"output/nref-{args.nref}/lvpp.pvd")

NFAIL_MAX = 50

for step, T in enumerate(linspace(0, 5, 1001)[1:]):
    print(BLUE % f"Solving for T = {float(T)}", flush=True)
    bcminus.assign(-T)
    bcplus.assign(+T)

    alpha.assign(1)
    z_iter.assign(z)
    k = 1
    r = 2
    nfail = 0
    while True:
        try:
            print(BLUE % f"  Attempting {k=} α = {float(alpha)}", flush=True)
            problem = NonlinearVariationalProblem(F, z, bcs, J=J_reg)
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
        nrm = norm(c - c_iter, "L2")
        print(
            GREEN % f"  Solved {k=} α = {float(alpha)}. ||c_{k} - c_{k - 1}|| = {nrm}", flush=True
        )
        if nrm < 1.0e-4:
            break

        # Update alpha
        if solver.snes.getIterationNumber() <= 4:
            alpha.assign(alpha * r)
        elif solver.snes.getIterationNumber() >= 10:
            alpha.assign(alpha / r)

        # Update z_iter
        z_iter.assign(z)

        k += 1

    # When the object has broken (i.e. the crack has partitioned the domain),
    # the failure mode of the algorithm above is that it terminates in one
    # PG iteration that does no Newton iterations, so the solution doesn't
    # change
    if k == 1 and norm(z_prev - z) == 0.0:
        break

    if nfail == NFAIL_MAX:
        break
    c_conform_.interpolate(c_conform)

    if step % 10 == 0:
        pvd.write(z.subfunctions[0], z.subfunctions[1], z.subfunctions[2], c_conform_, time=T)

    z_prev.assign(z)
