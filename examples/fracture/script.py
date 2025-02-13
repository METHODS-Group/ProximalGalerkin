from mpi4py import MPI
from petsc4py import PETSc
import basix.ufl
from lvpp import SNESProblem
import dolfinx.fem.petsc
from ufl import (
    TestFunction,
    TrialFunction,
    split,
    inner,
    grad,
    dx,
    exp,
    Circumradius,
    derivative,
)
import numpy as np

from generate_mesh import create_crack_mesh


class NotConvergedError(Exception):
    pass


st = dolfinx.default_scalar_type
mesh, ft, _ = create_crack_mesh(MPI.COMM_WORLD, max_res=0.025)
el = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
el = basix.ufl.mixed_element([el, el, el])
Z = dolfinx.fem.functionspace(mesh, el)


G = dolfinx.fem.Constant(mesh, st(1.0))
Gc = dolfinx.fem.Constant(mesh, st(1.0))

# Compute maximum cell size
W = dolfinx.fem.functionspace(mesh, ("DG", 0))
h = dolfinx.fem.Expression(4 * Circumradius(mesh), W.element.interpolation_points())
diam = dolfinx.fem.Function(W)
diam.interpolate(h)
max_diam = mesh.comm.allreduce(np.max(diam.x.array), op=MPI.MAX)
l = dolfinx.fem.Constant(mesh, st(max_diam))
print(f"Using l = {float(l)}")


z = dolfinx.fem.Function(Z)
(u, c, psi) = split(z)

z_test = TestFunction(Z)
(v, d, phi) = split(z_test)
z_trial = TrialFunction(Z)
(v_trial, d_trial, phi_trial) = split(z_trial)

z_prev = dolfinx.fem.Function(Z)
_, c_prev, _ = split(z_prev)
z_iter = dolfinx.fem.Function(Z)
(_, c_iter, psi_iter) = split(z_iter)


output_space = dolfinx.fem.functionspace(mesh, ("Lagrange", 3))
c_conform_out = dolfinx.fem.Function(output_space, name="ConformingDamage")
alpha = dolfinx.fem.Constant(mesh, st(1.0))
c_conform = (c_prev + exp(psi)) / (exp(psi) + 1)
c_conform_expr = dolfinx.fem.Expression(
    c_conform, output_space.element.interpolation_points()
)

eps = dolfinx.fem.Constant(mesh, 1.0e-5)
E = (
    0.5 * G * ((1 - eps) * (1 - c) ** 2 + eps) * inner(grad(u), grad(u)) * dx
    + 0.5 * Gc / l * inner(c, c) * dx
    + 0.5 * Gc * l * inner(grad(c), grad(c)) * dx
)

F = (
    alpha * derivative(E, z, z_test)
    + inner(psi, d) * dx
    - inner(psi_iter, d) * dx
    + inner(c, phi) * dx
    - inner(c_conform, phi) * dx
)
F_compiled = dolfinx.fem.form(F)

reps = dolfinx.fem.Constant(mesh, 1.0e-3)
J_reg = dolfinx.fem.form(
    derivative(F, z, z_trial)
    + reps * inner(v, v_trial) * dx
    + reps * inner(d, d_trial) * dx
    - reps * inner(phi, phi_trial) * dx
)

# Right side of crack (4), left crack (7)
bcminus = dolfinx.fem.Constant(mesh, 0.0)
bcplus = dolfinx.fem.Constant(mesh, 0.0)
mesh.topology.create_connectivity(1, 2)
left_dofs = dolfinx.fem.locate_dofs_topological(Z.sub(0), ft.dim, ft.find(7))
right_dofs = dolfinx.fem.locate_dofs_topological(Z.sub(0), ft.dim, ft.find(4))
bcs = [
    dolfinx.fem.dirichletbc(bcplus, right_dofs, Z.sub(0)),
    dolfinx.fem.dirichletbc(bcminus, left_dofs, Z.sub(0)),
]


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
NFAIL_MAX = 50
A = dolfinx.fem.petsc.create_matrix(J_reg)
b = dolfinx.fem.Function(Z)
b_vec = b.x.petsc_vec
x = dolfinx.fem.Function(Z)
problem = SNESProblem(F_compiled, z, bcs=bcs, J=J_reg)

xdmf_file = dolfinx.io.XDMFFile(mesh.comm, "solution.xdmf", "w")
xdmf_file.write_mesh(mesh)
xdmf_file.close()
vtx_damage = dolfinx.io.VTXWriter(mesh.comm, "damage.bp", [c_conform_out])
L2_c = dolfinx.fem.form(inner(c - c_iter, c - c_iter) * dx)
L2_z = dolfinx.fem.form(inner(z - z_prev, z - z_prev) * dx)

U, U_to_Z = Z.sub(0).collapse()
u_out = dolfinx.fem.Function(U, name="u")
C, C_to_Z = Z.sub(1).collapse()
c_out = dolfinx.fem.Function(C, name="c")
Psi, Psi_to_Z = Z.sub(2).collapse()
psi_out = dolfinx.fem.Function(Psi, name="psi")

for step, T in enumerate(np.linspace(0, 5, 101)[1:]):
    if mesh.comm.rank == 0:
        print(f"Solving for T = {float(T)}", flush=True)
    bcminus.value = -T
    bcplus.value = T
    alpha.value = 1
    z_iter.interpolate(z)
    k = 1
    r = 2
    nfail = 0
    while nfail <= NFAIL_MAX:
        try:
            if mesh.comm.rank == 0:
                print(f"Attempting {k=} alpha={float(alpha)}", flush=True)
            snes = PETSc.SNES().create(comm=mesh.comm)  # type: ignore
            opts = PETSc.Options()  # type: ignore
            snes.setOptionsPrefix("snes_solve")
            option_prefix = snes.getOptionsPrefix()
            opts.prefixPush(option_prefix)
            for key, v in sp.items():
                opts[key] = v
            opts.prefixPop()
            snes.setFromOptions()

            # Set solve functions and variable bounds
            snes.setFunction(problem.F, b_vec)
            snes.setJacobian(problem.J, A)
            snes.solve(None, x.x.petsc_vec)
            x.x.scatter_forward()
            z.interpolate(x)
            if snes.getConvergedReason() < 0:
                raise NotConvergedError("Not converged")
            if snes.getIterationNumber() == 0:
                # solver didn't actually get to do any work,
                # we've just reduced alpha so much that the initial guess
                # satisfies the PDE
                raise NotConvergedError("Not converged")
        except NotConvergedError:
            nfail += 1
            if mesh.comm.rank == 0:
                print(f"Failed to converge, {k=} alpha={float(alpha)}", flush=True)
            alpha.value /= 2
            if k == 1:
                z.interpolate(z_prev)
            else:
                z.interpolate(z_iter)

            if nfail >= NFAIL_MAX:
                if mesh.comm.rank == 0:
                    print(f"Giving up. {T=} alpha={float(alpha)} {k=}", flush=True)
                break
            else:
                continue
        nrm = np.sqrt(
            mesh.comm.allreduce(dolfinx.fem.assemble_scalar(L2_c), op=MPI.SUM)
        )
        if mesh.comm.rank == 0:
            print(
                f"Solved {k=} alpha={float(alpha)} ||c_{k} - c_{k - 1}|| = {nrm}",
                flush=True,
            )
        if nrm < 1.0e-4:
            break

        # Update alpha
        if snes.getIterationNumber() <= 4:
            alpha.value *= r
        elif snes.getIterationNumber() >= 10:
            alpha.value /= r

        # Update z_iter
        z_iter.interpolate(z)

        k += 1

    # When the object has broken (i.e. the crack has partitioned the domain),
    # the failure mode of the algorithm above is that it terminates in one
    # PG iteration that does no Newton iterations, so the solution doesn't
    # change
    norm_Z = np.sqrt(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(L2_z), op=MPI.SUM))
    if k == 1 and np.isclose(norm_Z, 0.0):
        break

    if nfail == NFAIL_MAX:
        break
    c_conform_out.interpolate(c_conform_expr)

    if step % 10 == 0:
        with dolfinx.io.XDMFFile(mesh.comm, "solution.xdmf", "a") as xdmf_file:
            u_out.x.array[:] = z.x.array[U_to_Z]
            c_out.x.array[:] = z.x.array[C_to_Z]
            psi_out.x.array[:] = z.x.array[Psi_to_Z]
            xdmf_file.write_function(u_out, T)
            xdmf_file.write_function(c_out, T)
            xdmf_file.write_function(psi_out, T)

        z_prev.interpolate(z)
        vtx_damage.write(T)

vtx_damage.close()
snes.destroy()
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mesh_output.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    mesh.topology.create_connectivity(1, 2)
    xdmf.write_meshtags(ft, mesh.geometry)
