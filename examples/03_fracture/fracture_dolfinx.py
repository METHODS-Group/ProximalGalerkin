from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from mpi4py import MPI

import basix.ufl
import dolfinx.fem.petsc
import numpy as np
from generate_mesh import create_crack_mesh
from ufl import (
    Circumradius,
    TestFunction,
    TrialFunction,
    derivative,
    dx,
    exp,
    grad,
    inner,
    split,
)

from lvpp import SNESProblem, SNESSolver

_RED = "\033[31m"
_BLUE = "\033[34m"
_GREEN = "\033[32m"
_color_reset = "\033[0m"


class NotConvergedError(Exception):
    pass


parser = ArgumentParser(
    description="Solve the obstacle problem on a unit square using Galahad.",
    formatter_class=ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--res",
    "-r",
    dest="res",
    type=float,
    default=0.0125,
    help="Resolution of the mesh",
)
parser.add_argument(
    "--max-fail-iter",
    type=int,
    default=50,
    dest="NFAIL_MAX",
    help="Maximum number of iterations of the LVPP that can fail before termination",
)
parser.add_argument(
    "--write-frequency",
    type=int,
    default=25,
    dest="write_frequency",
    help="Frequency of writing output to XDMFFile",
)
parser.add_argument(
    "--num-load-steps",
    type=int,
    default=1001,
    dest="num_load_steps",
    help="Number of load steps",
)
parser.add_argument("--verbose", "-v", action="store_true",
                    help="Verbose output from PETSc SNES")
parser.add_argument("--Tmin", type=float, default=0.0, help="Minimum load")
parser.add_argument("--Tmax", type=float, default=5.0, help="Maximum load")

args = parser.parse_args()
res = args.res
NFAIL_MAX = args.NFAIL_MAX
write_frequency = args.write_frequency
num_load_steps = args.num_load_steps
Tmin = args.Tmin
Tmax = args.Tmax
verbose = args.verbose

st = dolfinx.default_scalar_type
mesh, ft, material_map = create_crack_mesh(MPI.COMM_WORLD, max_res=0.0125)
el = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
el = basix.ufl.mixed_element([el, el, el])
Z = dolfinx.fem.functionspace(mesh, el)


G = dolfinx.fem.Constant(mesh, st(1.0))
Gc = dolfinx.fem.Constant(mesh, st(1.0))

# Compute maximum cell size
W = dolfinx.fem.functionspace(mesh, ("DG", 0))
h = dolfinx.fem.Expression(4 * Circumradius(mesh),
                           W.element.interpolation_points())
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
(u_iter, c_iter, psi_iter) = split(z_iter)


output_space = dolfinx.fem.functionspace(mesh, ("Lagrange", 3))
c_conform_out = dolfinx.fem.Function(output_space, name="ConformingDamage")
alpha = dolfinx.fem.Constant(mesh, st(1.0))
c_conform = (c_prev + exp(psi)) / (exp(psi) + 1)
c_conform_expr = dolfinx.fem.Expression(
    c_conform, output_space.element.interpolation_points())

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
cffi_options = ["-Ofast", "-march=native"]
jit_options = {
    "cffi_extra_compile_args": cffi_options,
    "cffi_libraries": ["m"],
}
F_compiled = dolfinx.fem.form(F, jit_options=jit_options)

reps = dolfinx.fem.Constant(mesh, 1.0e-3)
J_reg = dolfinx.fem.form(
    derivative(F, z, z_trial)
    + reps * inner(v, v_trial) * dx
    + reps * inner(d, d_trial) * dx
    - reps * inner(phi, phi_trial) * dx,
    jit_options=jit_options,
)

# Right side of crack (4), left crack (7)
bcminus = dolfinx.fem.Constant(mesh, 0.0)
bcplus = dolfinx.fem.Constant(mesh, 0.0)
mesh.topology.create_connectivity(1, 2)

left_dofs = np.hstack(
    [
        dolfinx.fem.locate_dofs_topological(Z.sub(0), ft.dim, ft.find(value))
        for value in material_map["topleft"]
    ]
)
right_dofs = np.hstack(
    [
        dolfinx.fem.locate_dofs_topological(Z.sub(0), ft.dim, ft.find(value))
        for value in material_map["topright"]
    ]
)
bcs = [
    dolfinx.fem.dirichletbc(bcplus, right_dofs, Z.sub(0)),
    dolfinx.fem.dirichletbc(bcminus, left_dofs, Z.sub(0)),
]


sp = {
    "snes_linesearch_type": "l2",
    "snes_linesearch_maxstep": 1,
    "snes_atol": 1.0e-6,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_14": 500,
}
if verbose:
    sp.update(
        {
            "snes_monitor": None,
            "snes_converged_reason": None,
            "snes_linesearch_monitor": None,
            "ksp_monitor": None,
        }
    )


xdmf_file = dolfinx.io.XDMFFile(mesh.comm, "solution.xdmf", "w")
xdmf_file.write_mesh(mesh)
xdmf_file.close()
vtx_damage = dolfinx.io.VTXWriter(mesh.comm, "damage.bp", [c_conform_out])
L2_c = dolfinx.fem.form(inner(c - c_iter, c - c_iter) * dx)
L2_u = dolfinx.fem.form(inner(u - u_iter, u - u_iter) * dx)
L2_z = dolfinx.fem.form(inner(z - z_prev, z - z_prev) * dx)

U, U_to_Z = Z.sub(0).collapse()
u_out = dolfinx.fem.Function(U, name="u")
C, C_to_Z = Z.sub(1).collapse()
c_out = dolfinx.fem.Function(C, name="c")
Psi, Psi_to_Z = Z.sub(2).collapse()
psi_out = dolfinx.fem.Function(Psi, name="psi")
converged_reason = -1
num_iterations = -1


problem = SNESProblem(F_compiled, z, bcs=bcs, J=J_reg)
for step, T in enumerate(np.linspace(Tmin, Tmax, num_load_steps)[1:]):
    if mesh.comm.rank == 0:
        print(
            f"{_BLUE} Solving for T = {float(T)} ({step / num_load_steps * 100:.1f}%){_color_reset}",
            flush=True,
        )
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
                print(
                    f"{_RED}Failed to converge ({converged_reason})",
                    f", {k=} alpha={float(alpha)}{_color_reset}",
                    flush=True,
                )
            alpha.value /= 2
            if k == 1:
                z.interpolate(z_prev)
            else:
                z.interpolate(z_iter)

            if nfail >= NFAIL_MAX:
                if mesh.comm.rank == 0:
                    print(
                        f"{_RED}Giving up. {T=} alpha={float(alpha)} {k=}{_color_reset}", flush=True
                    )
                break
            else:
                continue

        # Termination
        nrm_c = np.sqrt(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(
            L2_c), op=MPI.SUM))
        nrm_u = np.sqrt(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(
            L2_u), op=MPI.SUM))
        if mesh.comm.rank == 0:
            print(
                f"{_GREEN}Solved {k=} {num_iterations=} alpha={float(alpha)},"
                f"||c_{k} - c_{k - 1}|| = {nrm_c}{_color_reset}",
                flush=True,
            )
        if nrm_c < 1.0e-4:  # and nrm_u < 1.0e-4:
            break

        # Update alpha
        if num_iterations <= 4:
            alpha.value *= r
        elif num_iterations >= 10:
            alpha.value /= r

        # Update z_iter
        z_iter.interpolate(z)

        k += 1

    # When the object has broken (i.e. the crack has partitioned the domain),
    # the failure mode of the algorithm above is that it terminates in one
    # PG iteration that does no Newton iterations, so the solution doesn't
    # change
    norm_Z = np.sqrt(mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(L2_z), op=MPI.SUM))
    if k == 1 and np.isclose(norm_Z, 0.0):
        break

    if nfail == NFAIL_MAX:
        break
    z_prev.interpolate(z)
    if step % write_frequency == 0:
        c_conform_out.interpolate(c_conform_expr)
        with dolfinx.io.XDMFFile(mesh.comm, "solution.xdmf", "a") as xdmf_file:
            u_out.x.array[:] = z.x.array[U_to_Z]
            c_out.x.array[:] = z.x.array[C_to_Z]
            psi_out.x.array[:] = z.x.array[Psi_to_Z]
            xdmf_file.write_function(u_out, T)
            xdmf_file.write_function(c_out, T)
            xdmf_file.write_function(psi_out, T)

        vtx_damage.write(T)
vtx_damage.close()
