"""
Solve obstacle problem with SNES
Author: Jørgen S. Dokken
SPDX-License-Identifier: MIT

The SNES solver is based on https://github.com/Wells-Group/asimov-contact
and is distributed under the MIT License.
The license file can be found under [../../licenses/LICENSE.asimov](../../licenses/LICENSE.asimov)
"""

import argparse
import typing
from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx.fem.petsc
import numpy as np
import ufl

from lvpp.problem import SNESProblem

parser = argparse.ArgumentParser(
    description="Solve the obstacle problem on a unit square.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--path",
    "-P",
    dest="infile",
    type=Path,
    required=True,
    help="Path to infile",
)


def snes_solve(
    filename: Path,
    snes_options: typing.Optional[dict] = None,
    petsc_options: typing.Optional[dict] = None,
    tol: float = 1e-8,
):
    snes_options = {} if snes_options is None else snes_options
    petsc_options = {} if petsc_options is None else petsc_options

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
        mesh = xdmf.read_mesh(name="mesh")

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    def phi_set(x):
        r = np.sqrt(x[0] ** 2 + x[1] ** 2)
        r0 = 0.5
        beta = 0.9
        b = r0 * beta
        tmp = np.sqrt(r0**2 - b**2)
        B = tmp + b * b / tmp
        C = -b / tmp
        cond_true = B + r * C
        cond_false = np.sqrt(r0**2 - r**2)
        true_indices = np.flatnonzero(r > b)
        cond_false[true_indices] = cond_true[true_indices]
        return cond_false

    # Lower bound for the obstacle
    phi = dolfinx.fem.Function(V)
    phi.interpolate(phi_set)

    uh = dolfinx.fem.Function(V)
    v = ufl.TestFunction(V)
    f = dolfinx.fem.Constant(mesh, (0.0))
    F = (ufl.inner(ufl.grad(uh), ufl.grad(v)) - ufl.inner(f, v)) * ufl.dx
    J = ufl.derivative(F, uh)
    F_compiled = dolfinx.fem.form(F)
    J_compiled = dolfinx.fem.form(J)

    # bc_expr = dolfinx.fem.Expression(u_ex, V.element.interpolation_points())
    u_bc = dolfinx.fem.Function(V)
    u_bc.x.array[:] = 0.0
    # u_bc.interpolate(bc_expr)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
    bcs = [dolfinx.fem.dirichletbc(u_bc, boundary_dofs)]

    u_max = dolfinx.fem.Function(V)
    u_max.x.array[:] = PETSc.INFINITY

    # Create semismooth Newton solver (SNES)
    snes = PETSc.SNES().create(comm=mesh.comm)  # type: ignore
    snes.setTolerances(tol, tol, tol, 1000)
    # Set SNES options
    opts = PETSc.Options()  # type: ignore
    snes.setOptionsPrefix("snes_solve")
    option_prefix = snes.getOptionsPrefix()
    opts.prefixPush(option_prefix)
    for k, v in snes_options.items():
        opts[k] = v
    opts.prefixPop()
    snes.setFromOptions()

    b = dolfinx.fem.Function(V)
    b_vec = b.x.petsc_vec

    # Create nonlinear problem
    problem = SNESProblem(F_compiled, uh, bcs=bcs, J=J_compiled)

    A = dolfinx.fem.petsc.create_matrix(J_compiled)

    # Set solve functions and variable bounds
    snes.setFunction(problem.F, b_vec)
    snes.setJacobian(problem.J, A)
    snes.setVariableBounds(phi.x.petsc_vec, u_max.x.petsc_vec)

    # Set ksp options
    ksp = snes.ksp
    ksp.setOptionsPrefix("snes_ksp")
    opts = PETSc.Options()  # type: ignore
    option_prefix = ksp.getOptionsPrefix()
    opts.prefixPush(option_prefix)
    for k, v in petsc_options.items():
        opts[k] = v
    opts.prefixPop()
    ksp.setFromOptions()

    snes.solve(None, uh.x.petsc_vec)
    mesh = uh.function_space.mesh
    degree = mesh.geometry.cmap.degree
    V_out = dolfinx.fem.functionspace(mesh, ("Lagrange", degree))
    u_out = dolfinx.fem.Function(V_out, name="llvp")
    u_out.interpolate(uh)

    return u_out, snes.getIterationNumber()


if __name__ == "__main__":
    args = parser.parse_args()
    snes_solve(
        args.infile,
        snes_options={"snes_type": "vinewtonssls", "snes_monitor": None},
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )
