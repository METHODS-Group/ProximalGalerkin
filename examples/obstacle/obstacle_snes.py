"""
Solve obstacle problem with SNES
Author: Jørgen S. Dokken
SPDX-License-Identifier: MIT
The SNES solver is based on https://github.com/Wells-Group/asimov-contact and is distributed under the MIT License
"""

import argparse
import typing
from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx.fem.petsc
import numpy as np
import ufl

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


class SNESProblem:
    def __init__(
        self,
        F: typing.Union[dolfinx.fem.form, ufl.form.Form],
        u: dolfinx.fem.Function,
        J: typing.Optional[typing.Union[dolfinx.fem.form, ufl.form.Form]] = None,
        bcs: typing.Optional[list[dolfinx.fem.DirichletBC]] = None,
        form_compiler_options: typing.Optional[dict] = None,
        jit_options: typing.Optional[dict] = None,
    ):
        """
        Initialize class for constructing the residual and Jacobian constructors for a SNES problem.

        :param F: Variational form of the residual
        :param u: The unknown function
        :param J: Variational form of the Jacobian
        :param bcs: List of Dirichlet boundary conditions to enforce
        :param form_compiler_options: Options for form compiler
        :param jit_options: Options for Just In Time compilation
        """
        form_compiler_options = {} if form_compiler_options is None else form_compiler_options
        jit_options = {} if jit_options is None else jit_options

        self.L = dolfinx.fem.form(
            F, form_compiler_options=form_compiler_options, jit_options=jit_options
        )
        if J is None:
            V = u.function_space
            du = ufl.TrialFunction(V)
            self.a = dolfinx.fem.form(
                ufl.derivative(F, u, du),
                form_compiler_options=form_compiler_options,
                jit_options=jit_options,
            )
        else:
            self.a = J
        self.bcs = bcs
        self._F, self._J = None, None
        self.u = u

    def F(self, snes, x, F):
        """Assemble residual vector."""
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.x.petsc_vec)
        self.u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        with F.localForm() as f_local:
            f_local.set(0.0)
        dolfinx.fem.petsc.assemble_vector(F, self.L)
        dolfinx.fem.petsc.apply_lifting(F, [self.a], bcs=[self.bcs], x0=[x], scale=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(F, self.bcs, x, -1.0)

    def J(self, snes, x, J, P):
        """Assemble Jacobian matrix."""
        J.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(J, self.a, self.bcs)
        J.assemble()


def solve_problem(
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
    result_dir = Path("results")
    with dolfinx.io.XDMFFile(mesh.comm, result_dir / "u_snes.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(u_out)

    # with dolfinx.io.XDMFFile(mesh.comm,result_dir /  "phi_snes.xdmf", "w") as xdmf:
    #     xdmf.write_mesh(mesh)
    #     xdmf.write_function(phi)


if __name__ == "__main__":
    args = parser.parse_args()
    snes_solver = "vinewtonssls"  # "vinewtonrsls"
    solve_problem(
        args.infile,
        snes_options={"snes_type": snes_solver, "snes_monitor": None},
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )
