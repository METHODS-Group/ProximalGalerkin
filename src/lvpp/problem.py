import typing

from petsc4py import PETSc

import dolfinx.fem.petsc
import ufl

__all__ = [
    "SNESProblem",
    "SNESSolver",
]


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
        try:
            dolfinx.fem.petsc.apply_lifting(F, [self.a], bcs=[self.bcs], x0=[x], scale=-1.0)
        except TypeError:
            dolfinx.fem.petsc.apply_lifting(F, [self.a], bcs=[self.bcs], x0=[x], alpha=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(F, self.bcs, x, -1.0)

    def J(self, snes, x, J, P):
        """Assemble Jacobian matrix."""
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.x.petsc_vec)
        self.u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        J.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(J, self.a, self.bcs)
        J.assemble()


class SNESSolver:
    def __init__(self, problem, options):
        """Initialize a PETSc-SNES solver

        Args:
            problem: A problem instance for PETSc SNES
            options: Solver options. Can include any options for sub objects such as KSP and PC
        """
        self.problem = problem
        self.options = options
        self.create_solver()

        self.create_data_structures()

    def create_solver(self):
        """Create the PETSc SNES object and set solver options"""
        self._snes = PETSc.SNES().create(comm=self.problem.u.function_space.mesh.comm)
        self._snes.setOptionsPrefix("snes_solve")
        option_prefix = self._snes.getOptionsPrefix()
        opts = PETSc.Options()
        opts.prefixPush(option_prefix)
        for key, v in self.options.items():
            opts[key] = v
        opts.prefixPop()
        self._snes.setFromOptions()

    def create_data_structures(self):
        """
        Create PETSc objects for the matrix, residual and solution
        """
        self._A = dolfinx.fem.petsc.create_matrix(self.problem.a)
        self._b = dolfinx.fem.Function(self.problem.u.function_space, name="Residual")
        self._x = dolfinx.fem.Function(self.problem.u.function_space, name="tmp_solution")

    def solve(self):
        self._snes.setFunction(self.problem.F, self._b.x.petsc_vec)
        self._snes.setJacobian(self.problem.J, self._A)
        self._x.interpolate(self.problem.u)
        self._snes.solve(None, self._x.x.petsc_vec)
        self._x.x.scatter_forward()
        converged_reason = self._snes.getConvergedReason()
        if converged_reason > 0:
            # Only update the solution if the solver converged
            self.problem.u.interpolate(self._x)
        return converged_reason, self._snes.getIterationNumber()

    def __del__(self):
        self._snes.destroy()
