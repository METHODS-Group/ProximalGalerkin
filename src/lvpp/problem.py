from petsc4py import PETSc
import typing
import dolfinx
import ufl
import numpy as np
import numpy.typing as npt

__all__ = ["OptimizationProblem", "galahad_solver", "ipopt_solver", "SNESProblem"]


class OptimizationProblem(typing.Protocol):
    def objective(self, x: npt.NDArray[np.float64]) -> np.float64:
        """Returns the scalar value of the objective given x."""
        ...

    def gradient(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Returns the gradient of the objective with respect to x."""
        ...

    def pure_hessian(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Returns the lower triangular part of the Hessian of the objective with respect to x."""
        ...

    def hessian(
        self, x: npt.NDArray[np.float64], lagrange: float, obj_factor: float
    ) -> npt.NDArray[np.float64]:
        """Return an IPOPT compatible Hessian"""
        return obj_factor * self.pure_hessian(x)

    def hessianstructure(self) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]: ...


try:
    from galahad import trb

    def galahad_solver(
        problem: OptimizationProblem,
        x_init: npt.NDArray[np.float64],
        bounds: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
        log_level: int = 1,
        use_hessian: bool = True,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> tuple[npt.NDArray[np.float64], int]:
        """A wrapper around Galahad to solve optimization problems with constrained primal variables
        Args:
            problem: An :class:`OptimizationProblem` instance.
            x_init: Initial condition
            bounds: (lower_bound, upper_bound) of the optimization problem
            log_level: Verbosity level for galahad (0-3)
            use_hessian: If True use second order method, otherwise use first order
            max_iter: Maximum number of iterations
            tol: Relative convergence tolerance
        Returns:
            Optimized solution and number of iterations used
        """

        assert (
            len(bounds[0]) == len(bounds[1]) == len(x_init)
        ), "Bounds and x_init must have the same length"

        options = trb.initialize()

        # set some non-default options
        options["print_level"] = log_level
        options["model"] = 2 if use_hessian else 1
        options["maxit"] = max_iter
        options["hessian_available"] = True
        options["stop_pg_relative"] = tol
        options["subproblem_direct"] = True
        n = len(x_init)
        H_type = "coordinate"
        H_ne = len(problem.hessianstructure()[0])
        H_ptr = None
        # Add Dirichlet bounds 0 here
        trb.load(
            n,
            bounds[0].copy(),
            bounds[1].copy(),
            H_type,
            H_ne,
            problem.hessianstructure()[0].astype(np.int64),
            problem.hessianstructure()[1].astype(np.int64),
            H_ptr=H_ptr,
            options=options,
        )
        x_out, _ = trb.solve(
            n, H_ne, x_init, problem.objective, problem.gradient, problem.pure_hessian
        )
        return x_out, trb.information()["iter"]

except ModuleNotFoundError:

    def galahad_solver(
        problem: OptimizationProblem,
        x_init: npt.NDArray[np.float64],
        bounds: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
        log_level: int = 1,
        use_hessian: bool = True,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> tuple[npt.NDArray[np.float64], int]:
        raise ModuleNotFoundError("Galahad has not been installed")


try:
    import cyipopt

    def ipopt_solver(
        problem: OptimizationProblem,
        x_init: npt.NDArray[np.float64],
        bounds: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
        log_level: int = 5,
        max_iter: int = 100,
        tol: float = 1e-6,
        activate_hessian: bool = True,
    ) -> npt.NDArray[np.float64]:
        """A wrapper around CyIpopt to solve optimization problems with constrained primal variables
        Args:
            problem: An :class:`OptimizationProblem` instance.
            x_init: Initial condition
            bounds: (lower_bound, upper_bound) of the optimization problem
            log_level: Verbosity level for Ipopt
            use_hessian: If True use second order method, otherwise use first order
            max_iter: Maximum number of iterations
            tol: Relative convergence tolerance

        Returns:
            Optimized solution
        """
        assert (
            len(bounds[0]) == len(bounds[1]) == len(x_init)
        ), "Bounds and x_init must have the same length"

        options = {
            "print_level": log_level,
            "max_iter": max_iter,
            "tol": tol,
            "jacobian_approximation": "exact",
        }

        if activate_hessian:
            options["hessian_approximation"] = "exact"
            options["hessian_constant"] = "yes"
        else:
            options["hessian_approximation"] = "limited-memory"

        nlp = cyipopt.Problem(n=len(x_init), m=0, lb=bounds[0], ub=bounds[1], problem_obj=problem)
        for key, val in options.items():
            nlp.add_option(key, val)

        x_opt, _ = nlp.solve(x_init)
        return x_opt

except ModuleNotFoundError:

    def ipopt_solver(
        problem: OptimizationProblem,
        x_init: npt.NDArray[np.float64],
        bounds: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
        log_level: int = 5,
        max_iter: int = 100,
        tol: float = 1e-6,
        activate_hessian: bool = True,
    ) -> npt.NDArray[np.float64]:
        raise ModuleNotFoundError("cyipopt has not been installed")




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

