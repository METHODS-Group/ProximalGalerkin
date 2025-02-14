import typing

import numpy as np
import numpy.typing as npt

__all__ = [
    "OptimizationProblem",
    "galahad_solver",
    "ipopt_solver",
]


class OptimizationProblem(typing.Protocol):
    total_iteration_count: int

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

    def hessianstructure(
        self,
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]: ...


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

        assert len(bounds[0]) == len(bounds[1]) == len(x_init), (
            "Bounds and x_init must have the same length"
        )

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
        assert len(bounds[0]) == len(bounds[1]) == len(x_init), (
            "Bounds and x_init must have the same length"
        )

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

        nlp = cyipopt.Problem(
            n=len(x_init),
            m=0,
            lb=bounds[0],
            ub=bounds[1],
            problem_obj=problem,
        )

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
