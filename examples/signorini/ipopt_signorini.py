"""
Solving the obstacle problem using Galahad or IPOPT with DOLFINx generating the system matrices
Author: Jørgen S. Dokken
SPDX-License-Identifier: MIT
"""

import cyipopt
from pathlib import Path
from mpi4py import MPI
import dolfinx
import ufl
import argparse
import numpy as np
import scipy.sparse
from generate_half_disk import generate_half_disk

parser = argparse.ArgumentParser(
    description="Solve the obstacle problem on a unit square using Galahad.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("-R", dest="R", type=float, default=1.0, help="Radius of the half disk")
parser.add_argument("--cy", dest="cy", type=float, default=1.2, help="Center of the half disk")
parser.add_argument("-g", type=float, dest="g", default=0.3, help="Amount of forced displacement in y direction")
parser.add_argument("--ipopt", action="store_true", default=False, help="Use Ipopt")
parser.add_argument("--galahad", action="store_true", default=False, help="Use Galahad")
parser.add_argument(
    "--max-iter", type=int, default=200, help="Maximum number of iterations"
)
parser.add_argument("--tol", type=float, default=1e-6, help="Convergence tolerance")
parser.add_argument("--hessian", dest="use_hessian", action="store_true", default=False, help="Use exact hessian")
physical_parameters = parser.add_argument_group("Physical parameters")
physical_parameters.add_argument(
    "--E", dest="E", type=float, default=2.0e5, help="Young's modulus"
)
physical_parameters.add_argument(
    "--nu", dest="nu", type=float, default=0.3, help="Poisson's ratio"
)

def epsilon(u):
    return ufl.sym(ufl.grad(u))

def setup_problem(
     R: float,
     cy: float,
     g: float,
     E: float,
     nu: float,
     f: tuple[float, float] = (0.0, 0.0),
     r_lvl: int = 0
):
    mesh = generate_half_disk(cy, R, 0.1, refinement_level=r_lvl, order=1)

    Vh = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim, )))
    u = ufl.TrialFunction(Vh)
    v = ufl.TestFunction(Vh)


    mu_s = E / (2.0 * (1.0 + nu))
    mu = dolfinx.fem.Constant(mesh, [[mu_s for _ in range(mesh.geometry.dim)] for _ in range(mesh.geometry.dim)])
    lmbda = dolfinx.fem.Constant(mesh, E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))
    C = lmbda * ufl.Identity(mesh.geometry.dim) + 2.0 * mu 

    f_c = dolfinx.fem.Constant(mesh, f)
    stiffness = ufl.inner(C*epsilon(u), epsilon(v)) * ufl.dx

    rhs = ufl.inner(f_c, v) * ufl.dx

    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    fixed_facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim-1, lambda x: np.isclose(x[1], cy))
    fixed_dofs = dolfinx.fem.locate_dofs_topological(Vh, tdim - 1, fixed_facets)

    u_bc = dolfinx.fem.Function(Vh)
    u_bc.interpolate(lambda x: (np.zeros(x.shape[1]), np.full(x.shape[1], -g)))

    # Get dofs to deactivate
    bcs = [dolfinx.fem.dirichletbc(u_bc, fixed_dofs)]

    def gap(x):
        return (np.zeros(x.shape[1]), x[1])
    gap_Vh = dolfinx.fem.Function(Vh)
    gap_Vh.interpolate(gap)

    all_boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    potential_contact_facets = np.setdiff1d(all_boundary_facets, fixed_facets)
    bound_dofs = dolfinx.fem.locate_dofs_topological(Vh.sub(1), tdim - 1, potential_contact_facets)

    lower_bound = dolfinx.fem.Function(Vh, name="lower_bound")
    upper_bound = dolfinx.fem.Function(Vh, name="upper_bound")
    
    
    lower_bound.x.array[:] = -np.inf
    lower_bound.x.array[bound_dofs] = -gap_Vh.x.array[bound_dofs]
    upper_bound.x.array[:] = np.inf
 
    dolfinx.fem.set_bc(lower_bound.x.array, bcs)
    dolfinx.fem.set_bc(upper_bound.x.array, bcs)

    x_init = dolfinx.fem.Function(Vh)
    dolfinx.fem.set_bc(x_init.x.array, bcs)
    S = dolfinx.fem.assemble_matrix(dolfinx.fem.form(stiffness))
    f = dolfinx.fem.Function(Vh)
    dolfinx.fem.assemble_vector(f.x.array, dolfinx.fem.form(rhs))
    return S.to_scipy(),f, Vh, (lower_bound, upper_bound), x_init


def galahad(
    problem,
    x,
    bounds,
    log_level: int = 1,
    use_hessian: bool = True,
    max_iter: int = 100,
    tol: float = 1e-6,
):
    """
    :param problem: Problem instance
    :param x: Initial condition
    :param bounds: (lower_bound, upper_bound)_
    :param loglevel: Verbosity level for galahad (0-3)
    :param use_hessian: If True use second order method, otherwise use first order
    :param max_iter: Maximum number of iterations
    :param tol: Relative convergence tolerance
    :return: Optimized solution
    """

    from galahad import trb

    options = trb.initialize()

    # set some non-default options
    options["print_level"] = log_level
    options["model"] = 2 if use_hessian else 1
    options["maxit"] = max_iter
    options["hessian_available"] = True
    options["stop_pg_relative"] = tol
    options["subproblem_direct"] = True
    n = len(x)
    H_type = "coordinate"
    H_ne = len(problem.sparsity[0])
    H_ptr = None
    # Add Dirichlet bounds 0 here
    trb.load(
        n,
        bounds[0].copy(),
        bounds[1].copy(),
        H_type,
        H_ne,
        problem.sparsity[0].astype(np.int64),
        problem.sparsity[1].astype(np.int64),
        H_ptr=H_ptr,
        options=options,
    )
    x_out, _ = trb.solve(
        n, H_ne, x, problem.objective, problem.gradient, problem.pure_hessian
    )
    return x_out, trb.information()["iter"]


class SignoriniProblem:
    def __init__(self, S, f):
        S.eliminate_zeros()
        self._S = S
        self._f = f
        tri_S = scipy.sparse.tril(self._S)
        self.sparsity = tri_S.nonzero()
        self._H_data = np.copy(tri_S.todense())[self.sparsity[0], self.sparsity[1]]

    def objective(self, x):
        """Returns the scalar value of the objective given x."""
        return 0.5 * x.T @ (self._S @ x) - np.dot(self._f, x)

    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""

        return self._S @ x - self._f

    def pure_hessian(self, x):
        return self._H_data

    def hessian(self, x, lagrange, obj_factor):
        return obj_factor*self.pure_hessian(x)

    def hessianstructure(self):
        return self.sparsity
    


def ipopt(
    problem, x, bounds, log_level: int = 5, max_iter: int = 100, tol: float = 1e-6,
    activate_hessian:bool=True
):
    """
    :param problem: Problem instance
    :param x: Initial condition
    :param log_level: Veribosity level for Ipopt
    :param bounds: (lower_bound, upper_bound)
    :param tol: Relative convergence tolerance
    :return: Optimized solution
    """
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
        n=len(x), m=0, lb=bounds[0], ub=bounds[1], problem_obj=problem
    )
    for key, val in options.items():
        nlp.add_option(key, val)

    x_opt, _ = nlp.solve(x)
    return x_opt


if __name__ == "__main__":
    args = parser.parse_args()

    S_, f_, V, bounds, x_init = setup_problem(R=args.R, cy=args.cy, g=args.g, E=args.E, nu=args.nu)
   
    # Restrict all matrices and vectors to interior dofs
    S_d = S_.tocsr()
    f_d = f_.x.array.copy()

    problem = SignoriniProblem(S_d, f_d)

    lower_bound = bounds[0].x.array
    upper_bound = bounds[1].x.array
    outdir = Path("results")
    if args.galahad:
        x_g = dolfinx.fem.Function(V, name="galahad")
        x_g.x.array[:] = x_init.x.array
        init_galahad = x_g.x.array.copy()
        x_galahad, num_iterations = galahad(
            problem,
            init_galahad,
            (lower_bound, upper_bound),
            max_iter=args.max_iter,
            use_hessian=args.use_hessian,
            tol=args.tol
        )
        x_g.x.array[:] = x_galahad
        mesh = x_g.function_space.mesh
        degree = mesh.geometry.cmap.degree
        V_out = dolfinx.fem.functionspace(mesh, ("Lagrange", degree, (mesh.geometry.dim,)))
        x_g_out = dolfinx.fem.Function(V_out, name="galahad")
        x_g_out.interpolate(x_g)
        with dolfinx.io.VTXWriter(V.mesh.comm, outdir/"galahad.bp", [x_g_out]) as bp:
            bp.write(0.0)

    if args.ipopt:
        x_i = dolfinx.fem.Function(V, name="ipopt")
        x_i.x.array[:] = x_init.x.array
        init_ipopt = x_i.x.array.copy()
        x_ipopt = ipopt(
            problem, init_ipopt, (lower_bound.copy(), upper_bound.copy()), max_iter=args.max_iter,
            tol=args.tol, activate_hessian=args.use_hessian
        )

        x_i.x.array[:] = x_ipopt
        # Output on geometry space
        mesh = x_i.function_space.mesh
        degree = mesh.geometry.cmap.degree
        V_out = dolfinx.fem.functionspace(mesh, ("Lagrange", degree, (mesh.geometry.dim,)))
        x_i_out = dolfinx.fem.Function(V_out, name="ipopt")
        x_i_out.interpolate(x_i)
        with dolfinx.io.VTXWriter(mesh.comm, outdir/"ipopt.bp", [x_i_out]) as bp:
            bp.write(0.0)
