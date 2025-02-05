"""
Solving the obstacle problem using Galahad or IPOPT with DOLFINx generating the system matrices
Author: Jørgen S. Dokken
SPDX-License-Identifier: MIT
"""

import argparse
from pathlib import Path

import dolfinx
import numpy as np
import numpy.typing as npt
import scipy.sparse
import ufl
from run_lvpp_problem import solve_contact_problem
from snes import solve_snes_problem

from lvpp import AlphaScheme, galahad_solver, ipopt_solver
from lvpp.mesh_generation import create_half_disk

parser = argparse.ArgumentParser(
    description="Solve the obstacle problem on a unit square using Galahad.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--outdir", type=Path, default=Path("results"), help="Output directory")
solver_params = parser.add_argument_group("Solver parameters")
solver_params.add_argument("--ipopt", action="store_true", default=False, help="Use Ipopt")
solver_params.add_argument("--galahad", action="store_true", default=False, help="Use Galahad")
solver_params.add_argument("--lvpp", action="store_true", default=False, help="Use lvpp")
solver_params.add_argument("--snes", action="store_true", default=False, help="Use SNES")
solver_params.add_argument("--max-iter", type=int, default=200, help="Maximum number of iterations")
solver_params.add_argument("--tol", type=float, default=1e-6, help="Convergence tolerance")
solver_params.add_argument(
    "--hessian",
    dest="use_hessian",
    action="store_true",
    default=False,
    help="Use exact hessian",
)
problem_params = parser.add_argument_group("Problem parameters")
problem_params.add_argument("--E", dest="E", type=float, default=2.0e5, help="Young's modulus")
problem_params.add_argument("--nu", dest="nu", type=float, default=0.3, help="Poisson's ratio")
problem_params.add_argument("-R", dest="R", type=float, default=1.0, help="Radius of the half disk")
problem_params.add_argument(
    "--cy", dest="cy", type=float, default=1.2, help="Center of the half disk"
)
problem_params.add_argument(
    "-g",
    type=float,
    dest="g",
    default=0.3,
    help="Amount of forced displacement in y direction",
)
problem_params.add_argument(
    "-o",
    "--order",
    type=int,
    default=2,
    help="Order of finite element used to represent the geometry",
)
problem_params.add_argument(
    "-r",
    "--refinement-level",
    dest="r_lvl",
    type=int,
    required=True,
    help="Refinement level of the mesh",
)


def setup_problem(
    mesh: dolfinx.mesh.Mesh,
    facet_tags: dolfinx.mesh.MeshTags,
    c_marker: int,
    t_marker: int,
    g: float,
    E: float,
    nu: float,
    f: tuple[float, float] = (0.0, 0.0),
):
    """
    Generate the stiffness matrix, right hand side load vector
    and bounds for the Signorini problem.

    Args:
        mesh: The computational domain
        facet_tags: Marker function for boundary facets
        c_marker: Index indicating what facets in ``boundary_facets`` could be in contact
        t_marker: Index indicating what facets in ``boundary_facets`` has fixed
            displacement ``(0, -g)``
        g: Displacement enforced on top part of cylinder (enforced in negative y direction)
        E: Young's modulus
        nu: Poisson's ratio
        f: Body force
        r_lvl: Refinement level of the mesh

    Returns:
        The stiffness matrix, right hand side load vector and bounds
    """
    assert facet_tags.dim == mesh.topology.dim - 1
    Vh = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))

    # Create variational form
    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    u = ufl.TrialFunction(Vh)
    v = ufl.TestFunction(Vh)
    mu_s = E / (2.0 * (1.0 + nu))
    dst = dolfinx.default_scalar_type
    mu = dolfinx.fem.Constant(
        mesh,
        dst([[mu_s for _ in range(mesh.geometry.dim)] for _ in range(mesh.geometry.dim)]),
    )
    lmbda = dolfinx.fem.Constant(mesh, dst(E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))))
    C = lmbda * ufl.Identity(mesh.geometry.dim) + 2.0 * mu
    f_c = dolfinx.fem.Constant(mesh, dst(f))
    stiffness = ufl.inner(C * epsilon(u), epsilon(v)) * ufl.dx
    L = ufl.inner(f_c, v) * ufl.dx

    # Create boundary conditions
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    fixed_dofs = dolfinx.fem.locate_dofs_topological(Vh, tdim - 1, facet_tags.find(t_marker))
    u_bc = dolfinx.fem.Function(Vh)
    u_bc.interpolate(lambda x: (np.zeros(x.shape[1]), np.full(x.shape[1], -g)))
    bcs = [dolfinx.fem.dirichletbc(u_bc, fixed_dofs)]

    # Compute gap function
    def gap(x):
        return (np.zeros(x.shape[1]), x[1])

    gap_Vh = dolfinx.fem.Function(Vh)
    gap_Vh.interpolate(gap)

    # Restrict gap constraint to boundary dofs that are not in bcs
    potential_contact_facets = facet_tags.find(c_marker)
    bound_dofs = dolfinx.fem.locate_dofs_topological(Vh.sub(1), tdim - 1, potential_contact_facets)

    # Compute lower and upper bounds
    lower_bound = dolfinx.fem.Function(Vh, name="lower_bound")
    upper_bound = dolfinx.fem.Function(Vh, name="upper_bound")
    lower_bound.x.array[:] = -np.inf
    lower_bound.x.array[bound_dofs] = -gap_Vh.x.array[bound_dofs]
    upper_bound.x.array[:] = np.inf
    # Set equality for Dirichlet Dofs
    dolfinx.fem.set_bc(lower_bound.x.array, bcs)
    dolfinx.fem.set_bc(upper_bound.x.array, bcs)

    # Assemble system
    S = dolfinx.fem.assemble_matrix(dolfinx.fem.form(stiffness))
    rhs = dolfinx.fem.Function(Vh)
    dolfinx.fem.assemble_vector(rhs.x.array, dolfinx.fem.form(L))
    rhs.x.scatter_reverse(dolfinx.la.InsertMode.add)
    return S, rhs, (lower_bound, upper_bound)


class SignoriniProblem:
    def __init__(self, S: scipy.sparse.csr_matrix, f: npt.NDArray[np.float64]):
        S.eliminate_zeros()
        self._S = S
        self._f = f
        tri_S = scipy.sparse.tril(self._S)
        self._sparsity = tri_S.nonzero()
        self._H_data = tri_S.data

    def objective(self, x: npt.NDArray[np.float64]) -> np.float64:
        """Returns the scalar value of the objective given x."""
        return 0.5 * x.T @ (self._S @ x) - np.dot(self._f, x)

    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""

        return self._S @ x - self._f

    def pure_hessian(self, x):
        return self._H_data

    def hessian(self, x, lagrange, obj_factor):
        return obj_factor * self.pure_hessian(x)

    def hessianstructure(self):
        return self._sparsity

    def intermediate(self, *args):
        """Ipopt callback function"""
        self.total_iteration_count = args[1]


if __name__ == "__main__":
    args = parser.parse_args()
    c_marker = 2
    t_marker = 1
    # Generate the mesh and create function space
    mesh, _, ft = create_half_disk(
        args.cy,
        args.R,
        0.02,
        refinement_level=args.r_lvl,
        order=args.order,
        disk_marker=c_marker,
        top_marker=t_marker,
    )
    S, f, bounds = setup_problem(
        mesh=mesh,
        facet_tags=ft,
        c_marker=c_marker,
        t_marker=t_marker,
        g=args.g,
        E=args.E,
        nu=args.nu,
    )
    Vh = f.function_space

    # Restrict all matrices and vectors to interior dofs
    S_d = S.to_scipy().tocsr()
    f_d = f.x.array.copy()

    problem = SignoriniProblem(S_d, f_d)

    lower_bound = bounds[0].x.array
    upper_bound = bounds[1].x.array
    outdir = args.outdir
    iterations = {}
    if args.galahad:
        x_g = dolfinx.fem.Function(Vh, name="galahad")
        x_g.x.array[:] = 0
        x_galahad, num_iterations = galahad_solver(
            problem,
            x_g.x.array.copy(),
            (lower_bound, upper_bound),
            max_iter=args.max_iter,
            use_hessian=args.use_hessian,
            tol=args.tol,
        )
        iterations["galahad"] = num_iterations
        x_g.x.array[:] = x_galahad
        mesh = x_g.function_space.mesh
        degree = mesh.geometry.cmap.degree
        V_out = dolfinx.fem.functionspace(mesh, ("Lagrange", degree, (mesh.geometry.dim,)))
        x_g_out = dolfinx.fem.Function(V_out, name="galahad")
        x_g_out.interpolate(x_g)
        with dolfinx.io.VTXWriter(V_out.mesh.comm, outdir / "galahad.bp", [x_g_out]) as bp:
            bp.write(0.0)

    if args.ipopt:
        x_i = dolfinx.fem.Function(Vh, name="ipopt")
        x_i.x.array[:] = 0
        x_ipopt = ipopt_solver(
            problem,
            x_i.x.array.copy(),
            (lower_bound.copy(), upper_bound.copy()),
            max_iter=args.max_iter,
            tol=args.tol,
            activate_hessian=args.use_hessian,
        )
        x_i.x.array[:] = x_ipopt

        # Output on geometry space
        mesh = x_i.function_space.mesh
        degree = mesh.geometry.cmap.degree
        V_out = dolfinx.fem.functionspace(mesh, ("Lagrange", degree, (mesh.geometry.dim,)))
        x_i_out = dolfinx.fem.Function(V_out, name="ipopt")
        x_i_out.interpolate(x_i)
        with dolfinx.io.VTXWriter(mesh.comm, outdir / "ipopt.bp", [x_i_out]) as bp:
            bp.write(0.0)
        iterations["lvpp"] = problem.total_iteration_count
    if args.lvpp:
        bcs = {"contact": (c_marker,), "displacement": (t_marker,)}

        lvpp_it = solve_contact_problem(
            mesh=mesh,
            facet_tag=ft,
            boundary_conditions=bcs,
            degree=2,
            E=args.E,
            nu=args.nu,
            gap=0,
            disp=-args.g,
            newton_max_its=200,
            newton_tol=args.tol,
            max_iterations=args.max_iter,
            alpha_scheme=AlphaScheme.constant,
            alpha_0=5e-3,
            alpha_c=0.2,
            tol=args.tol,
            output=outdir,
            quadrature_degree=5,
        )
        iterations["lvpp"] = lvpp_it

    if args.snes:
        bcs = {"contact": (c_marker,), "displacement": (t_marker,)}
        snes_it = solve_snes_problem(
            mesh=mesh,
            facet_tag=ft,
            boundary_conditions=bcs,
            E=args.E,
            nu=args.nu,
            disp=-args.g,
            max_iterations=args.max_iter,
            tol=args.tol,
            output=outdir,
        )
        iterations["snes"] = snes_it

    print("Number of iterations")
    for k, v in iterations.items():
        print(f"{k}: {v}")
