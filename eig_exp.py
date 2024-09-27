# Contact example
# SPDX-License-Identifier: MIT

import argparse
import typing
from pathlib import Path

from mpi4py import MPI
from lvpp import AlphaScheme, NewtonSolver
import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl
from matrix_exp import expm

class _HelpAction(argparse._HelpAction):
    """From https://stackoverflow.com/questions/20094215"""

    def __call__(self, parser, namespace, values, option_string=None):
        parser.print_help()

        # retrieve subparsers from parser
        subparsers_actions = [
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        ]
        # there will probably only be one subparser_action,
        # but better save than sorry
        for subparsers_action in subparsers_actions:
            # get all subparsers and print help
            for choice, subparser in subparsers_action.choices.items():
                print("Subparser '{}'".format(choice))
                print(subparser.format_help())

        parser.exit()


class CustomParser(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
): ...


desc = (
    "Signorini contact problem solver\n\n"
    + "Uses the Latent Variable Proximal Point algorithm combined with"
    + " a Newton solver at each step in the proximal point algorithm\n"
)
parser = argparse.ArgumentParser(
    description=desc, formatter_class=CustomParser, add_help=False
)
parser.add_argument(
    "--help", "-h", action=_HelpAction, help="show this help message and exit"
)
parser.add_argument(
    "--output", "-o", type=Path, default=Path("output"), help="Output directory"
)
physical_parameters = parser.add_argument_group("Physical parameters")
physical_parameters.add_argument(
    "--E", dest="E", type=float, default=2.0, help="Young's modulus"
)
physical_parameters.add_argument(
    "--nu", dest="nu", type=float, default=0.4, help="Poisson's ratio"
)
physical_parameters.add_argument(
    "--disp", type=float, default=-0.0001, help="Displacement in the y/z direction (2D/3D)"
)
fem_parameters = parser.add_argument_group("FEM parameters")
fem_parameters.add_argument(
    "--degree",
    dest="degree",
    type=int,
    default=1,
    help="Degree of primal and latent space",
)
fem_parameters.add_argument(
    "--quadrature-degree", type=int, default=4, help="Quadrature degree for integration"
)

newton_parameters = parser.add_argument_group("Newton solver parameters")
newton_parameters.add_argument(
    "--n-max-iterations",
    dest="newton_max_iterations",
    type=int,
    default=25,
    help="Maximum number of iterations of Newton iteration",
)
newton_parameters.add_argument(
    "--n-tol",
    dest="newton_tol",
    type=float,
    default=1e-6,
    help="Tolerance for Newton iteration",
)


llvp = parser.add_argument_group(
    title="Options for latent variable Proximal Point algorithm"
)
llvp.add_argument(
    "--max-iterations",
    dest="max_iterations",
    type=int,
    default=25,
    help="Maximum number of iterations of the Latent Variable Proximal Point algorithm",
)
llvp.add_argument(
    "--tol",
    type=float,
    default=1e-6,
    help="Tolerance for the Latent Variable Proximal Point algorithm",
)
alpha_options = parser.add_argument_group(
    title="Options for alpha-variable in Proximal Galerkin scheme"
)
alpha_options.add_argument(
    "--alpha_scheme",
    type=str,
    default="constant",
    choices=["constant", "linear", "doubling"],
    help="Scheme for updating alpha",
)
alpha_options.add_argument(
    "--alpha_0", type=float, default=1.0, help="Initial value of alpha"
)
alpha_options.add_argument(
    "--alpha_c", type=float, default=1.0, help="Increment of alpha in linear scheme"
)
mesh = parser.add_subparsers(
    dest="mesh", title="Parser for mesh options", required=True
)
built_in_parser = mesh.add_parser(
    "native", help="Use built-in mesh", formatter_class=CustomParser
)
built_in_parser.add_argument(
    "--nx", type=int, default=16, help="Number of elements in x-direction"
)
built_in_parser.add_argument(
    "--ny", type=int, default=7, help="Number of elements in y-direction"
)

load_mesh = mesh.add_parser(
    "file", help="Load mesh from file", formatter_class=CustomParser
)
load_mesh.add_argument("--filename", type=Path, help="Filename of mesh to load")
load_mesh.add_argument(
    "--contact-tag", dest="ct", type=int, default=2, help="Tag of contact surface"
)
load_mesh.add_argument(
    "--displacement-tag",
    dest="dt",
    type=int,
    default=1,
    help="Tag of displacement surface",
)
dst = dolfinx.default_scalar_type


def epsilon(w):
    return ufl.sym(ufl.grad(w))


def sigma(w, mu, lmbda):
    ew = epsilon(w)
    gdim = ew.ufl_shape[0]
    return 2.0 * mu * epsilon(w) + lmbda * ufl.tr(ufl.grad(w)) * ufl.Identity(gdim)


def solve_contact_problem(
    mesh: dolfinx.mesh.Mesh,
    facet_tag: dolfinx.mesh.MeshTags,
    boundary_conditions: dict[typing.Literal["contact", "displacement"], tuple[int]],
    degree: int,
    E: float,
    nu: float,
    disp: float,
    newton_max_its: int,
    newton_tol: float,
    max_iterations: int,
    alpha_scheme: AlphaScheme,
    alpha_0: float,
    alpha_c: float,
    tol: float,
    output: Path,
    quadrature_degree: int = 4,
):
    """
    Solve a contact problem with Signorini contact conditions using the
    Latent Variable Proximal Point algorithm.

    :param mesh: The mesh
    :param facet_tag: Mesh tags for facets
    :param boundary_conditions: Dictionary with boundary conditions mapping
        from type of boundary to values in `facet_tags`
    :param degree: Degree of primal and latent space
    :param E: Young's modulus
    :param nu: Poisson's ratio
    :param disp: Displacement in the y/z direction (2D/3D)
    :param newton_max_its: Maximum number of iterations in a Newton iteration
    :param newton_tol: Tolerance for Newton iteration
    :param max_iterations: Maximum number of iterations of
        the Latent Variable Proximal Point algorithm
    :param alpha_scheme: Scheme for updating alpha
    :param alpha_0: Initial value of alpha
    :param alpha_c: Increment of alpha in linear scheme
    :param tol: Tolerance for the Latent Variable Proximal Point algorithm
    :param quadrature_degree: Quadrature degree for integration
    """


    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    # NOTE: If we get facet-bubble spaces in DOLFINx we could use an alternative pair here
    gdim = mesh.geometry.dim
    element_u = basix.ufl.element(
        "Lagrange", mesh.topology.cell_name(), degree, shape=(gdim,)
    )
    V = dolfinx.fem.functionspace(mesh, element_u)
    u = dolfinx.fem.Function(V, name="displacement")

    v = ufl.TestFunction(V)

    element_p = basix.ufl.element("DG", mesh.topology.cell_name(), 0,
                                  shape=(mesh.geometry.dim, mesh.geometry.dim))
    W = dolfinx.fem.functionspace(mesh, element_p)

    v = ufl.TestFunction(V)
    psi = dolfinx.fem.Function(W)
    psi_k = dolfinx.fem.Function(W)
    w = ufl.TestFunction(W)

    alpha = dolfinx.fem.Constant(mesh, dst(alpha_0))

    f = dolfinx.fem.Constant(mesh, np.zeros(gdim, dtype=dst))
    dx = ufl.dx(domain=mesh)
    F00 = alpha * ufl.inner(sigma(u, mu, lmbda), epsilon(v)) * dx
    F00 -= alpha * ufl.inner(f, v) * dx

    F01 = ufl.inner(psi - psi_k, sigma(v, mu, lmbda)) * dx

    sigma_max = dolfinx.fem.Constant(mesh, dst(1.0))
    F10 = ufl.inner(sigma(u,mu, lmbda), w) * ufl.dx(domain=mesh)

    F11 = -ufl.inner(sigma_max*expm(psi), w) * ufl.dx(domain=mesh)
    F0 = F00 + F01
    F1 = F10 + F11

    F0 = F00 + F01
    F1 = F10 + F11
    residual_0 = dolfinx.fem.form(F0)
    residual_1 = dolfinx.fem.form(F1)
    jac00 = ufl.derivative(F0, u)
    jac01 = ufl.derivative(F0, psi)
    jac10 = ufl.derivative(F1, u)
    jac11 = ufl.derivative(F1, psi)
    J00 = dolfinx.fem.form(jac00)
    J01 = dolfinx.fem.form(jac01)
    J10 = dolfinx.fem.form(jac10)
    J11 = dolfinx.fem.form(jac11)

    J = [[J00, J01], [J10, J11]]
    F = [residual_0, residual_1]
    u_bc = dolfinx.fem.Function(V)
    def disp_func(x):
        values = np.zeros((gdim, x.shape[1]), dtype=dst)
        values[gdim - 1, :] = disp
        return values

    u_bc.interpolate(disp_func)
    _, V0_to_V = V.sub(gdim - 1).collapse()  # Used for partial loading in y/z direction
    disp_facets = [facet_tag.find(d) for d in boundary_conditions["displacement"]]
    bc_facets = np.unique(np.concatenate(disp_facets))
    bc = dolfinx.fem.dirichletbc(
        u_bc, dolfinx.fem.locate_dofs_topological(V, fdim, bc_facets)
    )

    def fixed_boundary(x):
        return np.zeros((gdim, x.shape[1]), dtype=dst)
    u_fixed = dolfinx.fem.Function(V)
    u_fixed.interpolate(fixed_boundary)
    fixed_facets = [facet_tag.find(d) for d in boundary_conditions["contact"]]
    bc_fixed_facets = np.unique(np.concatenate(fixed_facets))

    bc_fixed = dolfinx.fem.dirichletbc(
        u_fixed, dolfinx.fem.locate_dofs_topological(V, fdim, bc_fixed_facets))
    bcs = [bc, bc_fixed]

    solver = NewtonSolver(
        F,
        J,
        [u, psi],
        bcs=bcs,
        max_iterations=newton_max_its,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "superlu_dist",
            # "mat_mumps_icntl_14": 4000,
            # "mat_mumps_icntl_24": 1,
            "ksp_error_if_not_converged": True,
        },
        error_on_nonconvergence=True,
    )
    bp = dolfinx.io.VTXWriter(mesh.comm, output / "uh.bp", [u])

    diff = dolfinx.fem.Function(V)
    u_prev = dolfinx.fem.Function(V)
    for it in range(max_iterations):
        print(
            f"{it=}/{max_iterations}"
        )
        u_bc.x.array[V0_to_V] = disp  # (it+1)/M * disp

        if alpha_scheme == AlphaScheme.constant:
            pass
        elif alpha_scheme == AlphaScheme.linear:
            alpha.value = alpha_0 + alpha_c * it
        elif alpha_scheme == AlphaScheme.doubling:
            alpha.value = alpha_0 * 2**it
        print(alpha.value)
        solver_tol = np.sqrt(newton_tol) if it < 2 else newton_tol
        converged = solver.solve(solver_tol, 1)

        diff.x.array[:] = u.x.array - u_prev.x.array
        diff.x.petsc_vec.normBegin(2)
        normed_diff = diff.x.petsc_vec.normEnd(2)
        if normed_diff <= tol:
            print(f"Converged at {it=} with increment norm {normed_diff:.2e}<{tol:.2e}")
            break
        u_prev.x.array[:] = u.x.array
        psi_k.x.array[:] = psi.x.array


        bp.write(it)
        if not converged:
            print(f"Solver did not convert at {it=}, exiting with {converged=}")
            break

    if it == max_iterations - 1:
        print(f"Did not converge within {max_iterations} iterations")
    bp.close()


# python3 eig_exp.py native --nx 16 --ny 7 --output output_matrix
if __name__ == "__main__":
    args = parser.parse_args()
    if args.mesh == "native":

        def bottom_boundary(x):
            return np.isclose(x[1], 0.0)

        def top_boundary(x):
            return np.isclose(x[1], 1.0)

        mesh = dolfinx.mesh.create_unit_square(
            MPI.COMM_WORLD, args.nx, args.ny, dolfinx.mesh.CellType.quadrilateral
        )

        tdim = mesh.topology.dim
        fdim = tdim - 1
        top_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, top_boundary)
        contact_facets = dolfinx.mesh.locate_entities_boundary(
            mesh, fdim, bottom_boundary
        )
        assert len(np.intersect1d(top_facets, contact_facets)) == 0
        facet_map = mesh.topology.index_map(fdim)
        num_facets_local = facet_map.size_local + facet_map.num_ghosts
        values = np.zeros(num_facets_local, dtype=np.int32)
        values[top_facets] = 1
        values[contact_facets] = 2
        mt = dolfinx.mesh.meshtags(
            mesh, fdim, np.arange(num_facets_local, dtype=np.int32), values
        )
        bcs = {"contact": (2,), "displacement": (1,)}
    else:
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, args.filename, "r") as xdmf:
            mesh = xdmf.read_mesh()
            mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
            mt = xdmf.read_meshtags(mesh, name="Facet tags")
            bcs = {"contact": (args.ct,), "displacement": (args.dt,)}
        assert(mesh.geometry.dim == 2)
    solve_contact_problem(
        mesh=mesh,
        facet_tag=mt,
        boundary_conditions=bcs,
        degree=args.degree,
        E=args.E,
        nu=args.nu,
        disp=args.disp,
        newton_max_its=args.newton_max_iterations,
        newton_tol=args.newton_tol,
        max_iterations=args.max_iterations,
        alpha_scheme=AlphaScheme.from_string(args.alpha_scheme),
        alpha_0=args.alpha_0,
        alpha_c=args.alpha_c,
        tol=args.tol,
        output=args.output,
        quadrature_degree=args.quadrature_degree,
    )
