"""
Solve Signorini contact problem using the Latent Variable Proximal Point algorithm

Author: Jørgen S. Dokken
SPDX-License-Identifier: MIT
"""

import argparse
import typing
from pathlib import Path

from mpi4py import MPI

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl
from packaging.version import Version
from scifem import BlockedNewtonSolver

AlphaScheme = typing.Literal["constant", "linear", "doubling"]


class _HelpAction(argparse._HelpAction):
    """From https://stackoverflow.com/questions/20094215"""

    def __call__(self, parser, namespace, values, option_string=None):
        parser.print_help()

        # retrieve subparsers from parser
        subparsers_actions = [
            action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
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
parser = argparse.ArgumentParser(description=desc, formatter_class=CustomParser, add_help=False)
parser.add_argument("--help", "-h", action=_HelpAction, help="show this help message and exit")
parser.add_argument("--output", "-o", type=Path, default=Path("output"), help="Output directory")
physical_parameters = parser.add_argument_group("Physical parameters")
physical_parameters.add_argument("--E", dest="E", type=float, default=2.0e4, help="Young's modulus")
physical_parameters.add_argument("--nu", dest="nu", type=float, default=0.3, help="Poisson's ratio")
physical_parameters.add_argument(
    "--disp", type=float, default=-0.25, help="Displacement in the y/z direction (2D/3D)"
)
physical_parameters.add_argument(
    "--gap", type=float, default=-0.00, help="y/z coordinate of rigid surface (2D/3D)"
)
fem_parameters = parser.add_argument_group("FEM parameters")
fem_parameters.add_argument(
    "--degree",
    dest="degree",
    type=int,
    default=2,
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
    default=250,
    help="Maximum number of iterations of Newton iteration",
)
newton_parameters.add_argument(
    "--n-tol",
    dest="newton_tol",
    type=float,
    default=1e-6,
    help="Tolerance for Newton iteration",
)


llvp = parser.add_argument_group(title="Options for latent variable Proximal Point algorithm")
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
    default="doubling",
    choices=typing.get_args(AlphaScheme),
    help="Scheme for updating alpha",
)
alpha_options.add_argument("--alpha_0", type=float, default=1.0, help="Initial value of alpha")
alpha_options.add_argument(
    "--alpha_c", type=float, default=1.0, help="Increment of alpha in linear scheme"
)
mesh = parser.add_subparsers(dest="mesh", title="Parser for mesh options", required=True)
built_in_parser = mesh.add_parser("native", help="Use built-in mesh", formatter_class=CustomParser)
built_in_parser.add_argument(
    "--dim", type=int, default=3, choices=[2, 3], help="Geometrical dimension of mesh"
)
built_in_parser.add_argument("--nx", type=int, default=16, help="Number of elements in x-direction")
built_in_parser.add_argument("--ny", type=int, default=7, help="Number of elements in y-direction")
built_in_parser.add_argument("--nz", type=int, default=5, help="Number of elements in z-direction")
load_mesh = mesh.add_parser("file", help="Load mesh from file", formatter_class=CustomParser)
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
    gap: float,
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
    :param gap: y/z coordinate of rigid surface (2D/3D)
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

    all_contact_facets = []
    for contact_marker in boundary_conditions["contact"]:
        all_contact_facets.append(facet_tag.find(contact_marker))
    contact_facets = np.unique(np.concatenate(all_contact_facets))

    gdim = mesh.geometry.dim
    fdim = mesh.topology.dim - 1
    # Create submesh for potential facets
    submesh, submesh_to_mesh = dolfinx.mesh.create_submesh(mesh, fdim, contact_facets)[0:2]

    # Invert map to get mapping from parent facet to submesh cell
    facet_imap = mesh.topology.index_map(fdim)
    num_facets = facet_imap.size_local + facet_imap.num_ghosts
    mesh_to_submesh = np.full(num_facets, -1)
    mesh_to_submesh[submesh_to_mesh] = np.arange(len(submesh_to_mesh))
    entity_maps = {submesh: mesh_to_submesh}

    # Define integration measure only on potential contact facets
    metadata = {"quadrature_degree": quadrature_degree}
    ds = ufl.Measure(
        "ds",
        domain=mesh,
        subdomain_data=facet_tag,
        subdomain_id=boundary_conditions["contact"],
        metadata=metadata,
    )

    # Create mixed finite element space
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", degree, (gdim,)))
    W = dolfinx.fem.functionspace(submesh, ("Lagrange", degree))
    if Version(dolfinx.__version__) < Version("0.9.0"):
        raise RuntimeError("This script requires dolfinx version 0.9.0 or later")
    Q = ufl.MixedFunctionSpace(V, W)

    # Define primal and latent variable + test functions
    v, w = ufl.TestFunctions(Q)
    u = dolfinx.fem.Function(V, name="displacement")
    psi = dolfinx.fem.Function(W)
    psi_k = dolfinx.fem.Function(W)

    # Define problem specific parameters
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    n_g = dolfinx.fem.Constant(mesh, np.zeros(gdim, dtype=dst))
    n_g.value[-1] = -1
    alpha = dolfinx.fem.Constant(mesh, dst(alpha_0))
    f = dolfinx.fem.Constant(mesh, np.zeros(gdim, dtype=dst))
    x = ufl.SpatialCoordinate(mesh)
    g = x[gdim - 1] + dolfinx.fem.Constant(mesh, dst(-gap))

    # Set up residual
    residual = alpha * ufl.inner(sigma(u, mu, lmbda), epsilon(v)) * ufl.dx(
        domain=mesh
    ) - alpha * ufl.inner(f, v) * ufl.dx(domain=mesh)
    residual += -ufl.inner(psi - psi_k, ufl.dot(v, n_g)) * ds
    residual += ufl.inner(ufl.dot(u, n_g), w) * ds
    residual += ufl.inner(ufl.exp(psi), w) * ds - ufl.inner(g, w) * ds

    # Compile residual
    F = dolfinx.fem.form(ufl.extract_blocks(residual), entity_maps=entity_maps)

    # Set up Jacobian
    jac = ufl.derivative(residual, [u, psi], ufl.TrialFunctions(Q))

    # Compile Jacobian
    J = dolfinx.fem.form(ufl.extract_blocks(jac), entity_maps=entity_maps)
    u_bc = dolfinx.fem.Function(V)

    def disp_func(x):
        values = np.zeros((gdim, x.shape[1]), dtype=dst)
        values[gdim - 1, :] = disp
        return values

    # Set up Dirichlet conditions
    u_bc.interpolate(disp_func)
    # Used for partial loading in y/z direction
    _, V0_to_V = V.sub(gdim - 1).collapse()
    disp_facets = [facet_tag.find(d) for d in boundary_conditions["displacement"]]
    bc_facets = np.unique(np.concatenate(disp_facets))
    bc = dolfinx.fem.dirichletbc(u_bc, dolfinx.fem.locate_dofs_topological(V, fdim, bc_facets))
    bcs = [bc]

    # Set up solver
    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        # "mat_mumps_icntl_14": 120,
        "ksp_error_if_not_converged": True,
    }
    solver = BlockedNewtonSolver(F, [u, psi], bcs=bcs, J=J, petsc_options=petsc_options)
    solver.max_iter = newton_max_its
    violation = dolfinx.fem.Function(V)
    bp = dolfinx.io.VTXWriter(mesh.comm, output / "uh.bp", [u, violation])
    bp_psi = dolfinx.io.VTXWriter(mesh.comm, output / "psi.bp", [psi])
    V_DG = dolfinx.fem.functionspace(mesh, ("DG", degree, (mesh.geometry.dim,)))
    stresses = dolfinx.fem.Function(V_DG, name="VonMises")
    u_dg = dolfinx.fem.Function(V_DG, name="u")
    bp_vonmises = dolfinx.io.VTXWriter(mesh.comm, output / "von_mises.bp", [stresses, u_dg])
    s = sigma(u, mu, lmbda) - 1.0 / 3 * ufl.tr(sigma(u, mu, lmbda)) * ufl.Identity(len(u))
    von_Mises = ufl.sqrt(3.0 / 2 * ufl.inner(s, s))
    stress_expr = dolfinx.fem.Expression(von_Mises, V_DG.element.interpolation_points())

    u_prev = dolfinx.fem.Function(V)
    diff = dolfinx.fem.Function(V)
    normed_diff = -1.0
    displacement = ufl.inner(u, n_g) - g
    expr = dolfinx.fem.Expression(displacement, V.element.interpolation_points())
    penetration = ufl.conditional(ufl.gt(displacement, 0), displacement, 0)
    boundary_penetration = dolfinx.fem.form(ufl.inner(penetration, penetration) * ds)

    def assemble_penetration():
        local_penetration = dolfinx.fem.assemble_scalar(boundary_penetration)
        return np.sqrt(mesh.comm.allreduce(local_penetration, op=MPI.SUM))

    iterations = []
    solver.error_on_nonconvergence = True
    solver.convergence_criterion = "residual"
    for it in range(1, max_iterations + 1):
        print(
            f"{it=}/{max_iterations} {normed_diff:.2e} Penetration L2(Gamma):",
            f" {assemble_penetration():.2e}",
        )
        u_bc.x.array[V0_to_V] = disp  # (it+1)/M * disp

        if alpha_scheme == "constant":
            pass
        elif alpha_scheme == "linear":
            alpha.value = alpha_0 + alpha_c * it
        elif alpha_scheme == "doubling":
            alpha.value = alpha_0 * 2**it

        solver_tol = 10 * newton_tol if it < 2 else newton_tol
        solver.atol = solver_tol
        solver.rtol = solver_tol
        num_its, converged = solver.solve()
        iterations.append(num_its)
        diff.x.array[:] = u.x.array - u_prev.x.array
        diff.x.petsc_vec.normBegin(2)
        normed_diff = diff.x.petsc_vec.normEnd(2)
        if normed_diff <= tol:
            print(f"Converged at {it=} with increment norm {normed_diff:.2e}<{tol:.2e}")
            break
        u_prev.x.array[:] = u.x.array
        psi_k.x.array[:] = psi.x.array

        stresses.sub(0).interpolate(stress_expr)
        u_dg.interpolate(u)
        bp_vonmises.write(it)
        violation.sub(0).interpolate(expr)
        bp.write(it)
        bp_psi.write(it)

        if not converged:
            print(
                f"Solver did not convert at {
                  it=}, exiting with {converged=}"
            )
            break
    if it == max_iterations - 1:
        print(f"Did not converge within {max_iterations} iterations")
    bp_psi.close()
    bp.close()
    bp_vonmises.close()
    num_dofs_u = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_global
    print(f"{num_dofs_u=}, {num_cells=}")

    return it, iterations


if __name__ == "__main__":
    args = parser.parse_args()
    if args.mesh == "native":

        def bottom_boundary(x):
            return np.isclose(x[args.dim - 1], 0.0)

        def top_boundary(x):
            return np.isclose(x[args.dim - 1], 1.0)

        if args.dim == 3:
            mesh = dolfinx.mesh.create_unit_cube(
                MPI.COMM_WORLD,
                args.nx,
                args.ny,
                args.nz,
                dolfinx.mesh.CellType.hexahedron,
            )
        elif args.dim == 2:
            mesh = dolfinx.mesh.create_unit_square(
                MPI.COMM_WORLD, args.nx, args.ny, dolfinx.mesh.CellType.quadrilateral
            )

        tdim = mesh.topology.dim
        fdim = tdim - 1
        top_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, top_boundary)
        contact_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, bottom_boundary)
        assert len(np.intersect1d(top_facets, contact_facets)) == 0
        facet_map = mesh.topology.index_map(fdim)
        num_facets_local = facet_map.size_local + facet_map.num_ghosts
        values = np.zeros(num_facets_local, dtype=np.int32)
        values[top_facets] = 1
        values[contact_facets] = 2
        mt = dolfinx.mesh.meshtags(mesh, fdim, np.arange(num_facets_local, dtype=np.int32), values)
        bcs = {"contact": (2,), "displacement": (1,)}
    else:
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, args.filename, "r") as xdmf:
            mesh = xdmf.read_mesh()
            mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
            mt = xdmf.read_meshtags(mesh, name="Facet tags")
            bcs = {"contact": (args.ct,), "displacement": (args.dt,)}

    it, iterations = solve_contact_problem(
        mesh=mesh,
        facet_tag=mt,
        boundary_conditions=bcs,
        degree=args.degree,
        E=args.E,
        nu=args.nu,
        gap=args.gap,
        disp=args.disp,
        newton_max_its=args.newton_max_iterations,
        newton_tol=args.newton_tol,
        max_iterations=args.max_iterations,
        alpha_scheme=args.alpha_scheme,
        alpha_0=args.alpha_0,
        alpha_c=args.alpha_c,
        tol=args.tol,
        output=args.output,
        quadrature_degree=args.quadrature_degree,
    )
    print(it, iterations, sum(iterations), min(iterations), max(iterations))
    assert it == len(iterations)
