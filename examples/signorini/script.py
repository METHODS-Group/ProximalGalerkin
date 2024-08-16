# Contact example
# SPDX-License-Identifier: MIT

from mpi4py import MPI
import dolfinx
import dolfinx.fem.petsc
import ufl
import numpy as np
from petsc4py import PETSc
import basix.ufl
import argparse
from enum import Enum
from pathlib import Path
import typing

class _HelpAction(argparse._HelpAction):
    """From https://stackoverflow.com/questions/20094215"""
    def __call__(self, parser, namespace, values, option_string=None):
        parser.print_help()

        # retrieve subparsers from parser
        subparsers_actions = [
            action for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)]
        # there will probably only be one subparser_action,
        # but better save than sorry
        for subparsers_action in subparsers_actions:
            # get all subparsers and print help
            for choice, subparser in subparsers_action.choices.items():
                print("Subparser '{}'".format(choice))
                print(subparser.format_help())

        parser.exit()


class CustomParser(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    ...


desc = (
    "Signorini contact problem solver\n\n"
    + "Uses the Latent Variable Proximal Point algorithm combined with a Newton solver at each step in the proximal point algorithm\n"
)
parser = argparse.ArgumentParser(description=desc, formatter_class=CustomParser, add_help=False)
parser.add_argument("--help", "-h", action=_HelpAction, help="show this help message and exit")
physical_parameters = parser.add_argument_group("Physical parameters")
physical_parameters.add_argument(
    "--E", dest="E", type=float, default=2.0e4, help="Young's modulus"
)
physical_parameters.add_argument(
    "--nu", dest="nu", type=float, default=0.3, help="Poisson's ratio"
)
physical_parameters.add_argument("--disp", type=float, default=-0.2, help="Displacement in the y/z direction (2D/3D)")
physical_parameters.add_argument("--gap", type=float, default=-0.05, help="y/z coordinate of rigid surface (2D/3D)")
fem_parameters = parser.add_argument_group("FEM parameters")
fem_parameters.add_argument(
    "--degree",
    dest="degree",
    type=int,
    default=1,
    help="Degree of primal and latent space",
)
fem_parameters.add_argument("--quadrature-degree", type=int, default=4, help="Quadrature degree for integration")

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
    default=10,
    help="Maximum number of iterations of the Latent Variable Proximal Point algorithm",
)
alpha_options = parser.add_argument_group(
    title="Options for alpha-variable in Proximal Galerkin scheme"
)
alpha_options.add_argument(
    "--alpha_scheme",
    type=str,
    default="linear",
    choices=["constant", "linear", "doubling"],
    help="Scheme for updating alpha",
)
alpha_options.add_argument(
    "--alpha_0", type=float, default=1.0, help="Initial value of alpha"
)
alpha_options.add_argument(
    "--alpha_c", type=float, default=1.0, help="Increment of alpha in linear scheme"
)
mesh = parser.add_subparsers(dest="mesh", title="Parser for mesh options", required=True)
built_in_parser = mesh.add_parser("native", help="Use built-in mesh", formatter_class=CustomParser)
built_in_parser.add_argument("--dim", type=int, default=3, choices=[2,3], help="Geometrical dimension of mesh")
built_in_parser.add_argument("--nx", type=int, default=16, help="Number of elements in x-direction")
built_in_parser.add_argument("--ny", type=int, default=7, help="Number of elements in y-direction")
built_in_parser.add_argument("--nz", type=int, default=5, help="Number of elements in z-direction")
load_mesh = mesh.add_parser("file", help="Load mesh from file", formatter_class=CustomParser)
load_mesh.add_argument("filename", type=Path, help="Filename of mesh to load")

dst = dolfinx.default_scalar_type


class AlphaScheme(Enum):
    constant = 1  # Constant alpha (alpha_0)
    linear = 2  # Linearly increasing alpha (alpha_0 + alpha_c * i) where i is the iteration number
    doubling = 3  # Doubling alpha (alpha_0 * 2^i) where i is the iteration number

    @classmethod
    def from_string(cls, method: str):
        if method == "constant":
            return AlphaScheme.constant
        elif method == "linear":
            return AlphaScheme.linear
        elif method == "doubling":
            return AlphaScheme.doubling
        else:
            raise ValueError(f"Unknown alpha scheme {method}")


class NewtonSolver:
    max_iterations: int
    bcs: list[dolfinx.fem.DirichletBC]
    A: PETSc.Mat
    b: PETSc.Vec
    J: dolfinx.fem.Form
    b: dolfinx.fem.Form
    dx: PETSc.Vec
    error_on_nonconvergence: bool

    def __init__(
        self,
        F: list[dolfinx.fem.form],
        J: list[list[dolfinx.fem.form]],
        w: list[dolfinx.fem.Function],
        bcs: list[dolfinx.fem.DirichletBC] | None = None,
        max_iterations: int = 5,
        petsc_options: dict[str, str | float | int | None] = None,
        error_on_nonconvergence: bool = True,
    ):
        self.max_iterations = max_iterations
        self.bcs = [] if bcs is None else bcs
        self.b = dolfinx.fem.petsc.create_vector_block(F)
        self.F = F
        self.J = J
        self.A = dolfinx.fem.petsc.create_matrix_block(J)
        self.dx = dolfinx.fem.petsc.create_vector_block(F)
        self.w = w
        self.x = dolfinx.fem.petsc.create_vector_block(F)
        self.norm_array = dolfinx.fem.Function(w[0].function_space)
        self.error_on_nonconvergence = error_on_nonconvergence
        # Set PETSc options
        opts = PETSc.Options()
        if petsc_options is not None:
            for k, v in petsc_options.items():
                opts[k] = v

        # Define KSP solver
        self._solver = PETSc.KSP().create(self.b.getComm().tompi4py())
        self._solver.setOperators(self.A)
        self._solver.setFromOptions()

        # Set matrix and vector PETSc options
        self.A.setFromOptions()
        self.b.setFromOptions()

    def _update_solution(self, beta):
        offset_start = 0
        for s in self.w:
            num_sub_dofs = (
                s.function_space.dofmap.index_map.size_local
                * s.function_space.dofmap.index_map_bs
            )
            s.x.array[:num_sub_dofs] -= (
                beta * self.dx.array_r[offset_start : offset_start + num_sub_dofs]
            )
            s.x.scatter_forward()
            offset_start += num_sub_dofs

    def solve(self, tol=1e-6, beta=1.0):
        i = 0

        while i < self.max_iterations:
            # Assemble F(u_{i-1}) - J(u_D - u_{i-1}) and set du|_bc= u_D - u_{i-1}
            with self.b.localForm() as b_loc:
                b_loc.set(0)
            dolfinx.fem.petsc.assemble_vector_block(
                self.b, self.F, self.J, bcs=self.bcs, x0=self.x, scale=-1.0
            )
            self.b.ghostUpdate(
                PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD
            )

            # Assemble Jacobian
            self.A.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix_block(self.A, self.J, bcs=self.bcs)
            self.A.assemble()
            with self.dx.localForm() as dx_loc:
                dx_loc.set(0)
            self._solver.solve(self.b, self.dx)
            self.dx.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )
            self._update_solution(beta)

            if self.error_on_nonconvergence:
                assert (
                    self._solver.getConvergedReason() > 0
                ), "Linear solver did not converge"
            else:
                converged = self._solver.getConvergedReason()
                import warnings

                if converged <= 0:
                    warnings.warn(
                        "Linear solver did not converge, exiting", RuntimeWarning
                    )
                    return 0
            # Compute norm of primal space diff
            dolfinx.cpp.la.petsc.scatter_local_vectors(
                self.x,
                [si.x.petsc_vec.array_r for si in self.w],
                [
                    (
                        si.function_space.dofmap.index_map,
                        si.function_space.dofmap.index_map_bs,
                    )
                    for si in self.w
                ],
            )

            self.x.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )

            local_du, _ = dolfinx.cpp.la.petsc.get_local_vectors(
                self.dx,
                [
                    (
                        si.function_space.dofmap.index_map,
                        si.function_space.dofmap.index_map_bs,
                    )
                    for si in self.w
                ],
            )

            self.norm_array.x.array[:] = local_du
            self.norm_array.x.petsc_vec.ghostUpdate(
                PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD
            )
            self.norm_array.x.petsc_vec.normBegin(1)
            correction_norm = self.norm_array.x.petsc_vec.normEnd(1)

            print(f"Iteration {i}: Correction norm {correction_norm}")
            if correction_norm < tol:
                break
            i += 1
        return 1




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
    quadrature_degree: int = 4,
):
    """
    Solve a contact problem with Signorini contact conditions using the Latent Variable Proximal Point algorithm

    :param mesh: The mesh
    :param facet_tag: Mesh tags for facets
    :param boundary_conditions: Dictionary with boundary conditions mapping from type of boundary to values in `facet_tags`
    :param degree: Degree of primal and latent space
    :param E: Young's modulus
    :param nu: Poisson's ratio
    :param gap: y/z coordinate of rigid surface (2D/3D)
    :param disp: Displacement in the y/z direction (2D/3D)
    :param newton_max_its: Maximum number of iterations in a Newton iteration
    :param newton_tol: Tolerance for Newton iteration
    :param max_iterations: Maximum number of iterations of the Latent Variable Proximal Point algorithm
    :param alpha_scheme: Scheme for updating alpha
    :param alpha_0: Initial value of alpha
    :param alpha_c: Increment of alpha in linear scheme
    :param quadrature_degree: Quadrature degree for integration
    """

    all_contact_facets = []
    for contact_marker in boundary_conditions["contact"]:
        all_contact_facets.append(facet_tag.find(contact_marker))
    contact_facets = np.unique(np.concatenate(all_contact_facets))

    fdim = mesh.topology.dim - 1
    submesh, submesh_to_mesh = dolfinx.mesh.create_submesh(
        mesh, fdim, contact_facets)[0:2]

    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    # NOTE: If we get facet-bubble spaces in DOLFINx we could use an alternative pair here
    element_u = basix.ufl.element(
        "Lagrange", mesh.topology.cell_name(), degree, shape=(mesh.geometry.dim,)
    )
    V = dolfinx.fem.functionspace(mesh, element_u)

    u = dolfinx.fem.Function(V, name="displacement")
    v = ufl.TestFunction(V)

    element_p = basix.ufl.element("Lagrange", submesh.topology.cell_name(), degree)
    W = dolfinx.fem.functionspace(submesh, element_p)

    v = ufl.TestFunction(V)
    psi = dolfinx.fem.Function(W)
    psi_k = dolfinx.fem.Function(W)
    w = ufl.TestFunction(W)

    facet_imap = mesh.topology.index_map(fdim)
    num_facets = facet_imap.size_local + facet_imap.num_ghosts
    mesh_to_submesh = np.full(num_facets, -1)
    mesh_to_submesh[submesh_to_mesh] = np.arange(len(submesh_to_mesh))
    entity_maps = {submesh: mesh_to_submesh}

    metadata = {"quadrature_degree": quadrature_degree}
    ds = ufl.Measure(
        "ds", domain=mesh, subdomain_data=facet_tag, subdomain_id=boundary_conditions["contact"],
        metadata=metadata
    )
    n = ufl.FacetNormal(mesh)
    gdim = mesh.geometry.dim
    n_g = dolfinx.fem.Constant(mesh, np.zeros(gdim, dtype=dst))
    n_g.value[-1] = -1

    alpha = dolfinx.fem.Constant(mesh, dst(alpha_0))
    
    f = dolfinx.fem.Constant(mesh, np.zeros(gdim, dtype=dst))
    x = ufl.SpatialCoordinate(mesh)
    g = x[gdim-1] + dolfinx.fem.Constant(mesh, dst(-gap))

    F00 = alpha * ufl.inner(sigma(u, mu, lmbda), epsilon(v)) * ufl.dx(
        domain=mesh
    ) - alpha * ufl.inner(f, v) * ufl.dx(domain=mesh)
    F01 = -ufl.inner(psi - psi_k, ufl.dot(v, n)) * ds
    F10 = ufl.inner(ufl.dot(u, n_g), w) * ds
    F11 = ufl.inner(ufl.exp(psi), w) * ds - ufl.inner(g, w) * ds

    F0 = F00 + F01
    F1 = F10 + F11

    F0 = F00 + F01
    F1 = F10 + F11
    residual_0 = dolfinx.fem.form(F0, entity_maps=entity_maps)
    residual_1 = dolfinx.fem.form(F1, entity_maps=entity_maps)
    jac00 = ufl.derivative(F0, u)
    jac01 = ufl.derivative(F0, psi)
    jac10 = ufl.derivative(F1, u)
    jac11 = ufl.derivative(F1, psi)
    J00 = dolfinx.fem.form(jac00, entity_maps=entity_maps)
    J01 = dolfinx.fem.form(jac01, entity_maps=entity_maps)
    J10 = dolfinx.fem.form(jac10, entity_maps=entity_maps)
    J11 = dolfinx.fem.form(jac11, entity_maps=entity_maps)

    J = [[J00, J01], [J10, J11]]
    F = [residual_0, residual_1]

    u_bc = dolfinx.fem.Function(V)

    def disp_func(x):
        values = np.zeros((gdim, x.shape[1]), dtype=dst)
        values[gdim-1, :] = disp
        return values
    u_bc.interpolate(disp_func)
    V0, V0_to_V = V.sub(gdim-1).collapse()
    disp_facets = [facet_tag.find(d) for d in boundary_conditions["displacement"]]
    bc_facets = np.unique(np.concatenate(disp_facets))
    print(len(bc_facets), MPI.COMM_WORLD.allreduce(len(bc_facets), op=MPI.SUM))
    bc = dolfinx.fem.dirichletbc(
        u_bc, dolfinx.fem.locate_dofs_topological(V, fdim, bc_facets)
    )
    bcs = [bc]

    A =  dolfinx.fem.petsc.create_matrix_block(J)
    solver = NewtonSolver(
        F,
        J,
        [u, psi],
        bcs=bcs,
        max_iterations=newton_max_its,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        error_on_nonconvergence=True,
    )

    bp = dolfinx.io.VTXWriter(mesh.comm, "uh.bp", [u])
    bp_psi = dolfinx.io.VTXWriter(mesh.comm, "psi.bp", [psi])
    print(max_iterations)
    for it in range(max_iterations):
        print(f"{it=}")
        u_bc.x.array[V0_to_V] = disp  # (it+1)/M * disp

        if alpha_scheme == AlphaScheme.constant:
            pass
        elif alpha_scheme == AlphaScheme.linear:
            alpha.value = alpha_0 + alpha_c * it
        elif alpha_scheme == AlphaScheme.doubling:
            alpha.value = alpha_0 * 2**it

        converged = solver.solve(newton_tol, 1)

        psi_k.x.array[:] = psi.x.array
        bp.write(it)
        bp_psi.write(it)
        if not converged:
            print(f"Solver did not convert at {it=}, exiting")
            break

    bp.close()


if __name__ == "__main__":
    args = parser.parse_args()
    if args.mesh == "native":
        def bottom_boundary(x):
            return np.isclose(x[args.dim-1], 0.0)


        def top_boundary(x):
            return np.isclose(x[args.dim-1], 1.0)

        if args.dim == 3:
            mesh = dolfinx.mesh.create_unit_cube(
                MPI.COMM_WORLD, args.nx, args.ny, args.nz, dolfinx.mesh.CellType.hexahedron
            )
        elif args.dim == 2:
            mesh= dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, args.nx, args.ny, dolfinx.mesh.CellType.quadrilateral)
    
        tdim = mesh.topology.dim
        fdim = tdim -1
        top_facets = dolfinx.mesh.locate_entities_boundary(
            mesh, fdim, top_boundary
        )
        contact_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, bottom_boundary)
        assert len(np.intersect1d(top_facets, contact_facets)) == 0
        facet_map = mesh.topology.index_map(fdim)
        num_facets_local = facet_map.size_local + facet_map.num_ghosts
        values = np.zeros(num_facets_local, dtype=np.int32)
        values[top_facets] = 1
        values[contact_facets] = 2
        mt = dolfinx.mesh.meshtags(
            mesh,
            fdim,
            np.arange(num_facets_local, dtype=np.int32),
            values
        )
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "facet_tags.xdmf", "w") as xdmf:
            xdmf.write_mesh(mesh)
            xdmf.write_meshtags(mt, mesh.geometry)
        bcs = {"contact": (2,), "displacement": (1,)}
    else:
        raise NotImplementedError("Only built-in meshes are supported")

    solve_contact_problem(
        mesh = mesh,
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
        alpha_scheme=AlphaScheme.from_string(args.alpha_scheme),
        alpha_0=args.alpha_0,
        alpha_c=args.alpha_c,
        quadrature_degree=args.quadrature_degree
    )
