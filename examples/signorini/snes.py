from petsc4py import PETSc
import dolfinx
import typing
from pathlib import Path
import numpy as np
import basix
import ufl
from lvpp.problem import SNESProblem


__all__ = ["solve_snes_problem"]


def epsilon(w):
    return ufl.sym(ufl.grad(w))


def sigma(w, mu, lmbda):
    ew = epsilon(w)
    gdim = ew.ufl_shape[0]
    return 2.0 * mu * epsilon(w) + lmbda * ufl.tr(ufl.grad(w)) * ufl.Identity(gdim)


def solve_snes_problem(
    mesh: dolfinx.mesh.Mesh,
    facet_tag: dolfinx.mesh.MeshTags,
    boundary_conditions: dict[typing.Literal["contact", "displacement"], tuple[int]],
    E: float,
    nu: float,
    disp: float,
    max_iterations: int,
    tol: float,
    output: Path,
):
    """
    Solve a contact problem with Signorini contact conditions using the
    Latent Variable Proximal Point algorithm.

    :param mesh: The mesh
    :param facet_tag: Mesh tags for facets
    :param boundary_conditions: Dictionary with boundary conditions mapping
        from type of boundary to values in `facet_tags`
    :param E: Young's modulus
    :param nu: Poisson's ratio
    :param disp: Displacement in the y/z direction (2D/3D)
    :param max_iterations: Maximum number of iterations for SNES
    :param tol: Tolerance for the Latent Variable Proximal Point algorithm
    :param quadrature_degree: Quadrature degree for integration
    """

    fdim = mesh.topology.dim - 1

    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    # NOTE: If we get facet-bubble spaces in DOLFINx we could use an alternative pair here
    gdim = mesh.geometry.dim
    element_u = basix.ufl.element(
        "Lagrange", mesh.topology.cell_name(), 1, shape=(gdim,)
    )
    V = dolfinx.fem.functionspace(mesh, element_u)
    u = dolfinx.fem.Function(V, name="displacement")
    v = ufl.TestFunction(V)

    dst = dolfinx.default_scalar_type
    f = dolfinx.fem.Constant(mesh, np.zeros(gdim, dtype=dst))

    # Compute gap function
    def gap(x):
        return (np.zeros(x.shape[1]), x[1])

    gap_V = dolfinx.fem.Function(V)
    gap_V.interpolate(gap)

    F = ufl.inner(sigma(u, mu, lmbda), epsilon(v)) * ufl.dx(domain=mesh) - ufl.inner(
        f, v
    ) * ufl.dx(domain=mesh)

    residual = dolfinx.fem.form(F)
    jacobian = dolfinx.fem.form(ufl.derivative(F, u))

    u_bc = dolfinx.fem.Function(V)

    def disp_func(x):
        values = np.zeros((gdim, x.shape[1]), dtype=dst)
        values[gdim - 1, :] = disp
        return values

    u_bc.interpolate(disp_func)
    disp_facets = [facet_tag.find(d) for d in boundary_conditions["displacement"]]
    bc_facets = np.unique(np.concatenate(disp_facets))
    bc = dolfinx.fem.dirichletbc(
        u_bc, dolfinx.fem.locate_dofs_topological(V, fdim, bc_facets)
    )
    bcs = [bc]

    snes_options = {"snes_type": "vinewtonrsls", "snes_monitor": None}
    petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }  # Create semismooth Newton solver (SNES)

    snes = PETSc.SNES().create(comm=mesh.comm)  # type: ignore
    snes.setTolerances(tol, tol, tol, max_iterations)
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
    problem = SNESProblem(residual, u, bcs=bcs, J=jacobian)

    A = dolfinx.fem.petsc.create_matrix(jacobian)

    # Set solve functions and variable bounds
    snes.setFunction(problem.F, b_vec)
    snes.setJacobian(problem.J, A)

    # Compute gap function
    def gap(x):
        return (np.zeros(x.shape[1]), x[1])

    gap_Vh = dolfinx.fem.Function(V)
    gap_Vh.interpolate(gap)

    # Restrict gap constraint to boundary dofs that are not in bcs
    all_contact_facets = []
    for contact_marker in boundary_conditions["contact"]:
        all_contact_facets.append(facet_tag.find(contact_marker))
    contact_facets = np.unique(np.concatenate(all_contact_facets))

    bound_dofs = dolfinx.fem.locate_dofs_topological(
        V.sub(1), facet_tag.dim, contact_facets
    )

    # Compute lower and upper bounds
    lower_bound = dolfinx.fem.Function(V, name="lower_bound")
    upper_bound = dolfinx.fem.Function(V, name="upper_bound")
    lower_bound.x.array[:] = -PETSc.INFINITY
    lower_bound.x.array[bound_dofs] = -gap_Vh.x.array[bound_dofs]
    upper_bound.x.array[:] = PETSc.INFINITY

    snes.setVariableBounds(lower_bound.x.petsc_vec, upper_bound.x.petsc_vec)

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

    snes.solve(None, u.x.petsc_vec)
    assert snes.getConvergedReason() > 0

    mesh = u.function_space.mesh
    degree = mesh.geometry.cmap.degree
    V_out = dolfinx.fem.functionspace(mesh, ("Lagrange", degree, (mesh.geometry.dim,)))
    u_out = dolfinx.fem.Function(V_out, name="snes")
    u_out.interpolate(u)
    with dolfinx.io.XDMFFile(mesh.comm, output / "u_snes.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(u_out)
    ksp.destroy()
    snes.destroy()