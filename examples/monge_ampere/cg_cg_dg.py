import numpy as np
import scipy as sc
from expm import expm
from firedrake import *

nonsmooth = False
z_prev = None
base = SquareMesh(2, 2, 2, quadrilateral=False)
mh = MeshHierarchy(base, 3)
for mesh in mh:
    mesh.coordinates.dat.data[:] -= 1

projsp = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}

errors = []
for mesh in mh:
    X = (x, y) = SpatialCoordinate(mesh)
    n = mesh.geometric_dimension()

    if nonsmooth:
        u_exact = conditional(
            le(dot(X, X), 1 / 4), 2 * dot(X, X), 2 * dot(X, X) + 2 * (sqrt(dot(X, X)) - 1 / 2) ** 2
        )
    else:
        u_exact = exp(dot(X, X) / 2)

    def hessian(u):
        return as_tensor([[u.dx(0).dx(0), u.dx(0).dx(1)], [u.dx(1).dx(0), u.dx(1).dx(1)]])

    rho = det(hessian(u_exact))  # forcing term
    g = u_exact  # boundary data

    k = 6
    V = FunctionSpace(mesh, "CG", k)
    U = VectorFunctionSpace(mesh, "CG", k + 1)
    W = VectorFunctionSpace(mesh, "DG", k, dim=3)

    C = FunctionSpace(mesh, "DG", W.ufl_element().degree())
    rho_c = Function(C, name="Density").interpolate(rho)
    VTKFile("output/density.pvd").write(rho_c)
    assert all(rho_c.dat.data > 0)

    Z = MixedFunctionSpace([V, U, W])
    z = Function(Z)
    print(f"# degrees of freedom: {Z.dim()} {tuple(W.dim() for W in Z)}", flush=True)

    (u, p, Psi) = split(z)
    psi = as_tensor([[Psi[0], Psi[1]], [Psi[1], Psi[2]]])
    (v, q, Phi) = split(TestFunction(Z))
    phi = as_tensor([[Phi[0], Phi[1]], [Phi[1], Phi[2]]])

    F = (
        inner(tr(psi) - ln(rho), v) * dx
        + inner(p, q) * dx
        - inner(grad(u), q) * dx
        + inner(grad(p), phi) * dx
        - inner(expm(psi), phi) * dx
    )

    bc = DirichletBC(Z.sub(0), g, "on_boundary")

    # Construct initial guesses
    z.subfunctions[0].rename("Solution")
    z.subfunctions[1].rename("GradSolution")
    z.subfunctions[2].rename("LatentVariable")

    if z_prev is not None:
        prolong(z_prev, z)
    else:
        if nonsmooth:
            u_guess = 2 * dot(X, X)
        else:
            u_guess = x**2 + y**2

        B = TensorFunctionSpace(mesh, "DG", W.ufl_element().degree())
        z.subfunctions[0].project(u_guess, solver_parameters=None)
        z.subfunctions[1].project(grad(u_guess), solver_parameters=None)
        psi_init = Function(B).interpolate(hessian(z.subfunctions[0]))
        # Should be able to vectorize this, but not done natively
        loghessian = Function(B, name="LogHessian")
        for i in range(loghessian.dat.data.shape[0]):
            loghessian.dat.data[i, :, :] = sc.linalg.logm(psi_init.dat.data[i, :, :])
        z.subfunctions[2].project(as_vector([loghessian[0, 0], loghessian[0, 1], loghessian[1, 1]]))
        VTKFile("output/initialguess.pvd").write(
            z.subfunctions[0], z.subfunctions[1], z.subfunctions[2]
        )

    sp = {
        "snes_monitor": None,
        "snes_linesearch_type": "l2",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "pc_svd_monitor": None,
        "mat_mumps_icntl_14": 10000,
    }
    solve(F == 0, z, bcs=bc, solver_parameters=sp)
    z_prev = z

    errors.append(norm((u_exact - u), "L2"))
    print("||skew(psi)||_L2: ", norm(0.5 * (psi - psi.T)), flush=True)
error_ = project(u_exact - u, V, solver_parameters=projsp)
u_exact_ = project(u_exact, V, solver_parameters=projsp)
error_.rename("Error")
u_exact_.rename("ExactSolution")
VTKFile("output/solution.pvd").write(
    z.subfunctions[0], z.subfunctions[1], z.subfunctions[2], u_exact_, error_
)

print("Errors: ", errors, flush=True)
convergence_orders = lambda x: np.log2(np.array(x)[:-1] / np.array(x)[1:])
print("Convergence orders: ", convergence_orders(errors), flush=True)
