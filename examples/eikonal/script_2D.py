from __future__ import annotations
import dolfinx.fem.petsc
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
import basix.ufl
import ufl
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem.petsc import NonlinearProblem
from pathlib import Path
opts = PETSc.Options()  # type: ignore
opts["ksp_type"] = "preonly"
opts["pc_type"] = "lu"
opts["pc_factor_mat_solver_type"] = "mumps"
# opts[f"{option_prefix}pc_factor_reuse_ordering"] = True


def solve_problem(N:int, M:int, L:int, H:int, degree: int,
                  cell_type:dolfinx.mesh.CellType=dolfinx.mesh.CellType.triangle,
                  quadrature_degree:int=15,
                  tol:float=1e-6, max_iter:int=20) -> tuple[float, float, int]:
    """
    Solve 2D eikonal equation on a [0, 0]x[L, H] domain with N x M elements
    using a broken Lagrange space of degree `degree` for the primal variable
    """
    if cell_type == dolfinx.mesh.CellType.triangle:
        mesh = dolfinx.mesh.create_rectangle(
            MPI.COMM_WORLD, [[0, 0], [L, H]], [N, M], cell_type=cell_type, diagonal=dolfinx.mesh.DiagonalType.crossed)
    else:

        mesh = dolfinx.mesh.create_rectangle(
            MPI.COMM_WORLD, [[0, 0], [L, H]], [N, M], cell_type=cell_type)

    el_0 = basix.ufl.element("DG", mesh.topology.cell_name(), degree)
    el_1 = basix.ufl.element(
        "RT", mesh.topology.cell_name(), degree+1)
    trial_el = basix.ufl.mixed_element([el_0, el_1])
    V_trial = dolfinx.fem.functionspace(mesh, trial_el)
    V_test = V_trial

    w = dolfinx.fem.Function(V_trial)
    _, U_to_W = V_trial.sub(0).collapse()
    u, psi = ufl.split(w)

    v, tau = ufl.TestFunctions(V_test)

    metadata = {"quadrature_degree": quadrature_degree}
    dx = ufl.Measure("dx",  domain=mesh, metadata=metadata)
    f = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1))

    alpha = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1))
    phi = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1))
    w0 = dolfinx.fem.Function(V_trial)
    _, psi0 = ufl.split(w0)

    F = ufl.inner(ufl.div(psi), v)*dx
    F -= ufl.inner(ufl.div(psi0), v)*dx
    F -= alpha * ufl.inner(f, v) * dx

    non_lin_term = 1/(ufl.sqrt(1 + ufl.dot(psi, psi)))
    F += ufl.inner(u, ufl.div(tau)) * dx
    F += phi * non_lin_term * ufl.dot(psi, tau)*dx


    J = ufl.derivative(F, w)

    problem = NonlinearProblem(F, w, bcs=[], J=J)
    solver = NewtonSolver(mesh.comm, problem)
    solver.convergence_criterion = "residual"
    solver.rtol = 1e-6
    solver.atol = 1e-6
    solver.max_it = 10
    solver.relaxation_parameter = 1
    solver.error_on_nonconvergence = True

    
    ksp = solver.krylov_solver
    ksp.setOptionsPrefix("")
    ksp.setFromOptions()


    u_out = w.sub(0).collapse()
    u_out.name = "u"

    dir = Path("output")
    dir.mkdir(exist_ok=True)
    bp_u = dolfinx.io.VTXWriter(mesh.comm, dir/f"u_{N}_{M}_{L}_{H}_{degree}_{quadrature_degree}_{mesh.basix_cell()}.bp", [u_out], engine="BP4")


    diff = w.sub(0)-w0.sub(0)
    L2_squared = ufl.dot(diff, diff)*dx
    compiled_diff = dolfinx.fem.form(L2_squared)
    num_total_iterations = 0
    for i in range(max_iter):

        alpha.value += 2**i

        dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
        num_newton_iterations, converged = solver.solve(w)
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)
        num_total_iterations += num_newton_iterations
        print(
            f"Iteration {i}: {converged=} {num_newton_iterations=} {ksp.getConvergedReason()=}")
        local_diff = dolfinx.fem.assemble_scalar(compiled_diff)
        global_diff = np.sqrt(mesh.comm.allreduce(local_diff, op=MPI.SUM))
        print(f"|delta u |= {global_diff}")

        # Update w0
        w0.x.array[:] = w.x.array

        # Write solution to file
        u_out.x.array[:] = w.sub(0).x.array[U_to_W]
        bp_u.write(i)

        if global_diff < tol:
            break
    mesh.comm.Barrier()
    bp_u.close()

    x = ufl.SpatialCoordinate(mesh)
    dist_x = ufl.min_value(abs(x[0]), abs(L-x[0]))
    dist_y = ufl.min_value(abs(x[1]), abs(H-x[1]))
    dist = ufl.min_value(dist_x, dist_y)
    
    diff = abs(w.sub(0)) - dist
    error = dolfinx.fem.form(ufl.inner(diff, diff) * dx)
    local_error = dolfinx.fem.assemble_scalar(error)
    global_error = np.sqrt(mesh.comm.allreduce(local_error, op=MPI.SUM))

    num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
    local_max = np.max(mesh.h(mesh.topology.dim, np.arange(num_cells_local, dtype=np.int32)))
    global_h =  mesh.comm.allreduce(local_max, op=MPI.MAX)

    expr = dolfinx.fem.Expression(dist, u_out.function_space.element.interpolation_points())
    u_exact = dolfinx.fem.Function(u_out.function_space)
    u_exact.interpolate(expr)
    print(max(u_exact.x.array-abs(u_out.x.array)))
    # with dolfinx.io.VTXWriter(mesh.comm, "exact.bp", [u_exact], engine="BP4") as writer:
    #     writer.write(0)

    print(global_h, degree, np.sqrt(global_error), num_total_iterations)
    ksp.destroy()
    return global_h, np.sqrt(global_error), num_total_iterations



if __name__ == "__main__":
    Ns = [2,4,8, 16, 32, 64]
    degrees = [1,3]
    L = 1
    H = 1
    errors = np.zeros((len(degrees),len(Ns), ))
    hs = np.zeros((len(degrees), len(Ns)))
    total_iterations = np.zeros((len(degrees), len(Ns)), dtype=np.int32)
    for j, degree in enumerate(degrees):
        for i, N in enumerate(Ns):
            hs[j, i], errors[j, i], total_iterations[j, i]  = solve_problem(N, N, L=L, H=H, degree=degree)

    print(hs)
    print(errors)
    print(total_iterations)
