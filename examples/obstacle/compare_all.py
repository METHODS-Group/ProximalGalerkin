from mpi4py import MPI
import dolfinx
from ipopt_galahad import setup_problem, galahad, ipopt, ObstacleProblem
from pathlib import Path
import argparse
import numpy as np
from lvpp import solve_problem
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="Solve the obstacle problem on a unit square using Galahad.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
    parser.add_argument(
        "--path",
        "-P",
        dest="infile",
        type=Path,
        required=True,
        help="Path to infile",
    )
    parser.add_argument(
        "--results",
        "-O",
        dest="result_dir",
        type=Path,
        default=Path("results"),
        help="Path to results ",
    )
    max_iter = 500
    tol = 1e-5
    args = parser.parse_args()

    args.result_dir.mkdir(parents=True, exist_ok=True)

    # Set up problem matrices. initial guess and bounds
    problem = setup_problem(args.infile)
    S_, M_, V, f_, bounds, bcs = setup_problem(args.infile)
    dof_indices = np.unique(np.hstack([bc._cpp_object.dof_indices()[0] for bc in bcs]))
    keep = np.full(len(f_.x.array), True, dtype=np.bool_)
    keep[dof_indices] = False
    keep_indices = np.flatnonzero(keep)

    # Restrict all matrices and vectors to interior dofs
    S_d = S_[keep_indices].tocsc()[:, keep_indices].tocsr()
    M_d = M_[keep_indices].tocsc()[:, keep_indices].tocsr()
    f_d = f_.x.array[keep_indices]

    problem = ObstacleProblem(S_d, M_d, f_d)
    lower_bound = bounds[0].x.array[keep_indices]
    upper_bound = bounds[1].x.array[keep_indices]

    mesh = bounds[0].function_space.mesh
    degree = mesh.geometry.cmap.degree
    V_out = dolfinx.fem.functionspace(mesh, ("Lagrange", degree))




    # Solve with Galahad
    x_g = dolfinx.fem.Function(V, name="galahad")
    x_g.x.array[:] = 0.0
    init_galahad = x_g.x.array[keep_indices].copy()
    x_galahad, num_galahad_iterations = galahad(
        problem,
        init_galahad,
        (lower_bound, upper_bound),
        max_iter=max_iter,
        use_hessian=True,
        tol=tol
    )
    x_g.x.array[keep_indices] = x_galahad
    dolfinx.fem.set_bc(x_g.x.array, bcs)
    x_g_out = dolfinx.fem.Function(V_out, name="ipopt")
    x_g_out.interpolate(x_g)
    with dolfinx.io.VTXWriter(V.mesh.comm,  args.result_dir/f"{args.infile.stem}_galahad.bp", [x_g_out]) as bp:
        bp.write(0.0)

    
    # Solve with llvp (first order)

    u_lvpp, max_it = solve_problem(args.infile, 1, maximum_number_of_outer_loop_iterations=max_iter, alpha_scheme="double_exponential", alpha_max=1e4, tol_exit=tol)

    mesh = u_lvpp.function_space.mesh
    degree = mesh.geometry.cmap.degree
    V_out = dolfinx.fem.functionspace(mesh, ("Lagrange", degree))
    u_out = dolfinx.fem.Function(V_out, name="llvp")
    u_out.interpolate(u_lvpp.sub(0))
    with dolfinx.io.VTXWriter(mesh.comm, args.result_dir/f"{args.infile.stem}_llvp_first_order.bp", [u_out]) as bp:
        bp.write(0.0)



    # Solve with llvp (second order)

    u_lvpp_2, max_it_2 = solve_problem(args.infile, 2, maximum_number_of_outer_loop_iterations=max_iter, alpha_scheme="double_exponential", alpha_max=1000, tol_exit=tol)
    u_out = u_lvpp_2.sub(0).collapse()
    with dolfinx.io.VTXWriter(u_out.function_space.mesh.comm, args.result_dir/f"{args.infile.stem}_llvp_second_order.bp", [u_out]) as bp:
        bp.write(0.0)


    with dolfinx.io.VTXWriter(mesh.comm, args.result_dir/f"{args.infile.stem}_obstacle.bp", [bounds[0]]) as bp:
        bp.write(0.0)


    # Solve with IPOPT (With hessian)
    for with_hessian  in [True, False]:
        x_i = dolfinx.fem.Function(V, name="ipopt")
        x_i.x.array[:] = 0.0
        init_ipopt = x_i.x.array[keep_indices].copy()
        x_ipopt = ipopt(
            problem, init_ipopt, (lower_bound, upper_bound), max_iter=max_iter,
            tol=tol, activate_hessian=with_hessian
        )

        x_i.x.array[keep_indices] = x_ipopt
        dolfinx.fem.set_bc(x_i.x.array, bcs)

        # Output on geometry space

        x_i_out = dolfinx.fem.Function(V_out, name="ipopt")
        x_i_out.interpolate(x_i)
        with dolfinx.io.VTXWriter(mesh.comm, args.result_dir/f"{args.infile.stem}_ipopt_hessian_{with_hessian}.bp", [x_i_out]) as bp:
            bp.write(0.0)

    print(np.min(mesh.h(mesh.topology.dim, np.arange(mesh.topology.index_map(mesh.topology.dim).size_local))))
    print(args.infile, "Galahad iterations: ", num_galahad_iterations)
    print(args.infile, "llvp iterations: (P: 1)", max_it)
    print(args.infile, "llvp iterations: (P: 2)", max_it_2)
