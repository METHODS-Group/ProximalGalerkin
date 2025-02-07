"""
Read mobius strip from pvd
"""

import meshio
import numpy as np
from mpi4py import MPI
import dolfinx
import ufl
import basix.ufl
from pathlib import Path

eps = np.finfo(dolfinx.default_scalar_type).eps


def remove_duplicates(points, tol=1e2 * eps):
    """
    Remove duplicate points from a list of points and create a map to the new, unique set
    """
    # Compute distances between boundary coordinates and T coordinates
    points_A = np.expand_dims(points, 1)
    points_B = np.expand_dims(points, 0)
    distances = np.sum(np.square(points_A - points_B), axis=2)
    is_close = distances < tol
    new_point_indices = []
    point_map = np.full(points.shape[0], -1, dtype=np.int32)
    for i in range(points.shape[0]):
        if sum(is_close[i, :i]) == 0:
            point_map[i] = len(new_point_indices)
            new_point_indices.append(i)
        else:
            point_map[i] = point_map[min(np.flatnonzero(is_close[i, :i]))]
    return points[new_point_indices], point_map


def read_mobius_strip(filename: Path):
    if MPI.COMM_WORLD.rank == 0:
        in_mesh = meshio.read(filename)

        cells = in_mesh.cells_dict["VTK_LAGRANGE_QUADRILATERAL"]
        points = in_mesh.points
        unique_points, point_map = remove_duplicates(points)
        # theta = np.pi / 9.2
        # unique_points[:, :1] += 0.1
        # rot_matrix = np.array(
        #     [[np.cos(theta), 0, -np.sin(theta)], [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]]
        # )
        # unique_points = (rot_matrix @ unique_points.T).T

        order = int(np.sqrt(cells.shape[1])) - 1
        perm_order = dolfinx.cpp.io.perm_vtk(dolfinx.mesh.CellType.quadrilateral, cells.shape[1])
        cells = point_map[cells]
        dx_cells = cells[:, perm_order]
        assert len(np.unique(dx_cells.flatten())) == unique_points.shape[0], (
            "More points than those in cells"
        )
        MPI.COMM_WORLD.bcast(order, root=0)
    else:
        order = MPI.COMM_WORLD.bcast(None, root=0)
    MPI.COMM_WORLD.barrier()

    if MPI.COMM_WORLD.rank != 0:
        dx_cells = np.empty((0, (order + 1) ** 2), dtype=np.int32)
        unique_points = np.empty((0, 3), dtype=np.float64)
    partitioner = dolfinx.cpp.mesh.create_cell_partitioner(dolfinx.mesh.GhostMode.shared_facet)
    ufl_domain = ufl.Mesh(basix.ufl.element("Lagrange", "quadrilateral", order, shape=(3,)))
    mesh = dolfinx.mesh.create_mesh(
        MPI.COMM_WORLD, dx_cells, unique_points, ufl_domain, partitioner
    )
    return mesh


if __name__ == "__main__":
    mesh = read_mobius_strip(Path("mobius-strip.mesh/Cycle000000/proc000000.vtu"))
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    exterior_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 4))
    u = dolfinx.fem.Function(V)
    dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, exterior_facets)
    u.x.array[dofs] = 1

    with dolfinx.io.VTXWriter(mesh.comm, "mobius.bp", [u]) as bp:
        bp.write(0.0)
