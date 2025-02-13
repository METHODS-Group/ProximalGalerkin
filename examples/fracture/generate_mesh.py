try:
    import netgen.geom2d
except ModuleNotFoundError:
    print("This example requires the netgen-mesher module to be installed.")
    exit(1)
from mpi4py import MPI
import numpy as np
from packaging.version import Version
import dolfinx
import ufl
import basix.ufl


# Porting: https://github.com/pefarrell/proximalgalerkin/blob/master/fracture/lvpp.py
def create_crack_mesh(comm, max_res: float = 0.05):
    geo = netgen.geom2d.CSG2d()
    poly = netgen.geom2d.Solid2d(
        [
            (0, 0),
            netgen.geom2d.EdgeInfo(bc="bottom"),
            (2, 0),
            netgen.geom2d.EdgeInfo(bc="right"),
            (2, 2),
            netgen.geom2d.EdgeInfo(bc="topright"),
            (1.01, 2),
            netgen.geom2d.EdgeInfo(bc="crackright"),
            (1, 1.5),
            netgen.geom2d.EdgeInfo(bc="crackleft"),
            (0.99, 2),
            netgen.geom2d.EdgeInfo(bc="topleft"),
            (0, 2),
            netgen.geom2d.EdgeInfo(bc="left"),
        ]
    )

    disk = netgen.geom2d.Circle((0.3, 0.3), 0.2)
    geo.Add(poly - disk)
    if comm.rank == 0:
        ngmesh = geo.GenerateMesh(maxh=max_res)
        x = ngmesh.Coordinates()
        ng_elements = ngmesh.Elements2D()
        cell_indices = ng_elements.NumPy()["nodes"]
        if Version(np.__version__) >= Version("2.2"):
            cells = np.trim_zeros(cell_indices, "b", axis=1).astype(np.int64) - 1
        else:
            cells = (
                np.array(
                    [list(np.trim_zeros(a, "b")) for a in list(cell_indices)],
                    dtype=np.int64,
                )
                - 1
            )
    else:
        x = np.zeros((0, 2))
        cells = np.zeros((0, 3), dtype=np.int64)

    MPI.COMM_WORLD.barrier()

    ud = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
    linear_mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, x, ud)

    if comm.rank == 0:
        regions = [
            (name, i + 1) for i, name in enumerate(ngmesh.GetRegionNames(codim=1))
        ]
        ng_facets = ngmesh.Elements1D()
        facet_indices = ng_facets.NumPy()["nodes"].astype(np.int64)
        if Version(np.__version__) >= Version("2.2"):
            facets = np.trim_zeros(facet_indices, "b", axis=1).astype(np.int64) - 1
        else:
            facets = (
                np.array(
                    [list(np.trim_zeros(a, "b")) for a in list(facet_indices)],
                    dtype=np.int64,
                )
                - 1
            )

        facet_values = ng_facets.NumPy()["index"].astype(np.int32)
        regions = comm.bcast(regions, root=0)
    else:
        facets = np.zeros((0, 3), dtype=np.int64)
        facet_values = np.zeros((0,), dtype=np.int32)
        regions = comm.bcast(None, root=0)
    local_entities, local_values = dolfinx.io.gmshio.distribute_entity_data(
        linear_mesh, linear_mesh.topology.dim - 1, facets, facet_values
    )
    linear_mesh.topology.create_connectivity(linear_mesh.topology.dim - 1, 0)
    adj = dolfinx.graph.adjacencylist(local_entities)
    ft = dolfinx.mesh.meshtags_from_entities(
        linear_mesh,
        linear_mesh.topology.dim - 1,
        adj,
        local_values.astype(np.int32, copy=False),
    )
    ft.name = "Facet tags"
    return linear_mesh, ft, regions


if __name__ == "__main__":
    linear_mesh, ft, region_map = create_crack_mesh(MPI.COMM_WORLD)
    with dolfinx.io.XDMFFile(linear_mesh.comm, "mesh.xdmf", "w") as xdmf:
        xdmf.write_mesh(linear_mesh)
        linear_mesh.topology.create_connectivity(
            linear_mesh.topology.dim - 1, linear_mesh.topology.dim
        )
        xdmf.write_meshtags(ft, linear_mesh.geometry)

    # with dolfinx.io.VTXWriter(linear_mesh.comm, "mesh.bp", linear_mesh) as bp:
    #     bp.write(0.0)
