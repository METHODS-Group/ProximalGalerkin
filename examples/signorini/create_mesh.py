from pathlib import Path

from mpi4py import MPI

import dolfinx.io.gmshio
import gmsh
import numpy as np


def create_half_sphere(
    filename: str | Path = "sphere.xdmf",
    model_name: str | None = None,
    order: int = 2,
    center: tuple[float, float, float] = (0.0, 0.0, 0.5),
    res: float = 0.02,
    r: float = 0.4,
    comm: MPI.Comm = MPI.COMM_WORLD,
    rank: int = 0,
    sphere_surface: int = 2,
    flat_surface: int = 1,
):
    """Create a half-sphere 3D sphere at `center` with radius `r` in negative z-direction.

    Args:
        model_name: Name of the GMSH model. Defaults to None.
        order: Order of the mesh elements.
        center: Center of the sphere..
        res: Resolution of the mesh around bottom of sphere.
        r: Radius of the sphere.
        comm: MPI communicator
        rank (int, optional): Rank of the process that creates the mesh.
        sphere_surface (int, optional): Tag of the sphere surface.
        flat_surface (int, optional): Tag of the flat surface.
    """
    gmsh.initialize()
    model_names = gmsh.model.list()
    if model_name is None:
        model_name = "half_sphere"
        assert model_name not in model_names, "GMSH model already exists"
        gmsh.model.add(model_name)
        gmsh.model.setCurrent(model_name)

    angle = 0
    lc_min = res
    lc_max = 2 * res
    if comm.rank == rank:
        p0 = gmsh.model.occ.addPoint(center[0], center[1], center[2] - r)
        gmsh.model.occ.addSphere(
            center[0], center[1], center[2], r, angle1=-np.pi / 2, angle2=-angle
        )

        # Synchronize and create physical tags
        gmsh.model.occ.synchronize()
        volumes = gmsh.model.getEntities(3)

        sphere_boundary = gmsh.model.getBoundary(volumes, oriented=False, combined=False)
        for dim, entity in sphere_boundary:
            tag = (
                flat_surface
                if np.isclose(gmsh.model.occ.getCenterOfMass(dim, entity)[2], center[2])
                else sphere_surface
            )
            gmsh.model.addPhysicalGroup(dim, [entity], tag=tag)

        p_v = [v_tag[1] for v_tag in volumes]
        gmsh.model.addPhysicalGroup(3, p_v, tag=1)

        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "NodesList", [p0])

        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        gmsh.model.mesh.field.setNumber(2, "LcMin", lc_min)
        gmsh.model.mesh.field.setNumber(2, "LcMax", lc_max)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.5 * r)
        gmsh.model.mesh.field.setNumber(2, "DistMax", r)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)

        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(order)
        gmsh.model.mesh.remove_duplicate_nodes()
    mesh, ct, ft = dolfinx.io.gmshio.model_to_mesh(gmsh.model, comm, 0)
    gmsh.finalize()

    filename = Path(filename).with_suffix(".xdmf")
    with dolfinx.io.XDMFFile(comm, filename, "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(ct, mesh.geometry)
        xdmf.write_meshtags(ft, mesh.geometry)


if __name__ == "__main__":
    create_half_sphere(res=0.025)
