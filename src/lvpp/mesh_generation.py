from mpi4py import MPI

import dolfinx.io
import gmsh
import numpy as np
from packaging.version import Version

__all__ = ["create_half_disk", "create_half_sphere"]


def create_half_disk(
    c_y: float,
    R: float,
    res: float,
    order: int = 1,
    refinement_level: int = 1,
    disk_marker: int = 2,
    top_marker: int = 1,
) -> tuple[dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags, dolfinx.mesh.MeshTags]:
    """Generate a half-disk.

    Args:
        c_y: Center of `half-disk in y direction
        R: Radius of the half-disk.
        res: Minimum resolution
        order: Order of the mesh elements.
        refinement_level: Number of gmsh refinements
        disk_marker: Integer used to mark curved surface
        top_marker: Integer used to mark flat surface
    """
    gmsh.initialize()
    if MPI.COMM_WORLD.rank == 0:
        membrane = gmsh.model.occ.addDisk(0, c_y, 0, R, R)
        square = gmsh.model.occ.addRectangle(-R, c_y, 0, 2 * R, 1.1 * R)
        gmsh.model.occ.synchronize()
        new_tags, _ = gmsh.model.occ.cut([(2, membrane)], [(2, square)])
        gmsh.model.occ.synchronize()

        boundary = gmsh.model.getBoundary(new_tags, oriented=False)
        contact_boundary = []
        dirichlet_boundary = []
        for bnd in boundary:
            mass = gmsh.model.occ.getMass(bnd[0], bnd[1])
            if np.isclose(mass, np.pi * R):
                contact_boundary.append(bnd[1])
            elif np.isclose(mass, 2 * R):
                dirichlet_boundary.append(bnd[1])
            else:
                raise RuntimeError("Unknown boundary")

        for i, tags in enumerate(new_tags):
            gmsh.model.addPhysicalGroup(tags[0], [tags[1]], i)

        gmsh.model.add_physical_group(1, contact_boundary, disk_marker)
        gmsh.model.add_physical_group(1, dirichlet_boundary, top_marker)
        distance_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", contact_boundary)
        threshold = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold, "IField", distance_field)
        gmsh.model.mesh.field.setNumber(threshold, "LcMin", res)
        gmsh.model.mesh.field.setNumber(threshold, "LcMax", 20 * res)
        gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.075 * R)
        gmsh.model.mesh.field.setNumber(threshold, "DistMax", 0.5 * R)

        gmsh.model.mesh.field.setAsBackgroundMesh(threshold)

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(order)
        for _ in range(refinement_level):
            gmsh.model.mesh.refine()
            gmsh.model.mesh.setOrder(order)
    gmsh_model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    model = dolfinx.io.gmsh.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=2)

    if Version(dolfinx.__version__) > Version("0.9.0"):
        msh = model.mesh
        ct = model.cell_tags
        ft = model.facet_tags
    else:
        msh, ct, ft = model
    gmsh.finalize()
    return msh, ct, ft


def create_half_sphere(
    model_name: str | None = None,
    order: int = 2,
    center: tuple[float, float, float] = (0.0, 0.0, 0.5),
    res: float = 0.02,
    r: float = 0.4,
    comm: MPI.Comm = MPI.COMM_WORLD,
    rank: int = 0,
    sphere_surface: int = 2,
    flat_surface: int = 1,
) -> tuple[dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags, dolfinx.mesh.MeshTags]:
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

    Returns:
        Tuple containing the mesh, cell tags and facet tags
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
    model = dolfinx.io.gmsh.model_to_mesh(gmsh.model, comm, 0)
    if Version(dolfinx.__version__) > Version("0.9.0"):
        msh = model.mesh
        ct = model.cell_tags
        ft = model.facet_tags
    else:
        msh, ct, ft = model
    gmsh.finalize()
    return msh, ct, ft
