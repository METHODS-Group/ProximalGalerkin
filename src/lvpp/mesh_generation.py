from mpi4py import MPI

import dolfinx.io
import gmsh
import numpy as np

__all__ = ["generate_half_disk"]


def generate_half_disk(
    c_y: float, R: float, res: float, order: int = 1, refinement_level: int = 1
) -> dolfinx.mesh.Mesh:
    """Generate a half-disk with center (0,1.5) with radius 1 and resolution `res`.

    Args:
        filename: Name of the file to save the mesh to.
        res: Resolution of the mesh.
        order: Order of the mesh elements.
        refinement_level: Number of gmsh refinements
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
    msh, _, _ = dolfinx.io.gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=2)
    gmsh.finalize()
    return msh
