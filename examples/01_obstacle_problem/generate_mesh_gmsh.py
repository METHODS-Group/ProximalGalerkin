from pathlib import Path

from mpi4py import MPI

import dolfinx.io
import gmsh
import packaging.version

__all__ = ["generate_disk"]


def generate_disk(filename: Path, res: float, order: int = 1, refinement_level: int = 1):
    """Generate a disk around the origin with radius 1 and resolution `res`.

    Args:
        filename: Name of the file to save the mesh to.
        res: Resolution of the mesh.
        order: Order of the mesh elements.
        refinement_level: Number of gmsh refinements
    """
    gmsh.initialize()
    if MPI.COMM_WORLD.rank == 0:
        membrane = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
        gmsh.model.occ.synchronize()
        gdim = 2
        gmsh.model.addPhysicalGroup(gdim, [membrane], 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", res)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", res)
        gmsh.model.mesh.generate(gdim)
        gmsh.model.mesh.setOrder(order)
        for _ in range(refinement_level):
            gmsh.model.mesh.refine()
            gmsh.model.mesh.setOrder(order)

    gmsh_model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    model = dolfinx.io.gmsh.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)
    msh = model[0]
    gmsh.finalize()
    out_name = filename.with_stem(f"{filename.stem}_{refinement_level}").with_suffix(".xdmf")
    filename.parent.mkdir(exist_ok=True, parents=True)
    with dolfinx.io.XDMFFile(mesh_comm, out_name, "w") as xdmf:
        xdmf.write_mesh(msh)


if __name__ == "__main__":
    for i in range(4):
        generate_disk(Path("meshes/disk.xdmf"), res=0.1, order=2, refinement_level=i)
