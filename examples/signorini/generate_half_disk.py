from mpi4py import MPI
import dolfinx.io
import gmsh
from pathlib import Path

__all__ = ["generate_half_disk"]

def generate_half_disk(c_y:float, R:float, res:float, order:int=1,refinement_level: int=1)->dolfinx.mesh.Mesh:
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
        square = gmsh.model.occ.addRectangle(-R, c_y, 0, 2*R, 1.1*R)
        gmsh.model.occ.synchronize()
        new_tags, _ = gmsh.model.occ.cut([(2, membrane)], [(2, square)])
        gmsh.model.occ.synchronize()

        for i,tags in enumerate(new_tags):
            gmsh.model.addPhysicalGroup(tags[0], [tags[1]], i)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", res)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", res)
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