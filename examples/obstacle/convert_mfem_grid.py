import meshio
import basix.ufl
from mpi4py import MPI
import dolfinx
import argparse
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description="Convert a MFEM mesh to a FEniCS mesh.")
parser.add_argument("-i", dest="intput", type=Path, help="Path to MFEM mesh (to vtu file)", required=True)
parser.add_argument("-o", dest="output", type=Path, help="Path to FEniCS mesh", required=True)






def convert_mesh(in_mesh: Path, out_mesh: Path):
    in_mesh = meshio.read(in_mesh.with_suffix(".vtu"))
    points = in_mesh.points
    topology = in_mesh.cells_dict["VTK_LAGRANGE_QUADRILATERAL"]
    order = int(np.sqrt(topology.shape[1]) - 1)
    cell = basix.CellType.quadrilateral
    ufl_quad = basix.ufl.element("Q", cell, order, shape=(2,))
    map_vtk = dolfinx.cpp.io.perm_vtk(dolfinx.mesh.CellType.quadrilateral, topology.shape[1])

    quad_mesh = dolfinx.mesh.create_mesh(
        MPI.COMM_WORLD, topology[:, map_vtk], points[:, :2], ufl_quad
    )
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, out_mesh.with_suffix(".xdmf"), "w") as xdmf:
        xdmf.write_mesh(quad_mesh)


if __name__ == "__main__":


    args = parser.parse_args()


    convert_mesh(args.intput, args.output)