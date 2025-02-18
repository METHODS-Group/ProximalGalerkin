"""Generate half sphere for contact problem"""

from pathlib import Path

from mpi4py import MPI

import dolfinx

import lvpp.mesh_generation

mesh_path = Path("meshes/half_sphere.xdmf")
mesh, cell_marker, facet_marker = lvpp.mesh_generation.create_half_sphere(res=0.04)
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, mesh_path, "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(cell_marker, mesh.geometry)
    xdmf.write_meshtags(facet_marker, mesh.geometry)
