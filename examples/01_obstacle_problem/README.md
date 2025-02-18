# Example 1 (Figure 2): The Obstacle Problem

## Finite element method

Figures 2 (a) and (b) are generated with `DOLFINx`.

To reproduce the results in Figures 2 (a) (the comparison between Proximal Galerkin, SNES, Galahad, and IPOPT), first deploy the `DOLFINx` Docker container. Then run the following commands within `examples/obstacle`:

```bash
python3 generate_mesh_gmsh.py
python3 compare_all.py -P ./meshes/disk_1.xdmf -O coarse
python3 compare_all.py -P ./meshes/disk_2.xdmf -O medium
python3 compare_all.py -P ./meshes/disk_3.xdmf -O fine
```

## Finite difference and spectral element method

To reproduce the finite difference and spectral element method results in Figure 2 (c), deploy the `julia:1.10.8` Docker container and call

```bash
julia finite_difference.jl
julia spectral.jl
```

within `examples/obstacle`.
