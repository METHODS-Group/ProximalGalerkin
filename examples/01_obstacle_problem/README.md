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

> [!NOTE]
> The comparison script requires both [GALAHAD](https://github.com/ralna/GALAHAD) and [IPOPT](https://coin-or.github.io/Ipopt/) which is not
> supplied by standard FEniCS/DOLFINx installation. They are supplied in `ghcr.io/methods-group/proximalgalerkin` images.

## Finite difference and spectral element method

To reproduce the finite difference and spectral element method results in Figure 2 (c), deploy a container with [Julia](https://julialang.org/), for instance (`julia:1.10.8` or `ghcr.io/methods-group/proximalgalerkin`).

```bash
julia finite_difference.jl
julia spectral.jl
```

within `examples/obstacle`.
