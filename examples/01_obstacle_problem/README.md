# Example 1 (Figure 3): The Obstacle Problem

## Finite element method (proximal Galerkin)

Figures 3 (a) and (b) are generated with `DOLFINx`.

To reproduce the results in Figures 3 (a) (the comparison between Proximal Galerkin, SNES, Galahad, and IPOPT),
first deploy the `ghcr.io/methods-group/proximalgalerkin:v0.3.0` Docker container.
Then execute the following scripts (in `dolfinx-mode`, see [../../README.md](../../README.md)):

```bash
python3 generate_mesh_gmsh.py
python3 compare_all.py -P ./meshes/disk_1.xdmf -O coarse
python3 compare_all.py -P ./meshes/disk_2.xdmf -O medium
python3 compare_all.py -P ./meshes/disk_3.xdmf -O fine
```

> [!NOTE]
> The comparison script requires both [GALAHAD](https://github.com/ralna/GALAHAD) and [IPOPT](https://coin-or.github.io/Ipopt/) which are not
> supplied by the standard FEniCS/DOLFINx installation.

## Proximal finite difference and spectral methods

To reproduce the finite difference and spectral method results in Figure 2 (c), deploy a container with [Julia](https://julialang.org/), for instance (`julia:1.10.8` or `ghcr.io/methods-group/proximalgalerkin`).

```bash
julia obstacle_finite_difference.jl
julia obstacle_spectral.jl
```

If using the docker containers, you can most likely use the flag

```bash
julia --compiled-modules=existing obstacle_finite_difference.jl
```

to speed up the run-time.
