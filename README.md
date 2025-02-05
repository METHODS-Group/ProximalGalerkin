# ProximalGalerkin

Examples of the proximal Galerkin finite element method.

| Figure |                                                                     File: examples/                                                                     |  Backend  | Instructions                     |
| :----: | :-----------------------------------------------------------------------------------------------------------------------------------------------------: | :-------: | -------------------------------- |
|   2b   |                 [obstacle/compare_all.py](https://github.com/METHODS-Group/ProximalGalerkin/blob/main/examples/obstacle/compare_all.py)                 |  FEniCSx  | [Obstacle problem](#obstacle)    |
| 2c(i)  |           [obstacle/finite_difference.jl](https://github.com/METHODS-Group/ProximalGalerkin/blob/main/examples/obstacle/finite_difference.jl)           |   Julia   | [Obstacle problem](#obstacle)    |
| 2c(ii) |                    [obstacle/spectral.jl](https://github.com/METHODS-Group/ProximalGalerkin/blob/main/examples/obstacle/spectral.jl)                    |   Julia   | [Obstacle problem](#obstacle)    |
|   3    |                     [signorini/script.py](https://github.com/METHODS-Group/ProximalGalerkin/blob/main/examples/signorini/script.py)                     |  FEniCSx  | [Signorini problem](#signorini)  |
|   4    |                                                                            ?                                                                            | Firedrake |                                  |
|   5    |                [cahn-hilliard/problem.py](https://github.com/METHODS-Group/ProximalGalerkin/blob/main/examples/cahn-hilliard/problem.py)                |  FEniCSx  | [Cahn-Hilliard](#ch)             |
|   6    | [thermoforming_qvi/thermoforming_lvpp.jl](https://github.com/METHODS-Group/ProximalGalerkin/blob/main/examples/thermoforming_qvi/thermoforming_lvpp.jl) |   Julia   | [Thermoforming QVI](#qvi)        |
|   7    |           [gradient_constraint/script.py](https://github.com/METHODS-Group/ProximalGalerkin/blob/main/examples/gradient_constraint/script.py)           |  FEniCSx  | [Gradient constraint](#gradient) |
|   8    |                                                                            ?                                                                            | Firedrake |                                  |
|   9    |                                                                            ?                                                                            | Firedrake |                                  |
|   10   |            [harmonic_maps/harmonic_1d.py](https://github.com/METHODS-Group/ProximalGalerkin/blob/main/examples/harmonic_maps/harmonic_1d.py)            |  FEniCSx  | [Harmonic map](#harmonic)        |
|   11   |                                                                            ?                                                                            |   MFEM    |                                  |

## Installation

We provide several docker container for the various methods used in the paper

- DOLFINx: `ghcr.io/methods-group/proximalgalerkin-dolfinx:main`
- MFEM: `ghcr.io/methods-group/proximalgalerkin-mfem:main`
- Firedrake: `ghcr.io/methods-group/proximalgalerkin-firedrake:main`
- Julia/GridAP: `julia:1.10.8`

<a name="obstacle"></a>

# Obstacle problem

Parts of this table is generated using DOLFINx.

To get the results from Galahad, IPOPT, SNES and LVPP (FEM) use the LVPP docker image for DOLFINx and run (within `examples/obstacle`)

```bash
python3 generate_mesh_gmsh.py
python3 compare_all.py -P ./meshes/disk_1.xdmf -O coarse
python3 compare_all.py -P ./meshes/disk_2.xdmf -O medium
python3 compare_all.py -P ./meshes/disk_3.xdmf -O fine
```

Julia is used to get results for the finite difference and spectral element method, use the `julia:1.10.8` Docker container and call

```bash
julia finite_difference.jl
julia spectral.jl
```

<a name="signorini"></a>

# Signorini problem

Requires DOLFINx and [scifem](https://github.com/scientificcomputing/scifem). These are installed in the docker image.

From within `examples/signorini`, call

```bash
python3 generate_mesh.py
```

to generate the mesh file `"meshes/half_sphere.xdmf"`.
Next run the LVPP algorithm with

```bash
python3 run_lvpp_problem.py --alpha_0=0.005 --degree=2 --disp=-0.3 --n-max-iterations=250 --alpha_scheme=doubling  --output output_lvpp file --filename=meshes/half_sphere.xdmf
```

<a name="ch"></a>

# Cahn-Hilliard problem

Codes can run from within `examples/cahn-hilliard` with

```bash
python3 problem.py
```

<a name="qvi"></a>

# Thermoforming quasi-variational inequalities

Requires Julia. Can be executed with

```bash
julia theroforming_lvpp.jl
```

from the `examples/thermoforming_qvi` folder.

<a name="gradient"></a>

# Gradient constraint

Run the `script.py` with the following input parameters:

```bash
python3 script.py -N 80 -M 80 --alpha_scheme=doubling
```

<a name="harmonic"></a>

## Harmonic maps

Requires DOLFINx. Run

```bash
python3 harmonic_1D.py
```
