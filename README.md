# ProximalGalerkin

Examples of the proximal Galerkin finite element method.

| Figure |                                                                     File: examples/                                                                     |  Backend  |
| :----: | :-----------------------------------------------------------------------------------------------------------------------------------------------------: | :-------: |
|   2b   |                 [obstacle/compare_all.py](https://github.com/METHODS-Group/ProximalGalerkin/blob/main/examples/obstacle/compare_all.py)                 |  FEniCSx  |
| 2c(i)  |           [obstacle/finite_difference.jl](https://github.com/METHODS-Group/ProximalGalerkin/blob/main/examples/obstacle/finite_difference.jl)           |   Julia   |
| 2c(ii) |                    [obstacle/spectral.jl](https://github.com/METHODS-Group/ProximalGalerkin/blob/main/examples/obstacle/spectral.jl)                    |   Julia   |
|   3    |                     [signorini/script.py](https://github.com/METHODS-Group/ProximalGalerkin/blob/main/examples/signorini/script.py)                     |  FEniCSx  |
|   4    |                                                                            ?                                                                            | Firedrake |
|   5    |                [cahn-hilliard/problem.py](https://github.com/METHODS-Group/ProximalGalerkin/blob/main/examples/cahn-hilliard/problem.py)                |  FEniCSx  |
|   6    | [thermoforming_qvi/thermoforming_lvpp.jl](https://github.com/METHODS-Group/ProximalGalerkin/blob/main/examples/thermoforming_qvi/thermoforming_lvpp.jl) |   Julia   |
|   7    |           [gradient_constraint/script.py](https://github.com/METHODS-Group/ProximalGalerkin/blob/main/examples/gradient_constraint/script.py)           |  FEniCSx  |
|   8    |                                                                            ?                                                                            | Firedrake |
|   9    |                                                                            ?                                                                            | Firedrake |
|   10   |            [harmonic_maps/harmonic_1d.py](https://github.com/METHODS-Group/ProximalGalerkin/blob/main/examples/harmonic_maps/harmonic_1d.py)            |  FEniCSx  |
|   11   |                                                                            ?                                                                            |   MFEM    |

## Obstacle problem (Figure 2)

To get the results from Galahad, IPOPT, SNES and LVPP (FEM) use the LVPP docker image for DOLFINx and run (within `examples/obstacle`)

```bash
python3 generate_mesh_gmsh.py
python3 compare_all.py -P ./meshes/disk_1.xdmf -O coarse
python3 compare_all.py -P ./meshes/disk_2.xdmf -O medium
python3 compare_all.py -P ./meshes/disk_3.xdmf -O fine
```

To get the results for the finite difference and spectral element method, use the `julia:1.10.8` Docker container and call

```bash
julia finite_difference.jl
julia spectral.jl
```

## Signorini problem (Figure 3)

From within `examples/signorini`, call

```bash
python3 generate_mesh.py
```

to generate the mesh file `"meshes/half_sphere.xdmf"`.
Next run the LVPP algorithm with

```bash
python3 run_lvpp_problem.py --alpha_0=0.005 --degree=2 --disp=-0.3 --n-max-iterations=250 --alpha_scheme=doubling  --output output_lvpp file --filename=meshes/half_sphere.xdmf
```

## Gradient constraint

Use DOLFINx docker image and run the `script.py` with the following input parameters:

```bash
python3 script.py -N 80 -M 80 --alpha_scheme=doubling
```

## Dependencies

Scripts may rely on FEniCSx, Firedrake, MFEM, or Julia package backends.
