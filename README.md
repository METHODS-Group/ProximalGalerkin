# ProximalGalerkin

Examples of the proximal Galerkin finite element method.

| Figure |                                        File: examples/                                        |     Backend      | Instructions                     |
| :----: | :-------------------------------------------------------------------------------------------: | :--------------: | -------------------------------- |
|   2b   |                 [obstacle/compare_all.py](./examples/obstacle/compare_all.py)                 |      FEniCS      | [Obstacle problem](#obstacle)    |
| 2c(i)  |           [obstacle/finite_difference.jl](./examples/obstacle/finite_difference.jl)           |      Julia       | [Obstacle problem](#obstacle)    |
| 2c(ii) |                    [obstacle/spectral.jl](./examples/obstacle/spectral.jl)                    |      Julia       | [Obstacle problem](#obstacle)    |
|   3    |                     [signorini/script.py](./examples/signorini/script.py)                     |      FEniCS      | [Signorini problem](#signorini)  |
|   4    |                                               ?                                               | Firedrake/FEniCS | [Fracture](#fracture)            |
|   5    |                [cahn-hilliard/problem.py](./examples/cahn-hilliard/problem.py)                |      FEniCS      | [Cahn-Hilliard](#ch)             |
|   6    | [thermoforming_qvi/thermoforming_lvpp.jl](./examples/thermoforming_qvi/thermoforming_lvpp.jl) |      Julia       | [Thermoforming QVI](#qvi)        |
|   7    |           [gradient_constraint/script.py](./examples/gradient_constraint/script.py)           |      FEniCS      | [Gradient constraint](#gradient) |
|   8    |                                               ?                                               |    Firedrake     |                                  |
|   9    |                                               ?                                               |    Firedrake     |                                  |
|   10   |            [harmonic_maps/harmonic_1d.py](./examples/harmonic_maps/harmonic_1d.py)            |      FEniCS      | [Harmonic map](#harmonic)        |
|   11   |                        [eikonal/ex40.cpp](./examples/eikonal/ex40.cpp)                        |       MFEM       | [Eikonal](#eikonal)              |
|   12   |                [monge_ampere/cg_cg_dg.py](./examples/monge_ampere/cg_cg_dg.py)                | Firedrake/FEniCS | [Monge-Ampere](#monge)           |

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

<a name="fracture"></a>

# Fracture

Can be simulated with either FEniCS/DOLFINx or Firedrake.
The DOLFINx code can be executed with

```bash
python3 script.py
```

while the Firedrake code can be executed with:

> [!WARNING]  
> Add instructions

```bash

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

## Eikonal equation

The MFEM example can be executed by copying
[./examples/eikonal/ex40.cpp](./examples/eikonal/ex40.cpp) into the `mfem` examples folder
and call `make ex40`. It can then be executed with:

```bash
docker run -it --rm -v ./examples/eikonal:/home/euler/shared -w /home/euler/mfem --rm --entrypoint=/bin/bash ghcr.io/methods-group/proximalgalerkin-mfem:main
cp /home/euler/shared/ex40.cpp /home/euler/mfem/examples/
cd examples && make ex40
./ex40
```

For the non-manifold examples, with the [Star](https://github.com/mfem/mfem/blob/master/data/star.mesh)
and [Ball](https://github.com/mfem/mfem/blob/master/data/ball-nurbs.mesh) you can compile the [official demo](https://mfem.org/examples/), `ex40.cpp` or `ex40p.cpp` without copying any files from this repository.

The DOLFINx example, in [./examples/eikonal/script.py](./examples/eikonal/script.py) requires to convert the mobius strip mesh from `mfem`, called [mobius-strip.mesh](https://github.com/mfem/mfem/blob/master/data/mobius-strip.mesh)

From the root of the repository, you can call the following commands to compile the code

```bash
docker run -it --rm -v ./examples/eikonal:/home/euler/shared -w /home/euler/mfem --rm --entrypoint=/bin/bash ghcr.io/methods-group/proximalgalerkin-mfem:main
cp /home/euler/shared/convert_mesh.cpp /home/euler/mfem/examples/
cd examples && make convert_mesh
./convert_mesh --mesh ../data/mobius-strip.mesh
cp -r  mobius-strip.mesh/ ../../shared/
```

<a name="monge"></a>

# Monge-Ampere

The firedrake code can be run with

```bash
python3 cg_cg_dg.py
```

The equivalent FEniCS/DOLFINx code can be run with

```bash
python3 cg_cg_dg_fenics.py
```
