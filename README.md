# ProximalGalerkin

This repository contains implementations of the proximal Galerkin finite element method and other proximal numerical methods for variational problems with inequality constraints derived in

```bibtex
@misc{dokken2025latent,
      title={The latent variable proximal point algorithm for variational problems with constraints}, 
      author={Dokken, {J\o rgen} S. and Farrell, Patrick~E. and Keith, Brendan and Papadopoulos, Ioannis~P.A. and Surowiec, Thomas~M.},
      year={2025},
}
```

Please cite the aforementioned manuscript if using the code in this repository.

## Instructions

We encourage using following Docker containers to run the  codes described below:

- DOLFINx: [ghcr.io/methods-group/proximalgalerkin-dolfinx:main](https://github.com/METHODS-Group/ProximalGalerkin/pkgs/container/proximalgalerkin-dolfinx)
- MFEM: [ghcr.io/methods-group/proximalgalerkin-mfem:main](https://github.com/METHODS-Group/ProximalGalerkin/pkgs/container/proximalgalerkin-mfem)
- Firedrake: [ghcr.io/methods-group/proximalgalerkin-firedrake:main](https://github.com/METHODS-Group/ProximalGalerkin/pkgs/container/proximalgalerkin-firedrake)
- Julia/GridAP: [julia:1.10.8](https://hub.docker.com/layers/library/julia/1.10.8/images/sha256-66656909ed7b5e75f4208631b01fc585372f906d68353d97cc06b40a8028c437)

<a name="obstacle"></a>

## Table of Examples and Figures

The following table associates each implementation to the examples and figures in the paper. Further information to run the codes is provided for each specific example.

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

## Example 1 (Figure 2): The Obstacle Problem

Figures 2 (a) and (b) are generated with `DOLFINx`.

To reproduce the results in Figures 2 (a) (the comparison between Proximal Galerkin, SNES, Galahad, and IPOPT), first deploy the `DOLFINx` Docker container. Then run the following commands within `examples/obstacle`:

```bash
python3 generate_mesh_gmsh.py
python3 compare_all.py -P ./meshes/disk_1.xdmf -O coarse
python3 compare_all.py -P ./meshes/disk_2.xdmf -O medium
python3 compare_all.py -P ./meshes/disk_3.xdmf -O fine
```

To reproduce the finite difference and spectral element method results in Figure 2 (c), deploy the `julia:1.10.8` Docker container and call

```bash
julia finite_difference.jl
julia spectral.jl
```

within `examples/obstacle`.

<a name="signorini"></a>

## Example 2 (Figure 3): The Signorini Problem

Deploy the `DOLFINx` Docker container to reproduce the results in this example.
Then call

```bash
python3 generate_mesh.py
```

from within `examples/signorini` to generate the mesh file `"meshes/half_sphere.xdmf"`.
Next, run the proximal Galerkin method with

```bash
python3 run_lvpp_problem.py --alpha_0=0.005 --degree=2 --disp=-0.3 --n-max-iterations=250 --alpha_scheme=doubling  --output output_lvpp file --filename=meshes/half_sphere.xdmf
```

<a name="fracture"></a>

## Example 3 (Figure 4): Variational Fracture

This example can be run from within `examples/fracture` using both the `DOLFINx` and the `Firedrake` Docker containers.
The `DOLFINx` code can be executed with

```bash
python3 script.py
```

while the `Firedrake` code can be executed with:

> [!WARNING]  
> Add instructions

```bash

```

<a name="ch"></a>

## Example 4 (Figure 5): Four-Phase Cahn–Hilliard Gradient Flow

To reproduce the results in this example, first deploy the `DOLFINx` Docker container.
Then run

```bash
python3 problem.py
```

from within `examples/cahn-hilliard`.

<a name="qvi"></a>

## Example 5 (Figure 6): Thermoforming Quasi-Variational Inequality

Reproducing the results in this example requires the `julia:1.10.8` Docker container.
Once this container is deployed, the code can be executed by running

```bash
julia theroforming_lvpp.jl
```

from within `examples/thermoforming_qvi`.

<a name="gradient"></a>

## Example 6 (Figure 7): Gradient Norm Constraints

Deploy the `DOLFINx` Docker container to reproduce the results in this example.
Then run `script.py` within `examples/gradient_constraint` with the following input parameters:

```bash
python3 script.py -N 80 -M 80 --alpha_scheme=doubling
```

<a name="harmonic"></a>

## Example 7 (Figure 8): Eigenvalue Constraints

Deploy the `Firedrake` Docker container to reproduce the results in this example.
Then run the following command within `examples/[TODO]`:

> [!WARNING]  
> Add instructions 

## Example 8 (Figure 9): Intersections of Constraints

Deploy the `Firedrake` Docker container to reproduce the results in this example.
Then run the following command within `examples/[TODO]`:

> [!WARNING]  
> Add instructions 

## Example 9 (Figure 10): Harmonic Maps to the Sphere

Deploy the `DOLFINx` Docker container to reproduce the results in this example.
Then run the following command within `examples/harmonic_maps`:

```bash
python3 harmonic_1D.py
```

## Example 10: Linear Equality Constraints
Note that there is no numerical example for this setting because the derived variational formulation is equivalent to the standard Lagrange multiplier formulation for this class of problems.

## Example 11 (Figure 11): The Eikonal Equation

We have provided code for this example for both the `MFEM` and `DOLFINx` Docker containers.

To reproduce the Möbius strip solution in Figure 11, you first need to copy [./examples/eikonal/ex40.cpp](./examples/eikonal/ex40.cpp) into the `MFEM` examples folder (`/home/euler/mfem/examples/`) and then calling `make ex40` and `./ex40 -step 10.0 -mi 10`. This following code will execute to entire process:

```bash
docker run -it --rm -v ./examples/eikonal:/home/euler/shared -w /home/euler/mfem --rm --entrypoint=/bin/bash ghcr.io/methods-group/proximalgalerkin-mfem:main
cp /home/euler/shared/ex40.cpp /home/euler/mfem/examples/
cd examples && make ex40
./ex40 -step 10.0 -mi 10
```

To reproduce the results in Figure 11 for the two geometries (i.e., the [Star](https://github.com/mfem/mfem/blob/master/data/star.mesh)
and [Ball](https://github.com/mfem/mfem/blob/master/data/ball-nurbs.mesh)), you should compile the [official examples](https://mfem.org/examples/) `ex40.cpp` or `ex40p.cpp` without copying any files from this repository
```bash
cd examples && make ex40
# Star Geometry
./ex40 -step 10.0 -mi 10
# Ball Geometry
./ex40 -step 10.0 -mi 10 -m ../data/ball-nurbs.mesh
```

The `DOLFINx` implementation, found in [./examples/eikonal/script.py](./examples/eikonal/script.py) requires first converting the `MFEM` Möbius strip mesh [mobius-strip.mesh](https://github.com/mfem/mfem/blob/master/data/mobius-strip.mesh).
To this end, run the following commands from the root of this repository:

```bash
docker run -it --rm -v ./examples/eikonal:/home/euler/shared -w /home/euler/mfem --rm --entrypoint=/bin/bash ghcr.io/methods-group/proximalgalerkin-mfem:main
cp /home/euler/shared/convert_mesh.cpp /home/euler/mfem/examples/
cd examples && make convert_mesh
./convert_mesh --mesh ../data/mobius-strip.mesh
cp -r  mobius-strip.mesh/ ../../shared/
```

The `DOLFINx` code is then executed by calling:

```bash
python3 script.py
```

from within `examples/eikonal`.

<a name="monge"></a>

## Example 12 (Figure 12): The Monge–Ampere Equation

This example can be run from within `examples/monge_ampere` using both the `DOLFINx` and the `Firedrake` Docker containers.

The `Firedrake` code can be run with the command

```bash
python3 cg_cg_dg.py
```

The equivalent `DOLFINx` code can be run with

```bash
python3 cg_cg_dg_fenics.py
```
