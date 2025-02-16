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

# Table of Examples

The following table associates each implementation to the figures in the paper. Further information to run the codes is provided below.

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

We provide the following Docker containers for the various methods used in the paper:

- DOLFINx: [ghcr.io/methods-group/proximalgalerkin-dolfinx:main](https://github.com/METHODS-Group/ProximalGalerkin/pkgs/container/proximalgalerkin-dolfinx)
- MFEM: [ghcr.io/methods-group/proximalgalerkin-mfem:main](https://github.com/METHODS-Group/ProximalGalerkin/pkgs/container/proximalgalerkin-mfem)
- Firedrake: [ghcr.io/methods-group/proximalgalerkin-firedrake:main](https://github.com/METHODS-Group/ProximalGalerkin/pkgs/container/proximalgalerkin-firedrake)
- Julia/GridAP: [julia:1.10.8](https://hub.docker.com/layers/library/julia/1.10.8/images/sha256-66656909ed7b5e75f4208631b01fc585372f906d68353d97cc06b40a8028c437)

<a name="obstacle"></a>

# Example 1: The obstacle problem

Figures 2 (a) and (b) are generated with DOLFINx.

To reproduce the results in Figures 2 (a) (the comparison between Proximal Galerkin, SNES, Galahad, and IPOPT), first deploy the `DOLFINx` Docker containers and then run the following commands within `examples/obstacle`:

```bash
python3 generate_mesh_gmsh.py
python3 compare_all.py -P ./meshes/disk_1.xdmf -O coarse
python3 compare_all.py -P ./meshes/disk_2.xdmf -O medium
python3 compare_all.py -P ./meshes/disk_3.xdmf -O fine
```

Reproduce the finite difference and spectral element method results in Figure 2 (c) by deploying the `julia:1.10.8` Docker container and calling

```bash
julia finite_difference.jl
julia spectral.jl
```

within `examples/obstacle`.

<a name="signorini"></a>

# Example 2: The Signorini problem

Deploy the `DOLFINx` Docker container to reproduce the results in Figure 3.
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

# Example 3: Variational Fracture

This example (cf. Figure 4) can be run from within both the `DOLFINx` and the `Firedrake` Docker containers.
The `DOLFINx` code can be executed within `examples/fracture`

```bash
python3 script.py
```

while the `Firedrake` code can be executed with:

> [!WARNING]  
> Add instructions

```bash

```

<a name="ch"></a>

# Example 4: Four-Phase Cahn–Hilliard Gradient Flow

Deploy the `DOLFINx` Docker container to reproduce the results in Figure 5.
Then run

```bash
python3 problem.py
```

from within `examples/cahn-hilliard`.

<a name="qvi"></a>

# Example 5: Thermoforming Quasi-Variational Inequality

This example requires the `julia:1.10.8` Docker container to reproduce the results in Figure 6.
The code can be executed by running

```bash
julia theroforming_lvpp.jl
```

from within `examples/thermoforming_qvi`.

<a name="gradient"></a>

# Example 6: Gradient Norm Constraints.

Deploy the `DOLFINx` Docker container to reproduce the results in Figure 7.
Then run `script.py` within `examples/gradient_constraint` with the following input parameters:

```bash
python3 script.py -N 80 -M 80 --alpha_scheme=doubling
```

<a name="harmonic"></a>

## Example 7: Eigenvalue Constraints

Deploy the `Firedrake` Docker container to reproduce the results in Figure 8.
Then run the following command within `examples/[TODO]`:

```bash
python3 script.py -N 80 -M 80 --alpha_scheme=doubling
```

> [!WARNING]  

## Example 8: Intersections of Constraints

Deploy the `Firedrake` Docker container to reproduce the results in Figure 9.
Then run the following command within `examples/[TODO]`:

> [!WARNING]  

## Example 9: Harmonic Maps to the Sphere

Deploy the `DOLFINx` Docker container to reproduce the results in Figure 10.
Then run the following command within `examples/harmonic_maps`:

```bash
python3 harmonic_1D.py
```

## Example 10: Linear Equality Constraints
Note that there is no numerical example for this setting because the derived variational formulation is equivalent to the standard Lagrange multiplier formulation for this class of problems.

## Example 11: Eikonal equation

We have provided code for this example for both the `MFEM` and `DOLFINx` Docker containers.

To reproduce the Möbius strip solution in Figure 11, first copy [./examples/eikonal/ex40.cpp](./examples/eikonal/ex40.cpp) into the `mfem` examples folder and then call `make ex40`. The code can then be executed with:

```bash
docker run -it --rm -v ./examples/eikonal:/home/euler/shared -w /home/euler/mfem --rm --entrypoint=/bin/bash ghcr.io/methods-group/proximalgalerkin-mfem:main
cp /home/euler/shared/ex40.cpp /home/euler/mfem/examples/
cd examples && make ex40
./ex40 -mi 10
```

For the other two geometries (i.e., the [Star](https://github.com/mfem/mfem/blob/master/data/star.mesh)
and [Ball](https://github.com/mfem/mfem/blob/master/data/ball-nurbs.mesh)) in Figure 11, you should compile the [official examples](https://mfem.org/examples/) `ex40.cpp` or `ex40p.cpp` without copying any files from this repository.

The `DOLFINx` implementation, found in [./examples/eikonal/script.py](./examples/eikonal/script.py) requires first converting the `MFEM` Möbius strip mesh [mobius-strip.mesh](https://github.com/mfem/mfem/blob/master/data/mobius-strip.mesh).
To this end, run the following commands from the root of this repository:

```bash
docker run -it --rm -v ./examples/eikonal:/home/euler/shared -w /home/euler/mfem --rm --entrypoint=/bin/bash ghcr.io/methods-group/proximalgalerkin-mfem:main
cp /home/euler/shared/convert_mesh.cpp /home/euler/mfem/examples/
cd examples && make convert_mesh
./convert_mesh --mesh ../data/mobius-strip.mesh
cp -r  mobius-strip.mesh/ ../../shared/
```

The `DOLFINx` code is then run by calling:

```bash
python3 script.py
```

from within `examples/eikonal`.

<a name="monge"></a>

# Monge-Ampere

This example (cf. Figure 4) can be run from within `examples/monge_ampere` using both the `DOLFINx` and the `Firedrake` Docker containers.

The `Firedrake` code can be run with the command

```bash
python3 cg_cg_dg.py
```

The equivalent `DOLFINx` code can be run with

```bash
python3 cg_cg_dg_fenics.py
```
