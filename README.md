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

We encourage using following Docker containers to run the codes described below:

- DOLFINx: [ghcr.io/methods-group/proximalgalerkin-dolfinx:main](https://github.com/METHODS-Group/ProximalGalerkin/pkgs/container/proximalgalerkin-dolfinx)
- MFEM: [ghcr.io/methods-group/proximalgalerkin-mfem:main](https://github.com/METHODS-Group/ProximalGalerkin/pkgs/container/proximalgalerkin-mfem)
- Firedrake: [ghcr.io/methods-group/proximalgalerkin-firedrake:main](https://github.com/METHODS-Group/ProximalGalerkin/pkgs/container/proximalgalerkin-firedrake)
- Julia/GridAP: [julia:1.10.8](https://hub.docker.com/layers/library/julia/1.10.8/images/sha256-66656909ed7b5e75f4208631b01fc585372f906d68353d97cc06b40a8028c437)

<a name="obstacle"></a>

## Table of Examples and Figures

The following table associates each implementation to the examples and figures in the paper. Further information to run the codes is provided for each specific example.

| Figure |                                Folder                                |              Backend              | Problem Type type            |
| :----: | :------------------------------------------------------------------: | :-------------------------------: | ---------------------------- |
|   2b   |       [1_obstacle_problem_fem](./examples/1_obstacle_problem/)       |              FEniCS               | Obstacle problem             |
| 2c(i)  |       [1_obstacle_problem_fd](./examples/1_obstacle_problem/)        |               Julia               | Obstacle problem             |
| 2c(ii) |    [1_obstacle_problem_spectral](./examples/1_obstacle_problem/)     | MultivariateOrthogonalPolynomials | Obstacle problem             |
|   3    |                [2_signorini](./examples/2_signorini)                 |              FEniCS               | Signorini                    |
|   4    |                 [3_fracture](./examples/3_fracture/)                 |         Firedrake/FEniCS          | Fracture                     |
|   5    |               [4_multiphase](./examples/4_multiphase)                |              FEniCS               | Cahn-Hilliard                |
|   6    |        [5_thermoforming_qvi](./examples/5_obstacle_type_qvi/)        |         Gridap.jl/FEniCS          | Thermoforming QVI            |
|   7    |     [6_gradient_constraints](./examples/6_gradient_constraints)      |              FEniCS               | Gradient constraint          |
|   8    |   [7_eigenvalue_constraints](./examples/7_eigenvalue_constraints)    |         Firedrake/FEniCS          | Landau–de Gennes             |
|   9    | [8_intersecting_constraints](./examples/8_intersecting_constraints/) |         Firedrake/FEniCS          | Intersections of constraints |
|   10   |     [9_equality_constraints](./examples/9_equality_constraints)      |              FEniCS               | Harmonic map                 |
|   11   |       [11_nonlinear_eikonal](./examples/11_nonlinear_eikonal)        |            MFEM/FEniCS            | Eikonal equation             |
|   12   |  [12_nonlinear_monge_ampere](./examples/12_nonlinear_monge_ampere)   |         Firedrake/FEniCS          | Monge-Ampere                 |

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

<a name="eigenvalue"></a>

## Example 7 (Figure 8): Eigenvalue Constraints

Deploy the `Firedrake` Docker container to reproduce the results in this example.
Then run the following command within `examples/[TODO]`:

> [!WARNING]  
> Add instructions

Deploy the `DOLFINx` Docker container to reproduce the results in this example.
Then run the following command within [examples/7_Landau_de_Gennes](./examples/7_Landau_de_Gennes/):

```bash
python3 dolfinx_implementation.py
```

<a name="intersections"></a>

## Example 8 (Figure 9): Intersections of Constraints

Deploy the `Firedrake` Docker container to reproduce the results in this example.
Then run the following command within `examples/[TODO]`:

> [!WARNING]  
> Add instructions

Deploy the `DOLFINx` Docker container to reproduce the results in this example.
Then run the following command within [examples/8_intersecting_constraints](./examples/8_intersecting_constraints/):

```bash
python3 dolfinx_implementation.py
```

<a name="harmonic"></a>

## Example 9 (Figure 10): Harmonic Maps to the Sphere

Deploy the `DOLFINx` Docker container to reproduce the results in this example.
Then run the following command within `examples/harmonic_maps`:

```bash
python3 harmonic_1D.py
```

## Example 10: Linear Equality Constraints

Note that there is no numerical example for this setting because the derived variational formulation is equivalent to the standard Lagrange multiplier formulation for this class of problems.

<a name=eikonal></a>

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
