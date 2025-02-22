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

## Installation instructions

We provide a single docker container `ghcr.io/methods-group/proximalgalerkin` from [Proximal Galerkin Docker container registry](https://github.com/METHODS-Group/ProximalGalerkin/pkgs/container/proximalgalerkin) that provides an installation of all dependencies used in the examples of this paper.

One can start the image with 
```bash
docker run -ti -v $(pwd):/root/shared --name=proximal-examples ghcr.io/methods-group/proximalgalerkin@sha256:360a5f7bc991157d146cccab764e5d24a0953ba8c838291c92e20fa3391c5eb6@sha256:dd41c6afdf8bf3ae54d719fa281c5db3412d69df27f13c374a20b6d35b8aa14b
```
Within this installation you find all examples under `/root/LVPP`.

> [!NOTE]
> When wanting to run [MFEM](https://mfem.org/)-examples, one has to navigate to
> `/root/LVPP/mfem/examples` and call `make name_of_example` to compile the corresponding demo.

> [!NOTE]
> When wanting to run [Firedrake](https://www.firedrakeproject.org/)-examples, one has to navigate to
> activate the Firedrake Python environment with `source /root/LVPP/firedrake/bin/activate`.
> To deactivate this environment, and return to a [FEniCS](https://fenicsproject.org/) environment call `deactivate` in the terminal.


## Table of Examples and Figures

The following table associates each implementation to the examples and figures in the paper. Further information to run the codes is provided for each specific example can be found in the corresponding folder.

| Figure |                                 Folder                                 |              Backend              | Problem Type            |
| :----: | :--------------------------------------------------------------------: | :-------------------------------: | ---------------------------- |
|   2b   |         [01_obstacle_problem](./examples/01_obstacle_problem/)         |              FEniCS               | Obstacle problem (FEM)       |
| 2c(i)  |         [01_obstacle_problem](./examples/01_obstacle_problem/)         |               Julia               | Obstacle problem (FD)        |
| 2c(ii) |         [01_obstacle_problem](./examples/01_obstacle_problem/)         | MultivariateOrthogonalPolynomials | Obstacle problem (Spectral)  |
|   3    |                [02_signorini](./examples/02_signorini)                 |              FEniCS               | Signorini                    |
|   4    |                 [03_fracture](./examples/03_fracture/)                 |         Firedrake/FEniCS          | Fracture                     |
|   5    |               [04_multiphase](./examples/04_multiphase)                |              FEniCS               | Cahn-Hilliard                |
|   6    |        [05_obstacle_type_qvi](./examples/05_obstacle_type_qvi/)        |         Gridap.jl/FEniCS          | Thermoforming QVI            |
|   7    |     [06_gradient_constraints](./examples/06_gradient_constraints)      |              FEniCS               | Gradient constraint          |
|   8    |   [07_eigenvalue_constraints](./examples/07_eigenvalue_constraints)    |         Firedrake/FEniCS          | Landau–de Gennes             |
|   9    | [08_intersecting_constraints](./examples/08_intersecting_constraints/) |         Firedrake/FEniCS          | Intersections of constraints |
|   10   |     [09_equality_constraints](./examples/09_equality_constraints)      |              FEniCS               | Harmonic map                 |
|   11   |                  [11_eikonal](./examples/11_eikonal)                   |            MFEM/FEniCS            | Eikonal equation             |
|   12   |             [12_monge_ampere](./examples/12_monge_ampere)              |         Firedrake/FEniCS          | Monge-Ampere                 |

## Example 10: Linear Equality Constraints

Note that there is no numerical example for this setting because the derived variational formulation is equivalent to the standard Lagrange multiplier formulation for this class of problems.
