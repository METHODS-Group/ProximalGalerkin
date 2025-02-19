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

## Table of Examples and Figures

The following table associates each implementation to the examples and figures in the paper. Further information to run the codes is provided for each specific example can be found in the corresponding folder.

| Figure |                                 Folder                                 |              Backend              | Problem Type type            |
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
