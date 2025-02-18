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


## Example 10: Linear Equality Constraints

Note that there is no numerical example for this setting because the derived variational formulation is equivalent to the standard Lagrange multiplier formulation for this class of problems.

