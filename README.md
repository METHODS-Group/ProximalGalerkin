# ProximalGalerkin

[![Launch on Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/METHODS-Group/ProximalGalerkin/HEAD)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14918044.svg)](https://doi.org/10.5281/zenodo.14918044)

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
docker run -ti -v $(pwd):/root/shared --name=proximal-examples ghcr.io/methods-group/proximalgalerkin:v0.3.0
```

This shares the current directory with the docker container under the location `/root/shared`.
To restart this container at a later instance call

```bash
docker container start -i proximal-examples
```

Within this installation you find all examples under `/root/LVPP`.

### MFEM

When wanting to run [MFEM](https://mfem.org/)-examples, one has to navigate to
`/root/LVPP/mfem/examples` and call `make name_of_example` to compile the corresponding demo.

The MFEM scripts from this paper are already placed in this location in the container.

### FEniCS/Firedrake compatibility

FEniCS and Firedrake are installed in separate virtual environments within the container.
At launch, the user gets to use `FEniCS` by default.
To change to `Firedrake`, call

```bash
source firedrake-mode
```

To change back to `FEniCS` call

```bash
source dolfinx-mode
```

## Table of Examples and Figures

The following table associates each implementation to the examples and figures in the paper. Further information to run the codes is provided for each specific example can be found in the corresponding folder.

| Figure |                                 Folder                                 |              Backend              | Problem Type                 |
| :----: | :--------------------------------------------------------------------: | :-------------------------------: | ---------------------------- |
|   3b   |         [01_obstacle_problem](./examples/01_obstacle_problem/)         |              FEniCS               | Obstacle problem (FEM)       |
| 3c(i)  |         [01_obstacle_problem](./examples/01_obstacle_problem/)         |               Julia               | Obstacle problem (FD)        |
| 3c(ii) |         [01_obstacle_problem](./examples/01_obstacle_problem/)         | MultivariateOrthogonalPolynomials | Obstacle problem (Spectral)  |
|   4    |                [02_signorini](./examples/02_signorini)                 |              FEniCS               | Signorini                    |
|   5    |                 [03_fracture](./examples/03_fracture/)                 |         Firedrake/FEniCS          | Fracture                     |
|   6    |               [04_multiphase](./examples/04_multiphase)                |              FEniCS               | Cahn-Hilliard                |
|   7    |        [05_obstacle_type_qvi](./examples/05_obstacle_type_qvi/)        |         Gridap.jl/FEniCS          | Thermoforming QVI            |
|   8    |     [06_gradient_constraints](./examples/06_gradient_constraints)      |              FEniCS               | Gradient constraint          |
|   9    |   [07_eigenvalue_constraints](./examples/07_eigenvalue_constraints)    |         Firedrake/FEniCS          | Landau–de Gennes             |
|   10   | [08_intersecting_constraints](./examples/08_intersecting_constraints/) |         Firedrake/FEniCS          | Intersections of constraints |
|   11   |                  [09_eikonal](./examples/09_eikonal)                   |            MFEM/FEniCS            | Eikonal equation             |
|   12   |             [10_monge_ampere](./examples/10_monge_ampere)              |         Firedrake/FEniCS          | Monge-Ampere                 |