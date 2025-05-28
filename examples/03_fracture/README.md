# Example 3 (Figure 5): Variational Fracture

This example can be executed using either FEniCS (DOLFINx) or Firedrake.

> [!NOTE]
> This demo requires [NetGen](https://github.com/NGSolve/netgen) and
> specifically [netgen-mesher](https://pypi.org/project/netgen-mesher/).
> Unfortunately, netgen-mesher does not supply `linux/arm64` wheels, and thus
> are not available within `ghcr.io/methods-group/proximalgalerkin` on `arm` machines.

## DOLFINx

The `DOLFINx` code can be executed with

```bash
python3 fracture_dolfinx.py
```

## Firedrake
while the `Firedrake` code can be executed after calling `source firedrake-mode` and then

```bash
python3 fracture_firedrake.py
```


