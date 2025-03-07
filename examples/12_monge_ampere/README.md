# Example 12 (Figure 12): The Monge–Ampere Equation

## Firedrake

Start a docker container based on `ghcr.io/methods-group/proximalgalerkin:v0.2.0`.
Then call `source firedrake-mode` and run the following command

```bash
python3 monge_ampere_firedrake.py
```

## DOLFINx

Start a docker container based on `ghcr.io/methods-group/proximalgalerkin:v0.2.0`.
Then call `source dolfinx-mode` and run the following command
A P-refinement study for monge ampere can be run with `DOLFINx`

```bash
python3 monge_ampere_dolfinx.py
```
