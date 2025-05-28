# Example 5 (Figure 7): Thermoforming Quasi-Variational Inequality

## Julia implementation

To reproduce the LVPP results presented in the paper, run

```bash
julia thermoforming_gridap.jl
```

To reproduce the results of a Moreau-Yosida penalty solver, a semismooth active set strategy, and a fixed point approach, respectively, run

```bash
julia solver_comparison/thermoforming_moreau_yosida.jl
julia solver_comparison/thermoforming_semismooth_active_set.jl
julia solver_comparison/thermoforming_fixed_point.jl
```


## FEniCS implementation
A FEniCS implementation of LVPP with a different linesearch and linear solver can be run in `dolfinx-mode` with

```bash
python3 thermoforming_dolfinx.py 
```