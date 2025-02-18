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
