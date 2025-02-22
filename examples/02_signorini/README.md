## Example 2 (Figure 3): The Signorini Problem

This demo requires:
- DOLFINx
- GMSH
- The LVPP package from this repository.

These are all installed in `ghcr.io/methods-group/proximalgalerkin`.
One can also use `conda`, or other DOLFINx docker images, and install the `LVPP`-package with
```bash
python3 -m pip install git+https://https://github.com/METHODS-Group/ProximalGalerkin
```

The mesh can be generated with
```bash
python3 generate_mesh.py
```
and is placed in `"meshes/half_sphere.xdmf"`.
Next, run the proximal Galerkin method with

```bash
python3 run_lvpp_problem.py --alpha_0=0.005 --degree=2 --disp=-0.3 --n-max-iterations=250 --alpha_scheme=doubling  --output output_lvpp file --filename=meshes/half_sphere.xdmf
```
