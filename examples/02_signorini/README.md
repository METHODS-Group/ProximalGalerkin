# Example 2 (Figure 4): The Signorini Problem

## Dependencies
This demo requires:
- DOLFINx
- GMSH
- The LVPP package from this repository.

These are all installed in the `ghcr.io/methods-group/proximalgalerkin`.

### Conda
One can also use `conda`, or other DOLFINx docker images, and install the `LVPP`-package with
```bash
python3 -m pip install git+https://https://github.com/METHODS-Group/ProximalGalerkin
```
See for instance [environment.yml](../../environment.yml) for a DOLFINx conda environment.

## Figure 4

The mesh can be generated with
```bash
python3 generate_mesh.py
```
and is placed in `"meshes/half_sphere.xdmf"`.
Next, run the proximal Galerkin method with

```bash
python3 signorini_dolfinx.py --alpha_0=0.005 --degree=2 --disp=-0.10 --n-max-iterations=250 --alpha_scheme=doubling --output=small_disp file --filename=meshes/half_sphere.xdmf
python3 signorini_dolfinx.py --alpha_0=0.005 --degree=2 --disp=-0.15 --n-max-iterations=250 --alpha_scheme=doubling --output=medium_disp file --filename=meshes/half_sphere.xdmf
python3 signorini_dolfinx.py --alpha_0=0.005 --degree=2 --disp=-0.20 --n-max-iterations=250 --alpha_scheme=doubling --output=large_disp file --filename=meshes/half_sphere.xdmf
```
