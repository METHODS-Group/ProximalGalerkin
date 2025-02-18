# Example 11 (Figure 11): The Eikonal Equation

We have provided code for this example for both the `MFEM` and `DOLFINx` Docker containers.

## MFEM

To reproduce the Möbius strip solution in Figure 11, you first need to copy [./examples/eikonal/ex40.cpp](./examples/eikonal/ex40.cpp) into the `MFEM` examples folder (`/home/euler/mfem/examples/`) and then calling `make ex40` and `./ex40 -step 10.0 -mi 10`. This following code will execute to entire process:

```bash
docker run -it --rm -v ./examples/eikonal:/home/euler/shared -w /home/euler/mfem --rm --entrypoint=/bin/bash ghcr.io/methods-group/proximalgalerkin-mfem:main
cp /home/euler/shared/ex40.cpp /home/euler/mfem/examples/
cd examples && make ex40
./ex40 -step 10.0 -mi 10
```

To reproduce the results in Figure 11 for the two geometries (i.e., the [Star](https://github.com/mfem/mfem/blob/master/data/star.mesh)
and [Ball](https://github.com/mfem/mfem/blob/master/data/ball-nurbs.mesh)), you should compile the [official examples](https://mfem.org/examples/) `ex40.cpp` or `ex40p.cpp` without copying any files from this repository

```bash
cd examples && make ex40
# Star Geometry
./ex40 -step 10.0 -mi 10
# Ball Geometry
./ex40 -step 10.0 -mi 10 -m ../data/ball-nurbs.mesh
```

## DOLFINx

The `DOLFINx` implementation requires converting the `MFEM` Möbius strip mesh [mobius-strip.mesh](https://github.com/mfem/mfem/blob/master/data/mobius-strip.mesh).
To this end, run the following commands from the root of this repository:

```bash
docker run -it --rm -v ./examples/eikonal:/home/euler/shared -w /home/euler/mfem --rm --entrypoint=/bin/bash ghcr.io/methods-group/proximalgalerkin-mfem:main
cp /home/euler/shared/convert_mesh.cpp /home/euler/mfem/examples/
cd examples && make convert_mesh
./convert_mesh --mesh ../data/mobius-strip.mesh
cp -r  mobius-strip.mesh/ ../../shared/
```

The `DOLFINx` code is then executed by calling:

```bash
python3 eikonal_dolfinx.py
```
