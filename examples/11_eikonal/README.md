# Example 11 (Figure 11): The Eikonal Equation

We have provided code for this example for both the `MFEM` and `DOLFINx` Docker containers.

## Using the Proximal-Galerkin docker containers

### MFEM
If using the `ghcr.io/methods-group/proximalgalerkin:v0.2.0` container, you can navigate to
`/root/LVPP/mfem/examples`, where there is a file called `ex40_taylor_hood.cpp`.

This script reproduces the  Möbius strip solution in Figure 11.
```bash
make ex40_taylor_hood
./ex40_taylor_hood -step 10.0 -mi 10
```

The results for the two other geoemtries can be reproduced with

```bash
cd examples && make ex40p
# Star Geometry
./ex40p -step 10.0 -mi 10
# Ball Geometry
./ex40p -step 10.0 -mi 10 -m ../data/ball-nurbs.mesh
```

### DOLFINx

The `DOLFINx` implementation requires converting the `MFEM` Möbius strip mesh [mobius-strip.mesh](https://github.com/mfem/mfem/blob/master/data/mobius-strip.mesh).
To convert the mesh, navigate to `/root/LVPP/mfem/examples`.
Then run the following commands
```bash
make convert_mesh
./convert_mesh --mesh ../data/mobius-strip.mesh
```
Next, navigate back to this folder (either the shared volume or `/root/LVPP/examples/11_eikonal`) and run 

```bash
python3 eikonal_dolfinx.py --mesh-dir=/root/LVPP/mfem/examples/mobius-strip.mesh
```

## Using a local installation of MFEM
Copy the file `ex40.cpp` into your `mfem/examples` repository, compile it with `make ex40.cpp` and run as described above.