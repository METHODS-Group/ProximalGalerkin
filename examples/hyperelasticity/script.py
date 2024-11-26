from petsc4py import PETSc
from mpi4py import MPI
import dolfinx.fem.petsc
import basix.ufl
import ufl
from expm import expm
import numpy

L = 1  # diameter of domain
h = 0.1  # height

basen = 10  # resolution of base mesh
mesh = dolfinx.mesh.create_box(
    MPI.COMM_WORLD,  [[0, 0, 0], [L, h, h]], [basen, basen, basen])

v_el = basix.ufl.element("Lagrange", mesh.basix_cell(),
                         1, shape=(mesh.geometry.dim, ))
t_el = basix.ufl.element("Lagrange", mesh.basix_cell(), 1,
                         shape=(mesh.geometry.dim, mesh.geometry.dim))
m_el = basix.ufl.mixed_element([v_el, t_el])
Z = dolfinx.fem.functionspace(mesh, m_el)

z = dolfinx.fem.Function(Z)
(u, psi) = ufl.split(z)
(v, phi) = ufl.split(ufl.TestFunction(Z))

z_prev = dolfinx.fem.Function(Z, name="PreviousContinuationSolution")
(_, psi_prev) = ufl.split(z_prev)
z_iter = dolfinx.fem.Function(Z, name="PreviousLVPPSolution")
(u_iter, psi_iter) = ufl.split(z_iter)

# Define strain measures
I = ufl.Identity(mesh.geometry.dim)  # the identity matrix
F = I + ufl.grad(u)  # the deformation gradient
C = F.T*F       # the right Cauchy-Green tensor
E = 0.5*(C - I)  # the Green-Lagrange strain tensor

# Define strain energy density
E = ufl.variable(E)
E_, nu = 1000000.0, 0.3
mu = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(E_/(2*(1 + nu))))
lmbda = dolfinx.fem.Constant(
    mesh, dolfinx.default_scalar_type(E_*nu/((1 + nu)*(1 - 2*nu))))
W = lmbda/2*(ufl.tr(E)**2) + mu*ufl.tr(E*E)

# Define Piola-Kirchoff stress tensors
S = ufl.diff(W, E)  # the second Piola-Kirchoff stress tensor
P = F*S        # the first Piola-Kirchoff stress tensor

B = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(
    (0, 0, -1000)))  # Body force per unit volume

left_facets = dolfinx.mesh.locate_entities_boundary(
    mesh, mesh.topology.dim-1, lambda x: x[0] < 1e-10)
right_facets = dolfinx.mesh.locate_entities_boundary(
    mesh, mesh.topology.dim-1, lambda x: x[0] > L - 1e-10)


V, V_to_Z = Z.sub(0).collapse()
eps = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0))


def eps_func(x):
    return numpy.full(x.shape[1], float(eps))


epsilon = dolfinx.fem.Function(V)
epsilon.sub(0).interpolate(eps_func)

zero = dolfinx.fem.Function(V)
bcl = dolfinx.fem.dirichletbc(zero, dolfinx.fem.locate_dofs_topological(
    (Z.sub(0), V), mesh.topology.dim-1, left_facets), Z.sub(0))

bcr = dolfinx.fem.dirichletbc(epsilon, dolfinx.fem.locate_dofs_topological(
    (Z.sub(0), V), mesh.topology.dim-1, right_facets), Z.sub(0))
bcs = [bcl, bcr]

alpha = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0))

dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 10})

G = (
    + ufl.inner(P, ufl.grad(v))*dx
    - ufl.inner(B, v)*dx
    + ufl.inner(psi, ufl.grad(v))*dx
    - ufl.inner(psi_prev, ufl.grad(v))*dx
    + ufl.inner(ufl.grad(u), phi)*dx
    + ufl.inner(I, phi)*ufl.dx
    - ufl.inner(expm(alpha*psi), phi)*dx
)

sp = {"snes_type": "newtonls",
      "snes_atol": 1.0e-8,
      "snes_monitor": None,
      "snes_linesearch_type": "l2",
      "snes_linesearch_monitor": None}

v_out = dolfinx.fem.Function(V, name="Displacement")
bp = dolfinx.io.VTXWriter(mesh.comm, "output/solution.bp",
                          [v_out], engine="BP5", mesh_policy=dolfinx.io.VTXMeshPolicy.reuse)

J = ufl.derivative(G, z)
J_compiled = dolfinx.fem.form(J)
F_compiled = dolfinx.fem.form(G)


class NonlinearPDE_SNESProblem:
    def __init__(self, F, J, u, bcs):
        self.L = dolfinx.fem.form(F)
        self.a = dolfinx.fem.form(J)
        self.bcs = bcs
        self._F, self._J = None, None
        self.u = u

    def F(self, snes, x, F):
        """Assemble residual vector."""
        from petsc4py import PETSc

        from dolfinx.fem.petsc import apply_lifting, assemble_vector, set_bc

        x.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                      mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.x.petsc_vec)
        self.u.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        with F.localForm() as f_local:
            f_local.set(0.0)
        assemble_vector(F, self.L)
        apply_lifting(F, [self.a], bcs=[self.bcs], x0=[x], alpha=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)
        set_bc(F, self.bcs, x, -1.0)

    def J(self, snes, x, J, P):
        """Assemble Jacobian matrix."""
        from dolfinx.fem.petsc import assemble_matrix

        J.zeroEntries()
        assemble_matrix(J, self.a, bcs=self.bcs)
        J.assemble()


# Create Newton solver and solve
b = dolfinx.fem.Function(Z)
J = dolfinx.fem.petsc.create_matrix(J_compiled)
problem = NonlinearPDE_SNESProblem(F=F_compiled, J=J_compiled, u=z, bcs=bcs)
snes = PETSc.SNES().create(mesh.comm)
snes.setFunction(problem.F, b.x.petsc_vec)
snes.setJacobian(problem.J, J)

snes.setTolerances(rtol=1.0e-9, max_it=10)
snes.getKSP().setType("preonly")
snes.getKSP().setTolerances(rtol=1.0e-9)
snes.getKSP().getPC().setType("lu")

sp = {"snes_type": "newtonls",
      "snes_atol": 1.0e-8,
      "snes_monitor": None,
      "snes_linesearch_type": "l2",
      "snes_linesearch_monitor": None}
opts = PETSc.Options()
for key, value in sp.items():
    opts[key] = value
snes.setFromOptions()
snes.view()
NFAIL_MAX = 50


diff = dolfinx.fem.form(ufl.inner(u-u_iter, u-u_iter)*dx)
diff_z = dolfinx.fem.form(ufl.inner(z-z_prev, z-z_prev)*dx)

# For SNES line search to function correctly it is necessary that the
# u.x.petsc_vec in the Jacobian and residual is *not* passed to
# snes.solve.
x = z.x.petsc_vec.copy()
for i, eps_ in enumerate(numpy.linspace(0, 0.5, 100)):
    eps.value = -eps_
    epsilon.sub(0).interpolate(eps_func)

    x.array_w[:] = z.x.petsc_vec.array_r[:]
    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    alpha.value = 1
    z_iter.x.array[:] = z.x.array
    print(f"Solving for eps = {eps_:.3e}", flush=True)

    k = 1
    r = 2
    nfail = 0
    while True:
        try:
            print(
                f"  Attempting eps = {eps_:.3e} k = {k} α = {float(alpha)}", flush=True)
            snes.solve(None, x)
            z.x.petsc_vec.copy(x)
            z.x.scatter_forward()
            if snes.getIterationNumber() == 0:
                # solver didn't actually get to do any work,
                # we've just reduced alpha so much that the initial guess
                # satisfies the PDE
                raise RuntimeError("Solver did not converge")
        except RuntimeError:
            nfail += 1
            print(
                f"  Failed eps = {eps_} k = {k} α = {float(alpha)}.", flush=True)
            alpha.value *= 0.5

            if k == 1:
                z.x.array[:] = z_prev.x.array
            else:
                z.x.array[:] = z_iter.x.array[:]

            if nfail >= NFAIL_MAX:
                print(f"  Giving up.", flush=True)
                break
            else:
                continue
        v_out.x.array[:] = z.x.array[V_to_Z]
        # Termination
        nrm = mesh.comm.allreduce(
            dolfinx.fem.assemble_scalar(diff), op=MPI.SUM)
        print(
            f"  Solved eps = {eps_:.3e} k = {k} α = {float(alpha)}. ||u_{k} - u_{k-1}|| = {nrm}", flush=True)
        if nrm < 1.0e-8:
            break

        # Update alpha
        if snes.getIterationNumber() <= 4:
            alpha.value *= r
        elif snes.getIterationNumber() >= 10:
            alpha.value /= r

        # Update z_iter
        z_iter.x.array[:] = z.x.array

        k += 1

    # Check for failure
    if k == 1 and mesh.comm.allreduce(dolfinx.fem.assemble_scalar(diff_z)) == 0.0:
        break

    if nfail == NFAIL_MAX:
        break

    print(f"Succeeded in {k} iterations.")
    bp.write(i)

    assert snes.getConvergedReason() > 0
    assert snes.getIterationNumber() < 6

bp.close()
J.destroy()
snes.destroy()
