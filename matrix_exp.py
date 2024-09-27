from mpi4py import MPI
import dolfinx
import ufl

def expm(A):
    """Exponential of a matrix.

    Implement explicit formulae found in

    doi:10.1109/9.233156
    doi:10.1016/s0024-3795(97)80028-6
    """

    assert len(A.ufl_shape) == 2
    assert A.ufl_shape[0] == A.ufl_shape[1]
    d = A.ufl_shape[0]

    if d == 1:
        return ufl.as_tensor([[ufl.exp(A[0, 0])]])
    elif d == 2:
        return expm2(A)
    elif d == 3:
        return expm3(A)
    else:
        raise NotImplementedError

def expm2(A):
    """
    Corollary 2.4 of doi:10.1109/9.233156

    Note typo in the paper, in the prefactor in the equation under (i)·
    """

    (a, b) = A[0, :]
    (c, d) = A[1, :]

    e = (a - d)**2 + 4*b*c

    # Branch 1: e == 0
    cond1 = ufl.eq(e, 0)
    branch1 = ufl.exp((a + d)/2) * ufl.as_tensor([[1 + (a - d)/2, b],
                                          [c, 1 - (a - d)/2]])

    # Branch 2: e > 0
    cond2 = ufl.gt(e, 0)
    delta = (1/2) * ufl.sqrt(e)
    branch2 = ufl.exp((a + d)/2) * ufl.as_tensor([[ufl.cosh(delta) + (a - d)/2 * ufl.sinh(delta) / delta, b * ufl.sinh(delta) / delta],
                                          [c * ufl.sinh(delta) / delta, ufl.cosh(delta) - (a - d)/2 * ufl.sinh(delta) / delta]])

    # Branch 3: e < 0
    #cond3 = ufl.lt(e, 0)
    eps = (1/2) * ufl.sqrt(-e)
    branch3 = ufl.exp((a + d)/2) * ufl.as_tensor([[ufl.cos(eps) + (a - d)/2 * ufl.sin(eps) / eps, b * ufl.sin(eps) / eps],
                                          [c * ufl.sin(eps) / eps, ufl.cos(eps) - (a - d)/2 * ufl.sin(eps) / eps]])


    return ufl.conditional(cond1, branch1,
           ufl.conditional(cond2, branch2,
                              branch3))

def expm3(A):
    raise NotImplementedError

if __name__ == "__main__":
    import numpy as np
    import scipy.linalg
    import basix.ufl

    nodes = np.array([[0.0, 0.0], [1.0, 0.0],[0.0, 1.0]], dtype=np.float64)
    connectivity = np.array([[0, 1, 2]], dtype=np.int64)
    c_el = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(nodes.shape[1],)))
    domain = dolfinx.mesh.create_mesh(MPI.COMM_SELF, connectivity, nodes, c_el)
    value_shape = (domain.geometry.dim, domain.geometry.dim)
    V = dolfinx.fem.functionspace(domain, ("DG", 0, value_shape))
    A = dolfinx.fem.Function(V)
    expA = dolfinx.fem.Function(V)

    tensors = (
               [[5, 3], [0, 5]],    # branch 1
               [[3, 5], [4, 2]],    # branch 2
               [[10, 2], [-2, 8]],  # branch 3
               np.random.normal(size=(2, 2)),
               )

    for tensor in tensors:
        (a, b) = tensor[0][0], tensor[0][1]
        (c, d) = tensor[1][0], tensor[1][1]
        e = (a - d)**2 + 4*b*c
        print("e: ", e)
        t_ufl = ufl.as_tensor(tensor)
        m_ufl = expm(A)

        tensor_expr = dolfinx.fem.Expression(t_ufl, V.element.interpolation_points(), comm=domain.comm)
        m_expr = dolfinx.fem.Expression(m_ufl, V.element.interpolation_points(), comm=domain.comm)
        A.interpolate(tensor_expr)
        expA.interpolate(m_expr)
        np.testing.assert_allclose(scipy.linalg.expm(A.x.array.reshape(*value_shape)), expA.x.array.reshape(*value_shape))