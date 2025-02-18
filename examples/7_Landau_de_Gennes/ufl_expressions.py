from ufl import (
    as_tensor,
    conditional,
    cos,
    cosh,
    eq,
    exp,
    gt,
    lt,
    sin,
    sinh,
    sqrt,
)


def expm2(A):
    """
    Corollary 2.4 of doi:10.1109/9.233156

    Note typo in the paper, in the prefactor in the equation under (i)·
    """

    (a, b) = A[0, :]
    (c, d) = A[1, :]

    e = (a - d) ** 2 + 4 * b * c

    # Branch 1: e == 0
    cond1 = eq(e, 0)
    branch1 = exp((a + d) / 2) * as_tensor([[1 + (a - d) / 2, b], [c, 1 - (a - d) / 2]])

    # Branch 2: e > 0
    cond2 = gt(e, 0)
    delta = (1 / 2) * sqrt(e)
    branch2 = exp((a + d) / 2) * as_tensor(
        [
            [cosh(delta) + (a - d) / 2 * sinh(delta) / delta, b * sinh(delta) / delta],
            [c * sinh(delta) / delta, cosh(delta) - (a - d) / 2 * sinh(delta) / delta],
        ]
    )

    # Branch 3: e < 0
    cond3 = lt(e, 0)
    eps = (1 / 2) * sqrt(-e)
    branch3 = exp((a + d) / 2) * as_tensor(
        [
            [cos(eps) + (a - d) / 2 * sin(eps) / eps, b * sin(eps) / eps],
            [c * sin(eps) / eps, cos(eps) - (a - d) / 2 * sin(eps) / eps],
        ]
    )

    return conditional(cond1, branch1, conditional(cond2, branch2, branch3))


if __name__ == "__main__":
    import numpy
    import scipy.linalg
    from firedrake import *

    mesh = UnitTriangleMesh()
    V = TensorFunctionSpace(mesh, "DG", 0)
    A = Function(V)
    expA = Function(V)

    tensors = (
        [[5, 3], [0, 5]],  # branch 1
        [[3, 5], [4, 2]],  # branch 2
        [[10, 2], [-2, 8]],  # branch 3
        numpy.random.normal(size=(2, 2)),
    )

    for tensor in tensors:
        A.interpolate(as_tensor(tensor))
        expA.interpolate(expm2(A))
        numpy.testing.assert_allclose(scipy.linalg.expm(A.dat.data), expA.dat.data)
