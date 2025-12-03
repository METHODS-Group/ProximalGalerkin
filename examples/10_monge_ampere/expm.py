from ufl import (
    Identity,
    acos,
    as_tensor,
    conditional,
    cos,
    cosh,
    dot,
    eq,
    exp,
    gt,
    lt,
    pi,
    sin,
    sinh,
    sqrt,
)


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
        return as_tensor([[exp(A[0, 0])]])
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


def expm3(A):
    """
    Compute the exponential of a real-valued 3x3 matrix using explicit formulas from
    doi:10.1016/s0024-3795(97)80028-6

    Note (28'), the key formula, has a sign error. The coefficient of A^2 and of I should be negated.
    """

    I = Identity(3)

    # Extract elements of the 3x3 matrix
    (a11, a12, a13) = A[0, :]
    (a21, a22, a23) = A[1, :]
    (a31, a32, a33) = A[2, :]

    # Compute trace and coefficients of characteristic polynomial
    p = -(a11 + a22 + a33)  # trace
    q = (a11 * a22 + a11 * a33 + a22 * a33) - (a12 * a21 + a13 * a31 + a23 * a32)
    r = -(a11 * a22 * a33 + a12 * a23 * a31 + a13 * a21 * a32) + (
        a13 * a22 * a31 + a11 * a23 * a32 + a12 * a21 * a33
    )  # determinant

    # Depress the cubic
    a = q - p**2 / 3
    b = 2 * p**3 / 27 - p * q / 3 + r

    # Compute roots of t^3 + 0*t^2 + a*t + b
    Delta = (a / 3) ** 3 + (b / 2) ** 2

    # Case 1: Delta negative, three roots real and unequal
    case_neg = lt(Delta, 0)
    theta = acos(-b / 2 / sqrt(-((a / 3) ** 3)))
    x1_neg = 2 * sqrt(-a / 3) * cos(theta / 3) - p / 3
    x2_neg = 2 * sqrt(-a / 3) * cos((theta + 2 * pi) / 3) - p / 3
    x3_neg = 2 * sqrt(-a / 3) * cos((theta + 4 * pi) / 3) - p / 3
    val_neg = (
        exp(x1_neg) * dot(A - x2_neg * I, A - x3_neg * I) / ((x1_neg - x2_neg) * (x1_neg - x3_neg))
        + exp(x2_neg)
        * dot(A - x1_neg * I, A - x3_neg * I)
        / ((x2_neg - x1_neg) * (x2_neg - x3_neg))
        + exp(x3_neg)
        * dot(A - x1_neg * I, A - x2_neg * I)
        / ((x3_neg - x1_neg) * (x3_neg - x2_neg))
    )

    # Case 2: Delta zero, either triple root or two roots equal one different
    # case_zero = eq(Delta, 0)
    tol = 1e-100
    case_zero = lt(real(Delta), tol)
    x1_zero = (
        conditional(
            lt(abs(a), tol), 0, conditional(lt(b, 0), +2 * (-b / 2) ** (1 / 3), -2 * (b / 2) ** (1 / 3))
        )
        - p / 3
    )
    x2_zero = (
        conditional(
            lt(abs(a), tol), 0, conditional(lt(b, 0), -1 * (-b / 2) ** (1 / 3), +1 * (b / 2) ** (1 / 3))
        )
        - p / 3
    )
    x3_zero = (
        conditional(
            eq(a, 0), 0, conditional(lt(b, 0), -1 * (-b / 2) ** (1 / 3), +1 * (b / 2) ** (1 / 3))
        )
        - p / 3
    )

    # if a == 0: triple root
    val_zero_one = exp(x1_zero) * (
        1 / 2 * dot(A - x1_zero * I, A - x1_zero * I) + (A - x1_zero * I) + I
    )

    # otherwise, just two distinct eigenvalues
    val_zero_two = (
        exp(x2_zero) / (x2_zero - x1_zero)
        - (exp(x2_zero) - exp(x1_zero)) / (x2_zero - x1_zero) ** 2
    ) * dot(A - x2_zero * I, A - x2_zero * I) + exp(x2_zero) * ((A - x2_zero * I) + I)
    val_zero = conditional(eq(a, 0), val_zero_one, val_zero_two)

    # Case 3: Delta positive, one real root, and two complex roots
    # This is the difficult one. Have to do it all in purely real arithmetic, argh
    case_pos = gt(Delta, 0)
    u_cube = (-b / 2) + sqrt(Delta)
    v_cube = (-b / 2) - sqrt(Delta)
    u = conditional(gt(u_cube, 0), u_cube ** (1 / 3), -((-u_cube) ** (1 / 3)))
    v = conditional(gt(v_cube, 0), v_cube ** (1 / 3), -((-v_cube) ** (1 / 3)))
    x1_pos = u + v - p / 3
    r = -(u + v) / 2 - p / 3
    c = sqrt(3) / 2 * (u - v)

    # Use variables from sympy calculation
    lam = x1_pos

    coeffAsq = (
        c * exp(lam) - c * exp(r) * cos(c) - lam * exp(r) * sin(c) + r * exp(r) * sin(c)
    ) / (c * (c**2 + lam**2 - 2 * lam * r + r**2))
    coeffA = (
        c**2 * exp(r) * sin(c)
        - 2 * c * r * exp(lam)
        + 2 * c * r * exp(r) * cos(c)
        + lam**2 * exp(r) * sin(c)
        - r**2 * exp(r) * sin(c)
    ) / (c * (c**2 + lam**2 - 2 * lam * r + r**2))
    coeffI = (
        c**3 * exp(lam)
        - c**2 * lam * exp(r) * sin(c)
        + c * lam**2 * exp(r) * cos(c)
        - 2 * c * lam * r * exp(r) * cos(c)
        + c * r**2 * exp(lam)
        - lam**2 * r * exp(r) * sin(c)
        + lam * r**2 * exp(r) * sin(c)
    ) / (c * (c**2 + lam**2 - 2 * lam * r + r**2))

    val_pos = coeffAsq * dot(A, A) + coeffA * A + coeffI * I

    out = conditional(case_neg, val_neg, conditional(case_zero, val_zero, val_pos))

    return out


if __name__ == "__main__":
    import numpy
    import scipy.linalg
    from firedrake import *

    #    mesh = UnitTriangleMesh()
    #    V = TensorFunctionSpace(mesh, "DG", 0)
    #    A = Function(V)
    #    expA = Function(V)
    #
    #    tensors = (
    #               [[5, 3], [0, 5]],    # branch 1
    #               [[3, 5], [4, 2]],    # branch 2
    #               [[10, 2], [-2, 8]],  # branch 3
    #               numpy.random.normal(size=(2, 2)),
    #               )
    #
    #    for tensor in tensors:
    #        A.interpolate(as_tensor(tensor))
    #        expA.interpolate(expm(A))
    #        numpy.testing.assert_allclose(expA.dat.data, scipy.linalg.expm(A.dat.data))

    mesh = UnitTetrahedronMesh()
    V = TensorFunctionSpace(mesh, "DG", 0)
    A = Function(V)
    expA = Function(V)

    tensors = (
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 1, 0], [0, 1, 1], [0, 0, 1]],
        [[1, 0, 0], [0, 2, 0], [0, 0, 3]],
        [[1, 0, 0], [0, 2, 0], [0, 0, 2]],
        [[2, -1, 0], [1, 2, 0], [0, 0, 3]],
        numpy.random.normal(size=(3, 3)),
        numpy.random.normal(size=(3, 3)),
        numpy.random.normal(size=(3, 3)),
        numpy.random.normal(size=(3, 3)),
    )

    for tensor in tensors:
        A.interpolate(as_tensor(tensor))
        expA.interpolate(expm(A))
        print("-" * 80)
        print("Input matrix:           ", repr(A.dat.data))
        print("Computed with our expm: ", repr(expA.dat.data))
        print("Computed with scipy:    ", repr(scipy.linalg.expm(A.dat.data)))
        numpy.testing.assert_allclose(expA.dat.data, scipy.linalg.expm(A.dat.data))
