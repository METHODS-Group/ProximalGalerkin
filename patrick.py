from ufl import exp, as_tensor, eq, cosh, sinh, conditional, cos, sin, sqrt, Identity, dot, conditional, And, gt, lt, acos, pi, real

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

    e = (a - d)**2 + 4*b*c

    # Branch 1: e == 0
    cond1 = eq(real(e), 0+0)
    branch1 = exp((a + d)/2) * as_tensor([[1 + (a - d)/2, b],
                                          [c, 1 - (a - d)/2]])

    # Branch 2: e > 0
    cond2 = gt(real(e), 0)
    delta = (1/2) * sqrt(e)
    branch2 = exp((a + d)/2) * as_tensor([[cosh(delta) + (a - d)/2 * sinh(delta) / delta, b * sinh(delta) / delta],
                                          [c * sinh(delta) / delta, cosh(delta) - (a - d)/2 * sinh(delta) / delta]])

    # Branch 3: e < 0
    cond3 = lt(real(e), 0)
    eps = (1/2) * sqrt(-e)
    branch3 = exp((a + d)/2) * as_tensor([[cos(eps) + (a - d)/2 * sin(eps) / eps, b * sin(eps) / eps],
                                          [c * sin(eps) / eps, cos(eps) - (a - d)/2 * sin(eps) / eps]])


    return conditional(cond1, branch1,
           conditional(cond2, branch2,
                              branch3))


def expm3(A):
    """
    Compute the exponential of a 3x3 matrix using explicit formulas from
    doi:10.1016/s0024-3795(97)80028-6
    """

    (lambda1, lambda2, lambda3) = eigvals3(A)
    I = Identity(3)

    # Case 1. One distinct eigenvalue: equation (25)
    cond_1 = And(eq(lambda1, lambda2), eq(lambda2, lambda3))
    val_1 = exp(lambda1) * (1/2 * dot(A - lambda1*I, A - lambda1*I) + (A - lambda1*I) + I)

    # Case 2. Two distinct eigenvalues: equation (27)
    # By the convention in eigvals3, the two equal roots are lambda1 and lambda2, with
    # lambda2 repeated. So in the formula lambda -> lambda2, mu -> lambda1
    cond_2 = eq(lambda2, lambda3)
    val_2 = (exp(lambda2)/(lambda2 - lambda1) - (exp(lambda2) - exp(lambda1))/(lambda2 - lambda1)**2) * dot(A - lambda2*I, A - lambda2*I) + \
             exp(lambda2)*((A - lambda2*I) + I)


    # Case 3. Three distinct eigenvalues: equation (28)
    val_3 = exp(lambda1) * dot(A - lambda2*I, A - lambda3*I) / ((lambda1 - lambda2)*(lambda1 - lambda3)) + \
            exp(lambda2) * dot(A - lambda1*I, A - lambda3*I) / ((lambda2 - lambda1)*(lambda2 - lambda3)) + \
            exp(lambda3) * dot(A - lambda1*I, A - lambda2*I) / ((lambda3 - lambda1)*(lambda3 - lambda2))


    out = conditional(cond_1, val_1,
          conditional(cond_2, val_2, val_3))

    return out

def eigvals3(A):
    """
    Return symbolic expressions for the eigenvalues.
    """
    # Extract elements of the 3x3 matrix
    (a11, a12, a13) = A[0, :]
    (a21, a22, a23) = A[1, :]
    (a31, a32, a33) = A[2, :]

    # Compute trace and coefficients of characteristic polynomial
    p = -(a11 + a22 + a33)  # trace
    q = (a11 * a22 + a11 * a33 + a22 * a33) - (a12 * a21 + a13 * a31 + a23 * a32)
    r = - (a11 * a22 * a33 + a12 * a23 * a31 + a13 * a21 * a32) \
        + (a13 * a22 * a31 + a11 * a23 * a32 + a12 * a21 * a33)  # determinant

    # Depress the cubic
    a = q - p**2/3
    b = 2*p**3 /27 - p*q/3 + r

    # Compute roots of t^3 + 0*t^2 + a*t + b
    Delta = (a/3)**3 + (b/2)**2

    # Case 1: Delta negative, three roots real and unequal
    case_neg = lt(real(Delta), 0)
    theta = acos(-b/2 / sqrt(-(a/3)**3))
    t1_neg = 2*sqrt(-a/3)*cos(theta/3)
    t2_neg = 2*sqrt(-a/3)*cos((theta + 2*pi)/3)
    t3_neg = 2*sqrt(-a/3)*cos((theta + 4*pi)/3)

    # Case 2: Delta zero, either triple root or two roots equal one different
    #case_zero = eq(Delta, 0)
    case_zero = lt(real(Delta), 1e-14)
    t1_zero = conditional(eq(real(a), 0), 0, conditional(lt(real(b), 0), +2*(-b/2)**(1/3), +1*(b/2)**(1/3)))
    t2_zero = conditional(eq(real(a), 0), 0, conditional(lt(real(b), 0), +1*(-b/2)**(1/3), -2*(b/2)**(1/3)))
    t3_zero = conditional(eq(real(a), 0), 0, conditional(lt(real(b), 0), +1*(-b/2)**(1/3), -2*(b/2)**(1/3)))

    # Case 3: Delta positive, one real root, and two complex roots
    case_pos = gt(real(Delta), 0)
    u_cube = ((-b/2) + sqrt(Delta))
    v_cube = ((-b/2) - sqrt(Delta))
    u = conditional(gt(real(u_cube), 0), u_cube**(1/3), -(-u_cube)**(1/3))
    v = conditional(gt(real(v_cube), 0), v_cube**(1/3), -(-v_cube)**(1/3))
    t1_pos = u + v
    t2_pos = -(u + v)/2 + 1j*sqrt(3)/2 * (u - v)
    t3_pos = -(u + v)/2 - 1j*sqrt(3)/2 * (u - v)

    x1 = conditional(case_neg, t1_neg,
         conditional(case_zero, t1_zero, t1_pos)) - p/3
    x2 = conditional(case_neg, t2_neg,
         conditional(case_zero, t2_zero, t2_pos)) - p/3
    x3 = conditional(case_neg, t3_neg,
         conditional(case_zero, t3_zero, t3_pos)) - p/3

    return np.array([x1, x2, x3])


if __name__ == "__main__":
    import numpy
    import scipy.linalg

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
#        numpy.testing.assert_allclose(scipy.linalg.expm(A.dat.data), expA.dat.data)

if __name__ == "__main__":
    from mpi4py import MPI
    import dolfinx
    import scipy.linalg
    import basix.ufl
    import numpy as np
    import ufl

    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],[0.0,1.0,0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    connectivity = np.array([[0, 1, 2, 3]], dtype=np.int64)
    c_el = ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(nodes.shape[1],)))
    domain = dolfinx.mesh.create_mesh(MPI.COMM_SELF, connectivity, nodes, c_el)
    value_shape = (domain.geometry.dim, domain.geometry.dim)
    V = dolfinx.fem.functionspace(domain, ("DG", 0, value_shape))
    A = dolfinx.fem.Function(V)
    expA = dolfinx.fem.Function(V)


    tensors = (
               [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
               #[[1, 1, 0], [0, 1, 1], [0, 0, 1]],
               #[[1, 0, 0], [0, 2, 0], [0, 0, 3]],
               #[[1, 0, 0], [0, 2, 0], [0, 0, 2]],
               #numpy.random.normal(size=(3, 3)),
               #numpy.random.normal(size=(3, 3)),
               #numpy.random.normal(size=(3, 3)),
               #numpy.random.normal(size=(3, 3)),
               )

    for tensor in tensors:
        ten = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(tensor))
        expr = dolfinx.fem.Expression(ten, V.element.interpolation_points())
        A.interpolate(expr)
        expr_us = dolfinx.fem.Expression(expm(A), V.element.interpolation_points())
        expA.interpolate(expr_us)
        print("-"*80)
        print("Input matrix:           ", repr(A.x.array))
        print("Computed with our expm: ", repr(expA.x.array))
        print("Computed with scipy:    ", repr(scipy.linalg.expm(A.x.array.reshape(3,3))))
        numpy.testing.assert_allclose(scipy.linalg.expm(A.x.array), expA.x.array)