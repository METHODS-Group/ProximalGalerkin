import numpy as np


def gamma(theta, t):
    return (
        np.cos(theta) * (2 + t * np.cos(theta / 2)),
        np.sin(theta) * (2 + t * np.cos(theta / 2)),
        t * np.sin(theta / 2),
    )


def make_mobius_strip(M, N):
    theta = np.linspace(
        np.pi / 8,
        2 * np.pi + np.pi / 8,
        (N) + (N - 1) * (M - 2) + N - 2,
        endpoint=False,
    )
    t = np.linspace(-1, 1, N)
    points = []
    for i in theta:
        for j in t:
            x, y, z = gamma(i, j)
            points.append([x, y, z])
    return points


def create_mobius_mesh(M, degree=3):
    points = np.array(make_mobius_strip(M, degree + 1))

    cell_vertices = np.full((M, (degree + 1) ** 2), -1, dtype=np.int64)
    base = (degree) * (degree + 1)
    for i in range(M):
        cell_vertices[i, 0] = base * i
        cell_vertices[i, 1] = base * i + degree
        cell_vertices[i, 2] = base * i + (degree + 1) * degree
        cell_vertices[i, 3] = base * i + (degree + 1) ** 2 - 1
        # Edge 0
        for j in range(1, degree):
            cell_vertices[i, 4 + j - 1] = base * i + j
        # Edge 1
        for j in range(1, degree):
            cell_vertices[i, 4 + (degree - 1) + j - 1] = base * i + j * (degree + 1)
        # Edge 2
        for j in range(1, degree):
            cell_vertices[i, 4 + 2 * (degree - 1) + j - 1] = base * i + (j + 1) * (degree + 1) - 1

        # Edge 3
        for j in range(1, degree):
            cell_vertices[i, 4 + 3 * (degree - 1) + j - 1] = base * i + degree * (degree + 1) + j

        # Interior
        for j in range(1, degree):
            for k in range(1, degree):
                cell_vertices[i, 4 + 4 * (degree - 1) + (j - 1) * (degree - 1) + k - 1] = (
                    base * i + j * (degree + 1) + k
                )

    cell_vertices[M - 1, 0] = cell_vertices[M - 2, 2]
    cell_vertices[M - 1, 1] = cell_vertices[M - 2, 3]
    cell_vertices[M - 1, 2] = cell_vertices[0, 1]
    cell_vertices[M - 1, 3] = cell_vertices[0, 0]
    for j in range(1, degree):
        cell_vertices[M - 1, 4 + 3 * (degree - 1) + j - 1] = cell_vertices[0, 4 + (degree + 1) - j]
    assert (cell_vertices.flatten() >= 0).all()

    from mpi4py import MPI
    import dolfinx
    import ufl
    import basix.ufl

    # import matplotlib.pyplot as plt
    # import matplotlib

    # matplotlib.use("Tkagg")
    # fig = plt.figure()
    # ax = plt.axes(projection="3d")
    # surf = ax.scatter(
    #     points[:, 0],
    #     points[:, 1],
    #     points[:, 2],
    #     c=np.arange(points.shape[0]),
    #     cmap="viridis",
    # )
    # for i in range(points.shape[0]):
    #     ax.text(points[i, 0], points[i, 1], points[i, 2], str(i))

    # plt.show()

    assert len(np.unique(cell_vertices.flatten())) == points.shape[0]
    ud = ufl.Mesh(basix.ufl.element("Lagrange", "quadrilateral", degree, shape=(3,)))
    mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cell_vertices, points, ud)
    return mesh


if __name__ == "__main__":
    M = 4
    degree = 5
    mesh = create_mobius_mesh(M, degree=degree)
    import dolfinx.io

    with dolfinx.io.VTXWriter(mesh.comm, "manual.bp", mesh) as vtx:
        vtx.write(0.0)
