import itertools
import numpy as np

from sba.indices import Indices


def calc_epsilon(x_true, x_pred):
    return x_true - x_pred


def calc_epsilon_a(indices, A, epsilon):
    m = indices.n_viewpoints

    n_pose_params = A.shape[2]
    epsilon_a = np.zeros((m, n_pose_params))

    for j in range(m):
        for ij in indices.points_by_viewpoint(j):
            epsilon_a[j] += np.dot(A[ij].T, epsilon[ij])
    return epsilon_a


def calc_epsilon_b(indices, B, epsilon):
    n = indices.n_points
    n_point_params = B.shape[2]

    epsilon_b = np.zeros((n, n_point_params))
    for i in range(n):
        for ij in indices.viewpoints_by_point(i):
            epsilon_b[i] += np.dot(B[ij].T, epsilon[ij])
    return epsilon_b


def calc_XtX(XS):
    XtX = np.zeros((XS.shape[2], XS.shape[2]))
    for X in XS:
        XtX += np.dot(X.T, X)
    return XtX


def calc_Uj(Aj):
    return calc_XtX(Aj)


def calc_Vi(Bi):
    return calc_XtX(Bi)


def calc_U(indices, A):
    n_pose_params = A.shape[2]
    m = indices.n_viewpoints

    U = np.empty((m, n_pose_params, n_pose_params))

    for j in range(m):
        # Aj = [Aij for i in points_by_viewpoint(j)]
        Aj = A[indices.points_by_viewpoint(j)]
        U[j] = calc_Uj(Aj)

    return U


def calc_V_inv(indices, B):
    n_point_params = B.shape[2]
    n = indices.n_points

    V_inv = np.empty((n, n_point_params, n_point_params))

    for i in range(n):
        J = indices.viewpoints_by_point(i)
        Vi = calc_Vi(B[J])
        V_inv[i] = np.linalg.inv(Vi)
    return V_inv


def calc_W(indices, A, B):
    assert(A.shape[0] == B.shape[0])

    n_pose_params, n_point_params = A.shape[2], B.shape[2]

    W = np.empty((indices.n_visible, n_pose_params, n_point_params))

    for index in range(indices.n_visible):
        W[index] = np.dot(A[index].T, B[index])

    return W


def calc_Y(indices, W, V_inv):
    Y = np.copy(W)
    for i in range(indices.n_points):
        Vi_inv = V_inv[i]
        for ij in indices.viewpoints_by_point(i):
            Y[ij] = np.dot(Y[ij], Vi_inv)
    return Y


# TODO accelerate
def calc_S(indices, U, Y, W):
    assert(Y.shape == W.shape)

    n_pose_params = U.shape[1]

    def block(index):
        return slice(n_pose_params * index, n_pose_params * (index + 1))

    m = indices.n_viewpoints
    S = np.zeros((m * n_pose_params, m * n_pose_params))

    for j, k in itertools.product(range(m), range(m)):
        indices_j, indices_k = indices.shared_point_indices(j, k)

        if len(indices_j) == 0 and len(indices_k) == 0:
            continue

        Sjk = S[block(j), block(k)]

        if j == k:
            Sjk += U[j]

        for ij, ik in zip(indices_j, indices_k):
            Sjk -= np.dot(Y[ij], W[ik].T)

        S[block(j), block(k)] = Sjk

    return S


def calc_e(indices, Y, epsilon_a, epsilon_b):
    d = np.zeros((indices.n_visible, Y.shape[1]))
    for i in range(indices.n_points):
        for ij in indices.viewpoints_by_point(i):
            d[ij] += np.dot(Y[ij], epsilon_b[i])

    e = np.copy(epsilon_a)
    for j in range(indices.n_viewpoints):
        I = indices.points_by_viewpoint(j)
        e[j] = e[j] - np.sum(d[I], axis=0)
    return e


def calc_delta_a(S, e):
    delta_a = np.linalg.solve(S, e.flatten())
    return delta_a.reshape(e.shape)


def calc_delta_b(indices, V_inv, W, epsilon_b, delta_a):
    d = np.zeros((indices.n_visible, W.shape[2]))
    for j in range(indices.n_viewpoints):
        for ij in indices.points_by_viewpoint(j):
            d[ij] = np.dot(W[ij].T, delta_a[j])

    e = np.copy(epsilon_b)
    for i in range(indices.n_points):
        J = indices.viewpoints_by_point(i)
        e[i] = e[i] - np.sum(d[J], axis=0)
        e[i] = np.dot(V_inv[i], e[i])
    return e


def check_args(indices, x_true, x_pred, A, B):
    # check the number of points
    assert(A.shape[0] == B.shape[0] == x_true.shape[0] == x_pred.shape[0])
    # check the jacobians' shape
    assert(A.shape[1] == B.shape[1] == 2)

    n_visible = x_true.shape[0]
    n_pose_params = A.shape[2]
    n_point_params = B.shape[2]

    n_rows = 2 * n_visible
    n_cols_a = n_pose_params * indices.n_viewpoints
    n_cols_b = n_point_params * indices.n_points

    # J' * J cannot be invertible if n_rows(J) < n_cols(J)
    if n_rows < n_cols_a + n_cols_b:
        raise ValueError("n_rows(J) must be greater than n_cols(J)")


class SBA(object):
    def __init__(self, viewpoint_indices, point_indices):
        self.indices = Indices(viewpoint_indices, point_indices)

    def compute(self, x_true, x_pred, A, B):
        """
        """
        check_args(self.indices, x_true, x_pred, A, B)

        U = calc_U(self.indices, A)
        V_inv = calc_V_inv(self.indices, B)
        W = calc_W(self.indices, A, B)
        Y = calc_Y(self.indices, W, V_inv)
        S = calc_S(self.indices, U, Y, W)
        epsilon = calc_epsilon(x_true, x_pred)
        epsilon_a = calc_epsilon_a(self.indices, A, epsilon)
        epsilon_b = calc_epsilon_b(self.indices, B, epsilon)
        e = calc_e(self.indices, Y, epsilon_a, epsilon_b)
        delta_a = calc_delta_a(S, e)
        delta_b = calc_delta_b(self.indices, V_inv, W, epsilon_b, delta_a)

        return delta_a, delta_b
