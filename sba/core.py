import itertools
import numpy as np

from sba.indices import Indices
from sba.utils import check_args, check_weights, identities2x2


def calc_epsilon(x_true, x_pred):
    return x_true - x_pred


def calc_epsilon_a(indices, A, epsilon, weights):
    m = indices.n_viewpoints

    n_pose_params = A.shape[2]
    epsilon_a = np.zeros((m, n_pose_params))

    for j in range(m):
        for ij in indices.points_by_viewpoint(j):
            epsilon_a[j] += np.dot(np.dot(A[ij].T, weights[ij]), epsilon[ij])
    return epsilon_a


def calc_epsilon_b(indices, B, epsilon, weights):
    n = indices.n_points
    n_point_params = B.shape[2]

    epsilon_b = np.zeros((n, n_point_params))
    for i in range(n):
        for ij in indices.viewpoints_by_point(i):
            epsilon_b[i] += np.dot(np.dot(B[ij].T, weights[ij]), epsilon[ij])
    return epsilon_b


def calc_XTWX(XS, weights):
    XTWX = np.zeros((XS.shape[2], XS.shape[2]))
    for X, weight in zip(XS, weights):
        XTWX += np.dot(np.dot(X.T, weight), X)
    return XTWX


def calc_Uj(Aj, weights):
    return calc_XTWX(Aj, weights)


def calc_Vi(Bi, weights):
    return calc_XTWX(Bi, weights)


def calc_U(indices, A, weights):
    n_pose_params = A.shape[2]
    m = indices.n_viewpoints

    U = np.empty((m, n_pose_params, n_pose_params))

    for j in range(m):
        I = indices.points_by_viewpoint(j)
        U[j] = calc_Uj(A[I], weights[I])
    return U


def calc_V_inv(indices, B, weights):
    n_point_params = B.shape[2]
    n = indices.n_points

    V_inv = np.empty((n, n_point_params, n_point_params))

    for i in range(n):
        J = indices.viewpoints_by_point(i)
        Vi = calc_Vi(B[J], weights[J])
        V_inv[i] = np.linalg.pinv(Vi)
    return V_inv


def calc_W(indices, A, B, weights):
    assert(A.shape[0] == B.shape[0])

    n_pose_params, n_point_params = A.shape[2], B.shape[2]

    W = np.empty((indices.n_visible, n_pose_params, n_point_params))

    for index in range(indices.n_visible):
        W[index] = np.dot(np.dot(A[index].T, weights[index]), B[index])

    return W


def calc_Y(indices, W, V_inv):
    Y = np.copy(W)
    for i in range(indices.n_points):
        Vi_inv = V_inv[i]
        for ij in indices.viewpoints_by_point(i):
            Y[ij] = np.dot(Y[ij], Vi_inv)
    return Y


def calc_S(indices, U, Y, W):
    m = indices.n_viewpoints
    n_pose_params = U.shape[1]

    def block(index):
        return slice(n_pose_params * index, n_pose_params * (index + 1))

    S = np.zeros((m * n_pose_params, m * n_pose_params))

    for j, k in itertools.product(range(m), range(m)):
        indices_j, indices_k = indices.shared_point_indices(j, k)

        if len(indices_j) == 0 and len(indices_k) == 0:
            continue

        if j == k:
            S[block(j), block(k)] += U[j]

        # sum(np.dot(Y[ij], W[ik].T) for ij, ik in zip(indices_j, indices_k))
        S[block(j), block(k)] -= np.einsum('ijk,ilk->jl',
                                           Y[indices_j], W[indices_k])

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

    delta_b = np.copy(epsilon_b)
    for i in range(indices.n_points):
        J = indices.viewpoints_by_point(i)
        delta_b[i] = delta_b[i] - np.sum(d[J], axis=0)
        delta_b[i] = np.dot(V_inv[i], delta_b[i])
    return delta_b


class SBA(object):
    """
    The constructor takes two arguments: `viewpoint_indices` and
    `point_indices`.

    In general, not all 3D points can be observed from all viewpoints.
    Some points cannot be observed because of occlusion, motion blur, etc.

    We consider an example that we have four 3D points
    :math:`\{\mathbf{p}_0, \mathbf{p}_2, \mathbf{p}_3, \mathbf{p}_4\}`
    that are observed from three cameras under the condition that:

    - All 3D points can be observed from the zeroth viewpoint.
    - :math:`\{\mathbf{p}_0, \mathbf{p}_2, \mathbf{p}_3\}` can be observed from
      the first viewpoint.
    - :math:`\{\mathbf{p}_1, \mathbf{p}_2\}` can be observed from the second
      viewpoint.

    Then, `viewpoint_indices` and `point_indices` should be the following:

    .. code-block:: python

        viewpoint_indices = [0, 0, 0, 0, 1, 1, 1, 2, 2]
            point_indices = [0, 1, 2, 3, 0, 2, 3, 1, 2]

    Args:
        viewpoint_indices (list of ints), size n_keypoints:
            Array of viewpoint indices.
        point_indices (list of ints), size n_keypoints:
            Array of point indices.
        check_args (bool, optional):
            | `SBA.compute` checks if given arguments are satisfying
              the condition that the approximated Hessian
              :math:`J^{\\top} J` is invertible.
            | This can be disabled by setting `check_args=False`.
    """

    def __init__(self, viewpoint_indices, point_indices, weights=None, do_check_args=True):
        self.indices = Indices(viewpoint_indices, point_indices)
        self.do_check_args = do_check_args

    def compute(self, x_true, x_pred, A, B):
        """
        Calculate a Gauss-Newton update.
        Elements of the arguments correspond to argument arrays of the
        constructor.
        For example, if the index arrays are like below,

        .. code-block:: python

            viewpoint_indices = [0, 0, 0, 0, 1, 1, 1, 2, 2]
            point_indices     = [0, 1, 2, 3, 0, 2, 3, 1, 2]

        then `x_true` should be

        .. math::
            \\mathbf{x}_{true} = \\begin{bmatrix}
                \\mathbf{x}_{00} & \\mathbf{x}_{01} & \\mathbf{x}_{02} &
                \\mathbf{x}_{03} & \\mathbf{x}_{10} & \\mathbf{x}_{12} &
                \\mathbf{x}_{13} & \\mathbf{x}_{21} & \\mathbf{x}_{22}
            \\end{bmatrix}^{\\top}.

        Other arguments also should follow this manner.

        Args:
            x_true (np.ndarray), shape (n_keypoints, 2):
                Observed 2D keypoints of shape.
            x_pred (np.ndarray), shape (n_keypoints, 2):
                2D keypoints predicted by a projection function
            A (np.ndarray), shape (n_keypoints, 2, n_pose_params):
                | Jacobian with respect to pose parameters.
                | Each block `A[index]` represents a jacobian of
                  `x_pred[index]` with respect to the corresponding
                  pose parameter.
            B (np.ndarray), shape (n_keypoints, 2, 3):
                | Jacobian with respect to 3D points.
                | Each block `B[index]` represents a jacobian of
                  `x_pred[index]` with respect to a 3D point coordinate.

        Returns:
            (tuple):
                delta_a (np.ndarray), shape (n_viewpoints, n_pose_params):
                    Update of pose parameters.
                delta_b (np.ndarray), shape (n_points, n_point_params):
                    Update of 3D points.
        """
        if self.do_check_args:
            check_args(self.indices, x_true, x_pred, A, B)

        U = calc_U(self.indices, A, self.weights)
        V_inv = calc_V_inv(self.indices, B, self.weights)
        W = calc_W(self.indices, A, B, self.weights)
        Y = calc_Y(self.indices, W, V_inv)
        S = calc_S(self.indices, U, Y, W)
        epsilon = calc_epsilon(x_true, x_pred)
        epsilon_a = calc_epsilon_a(self.indices, A, epsilon, self.weights)
        epsilon_b = calc_epsilon_b(self.indices, B, epsilon, self.weights)
        e = calc_e(self.indices, Y, epsilon_a, epsilon_b)
        delta_a = calc_delta_a(S, e)
        delta_b = calc_delta_b(self.indices, V_inv, W, epsilon_b, delta_a)

        return delta_a, delta_b
