import numpy as np


def all_symmetric(XS):
    return np.allclose(XS, np.swapaxes(XS, 1, 2))


def identities2x2(n):
    I = np.zeros((n, 2, 2))
    I[:, [0, 1], [0, 1]] = 1
    return I


def can_run_ba(n_viewpoints, n_points, n_visible,
               n_pose_params, n_point_params):
    n_rows = 2 * n_visible
    n_cols_a = n_pose_params * n_viewpoints
    n_cols_b = n_point_params * n_points
    n_cols = n_cols_a + n_cols_b
    # J' * J cannot be invertible if n_rows(J) < n_cols(J)
    return n_rows >= n_cols


def check_args(indices, x_true, x_pred, A, B):
    n_visible = x_true.shape[0]
    assert(A.shape[0] == B.shape[0] == n_visible)
    assert(x_pred.shape[0] == n_visible)

    # check the jacobians' shape
    assert(A.shape[1] == B.shape[1] == 2)

    if not can_run_ba(indices.n_viewpoints, indices.n_points, n_visible,
                      n_pose_params=A.shape[2],
                      n_point_params=B.shape[2]):
        raise ValueError("n_rows(J) must be greater than n_cols(J)")


def check_weights(weights, n_visible):
    assert(weights.shape[0] == n_visible)
    assert(weights.shape[1:3] == (2, 2))

    if not all_symmetric(weights):
        raise ValueError("All weights must be symmetric")
