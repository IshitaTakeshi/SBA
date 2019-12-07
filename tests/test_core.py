import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from sparseba.indices import Indices
from sparseba.core import (identities2x2, calc_delta_a, calc_delta_b, calc_e,
                      calc_epsilon, calc_epsilon_a, calc_epsilon_b,
                      calc_U, calc_V_inv, calc_W, calc_Y, calc_S)


n_pose_params = 4
n_point_params = 3
n_visible = 4  # number of visible points
n_points = 2
n_viewpoints = 3

mu = 0.8

weights = np.array([
    [[2, 3],
     [3, 2]],
    [[1, 0],
     [0, 1]],
    [[-3, 4],
     [4, -3]],
    [[-1, 2],
     [2, -1]]
])

A = np.array([
    [[1, 3, 0, -1],
     [2, -1, -1, 4]],
    [[3, 0, 4, 1],
     [2, 1, -1, -3]],
    [[-1, 4, 2, 0],
     [3, 1, 3, 3]],
    [[1, 0, 3, 2],
     [0, 2, 1, 1]]
])

ATWA = np.empty((4, 4, 4))
for index in range(A.shape[0]):
    ATWA[index] = np.dot(np.dot(A[index].T, weights[index]), A[index])


B = np.array([
    [[2, 3, 1],
     [1, 0, 2]],
    [[1, 0, 2],
     [1, 1, 3]],
    [[1, 2, 3],
     [1, 0, 1]],
    [[0, 0, 5],
     [4, 0, 3]]
])

BTWB = np.empty((4, 3, 3))
for index in range(B.shape[0]):
    BTWB[index] = np.dot(np.dot(B[index].T, weights[index]), B[index])

epsilon = np.array([
    [1, 3],
    [2, -1],
    [-3, 3],
    [1, 0]
])


# 2 points, 3 viewpoints
# X    = [x_11 x_12 x_22 x_23]
# mask = [   1    1    0;
#            0    1    1]

indices = Indices(viewpoint_indices=[0, 1, 1, 2],
                  point_indices=[0, 0, 1, 1])


def test_calc_delta():
    U = calc_U(indices, A, weights, mu)
    D = mu * np.identity(n_pose_params)
    assert(U.shape == (n_viewpoints, n_pose_params, n_pose_params))
    assert_array_equal(U[0], ATWA[0] + D)
    assert_array_equal(U[1], ATWA[1] + ATWA[2] + D)
    assert_array_equal(U[2], ATWA[3] + D)

    V_inv = calc_V_inv(indices, B, weights, mu)
    D = mu * np.identity(n_point_params)
    assert(V_inv.shape == (n_points, n_point_params, n_point_params))

    assert_array_almost_equal(V_inv[0], np.linalg.inv(BTWB[0] + BTWB[1] + D))
    assert_array_almost_equal(V_inv[1], np.linalg.inv(BTWB[2] + BTWB[3] + D))

    W = calc_W(indices, A, B, weights)
    assert(W.shape == (n_visible, n_pose_params, n_point_params))
    assert_array_equal(W[0], np.dot(np.dot(A[0].T, weights[0]), B[0]))
    assert_array_equal(W[1], np.dot(np.dot(A[1].T, weights[1]), B[1]))
    assert_array_equal(W[2], np.dot(np.dot(A[2].T, weights[2]), B[2]))
    assert_array_equal(W[3], np.dot(np.dot(A[3].T, weights[3]), B[3]))

    epsilon_a = calc_epsilon_a(indices, A, epsilon, weights)
    assert(epsilon_a.shape == (n_viewpoints, n_pose_params))
    assert_array_equal(epsilon_a[0],
                       np.dot(np.dot(A[0].T, weights[0]), epsilon[0]))
    assert_array_equal(epsilon_a[1],
                       np.dot(np.dot(A[1].T, weights[1]), epsilon[1]) +
                       np.dot(np.dot(A[2].T, weights[2]), epsilon[2]))
    assert_array_equal(epsilon_a[2],
                       np.dot(np.dot(A[3].T, weights[3]), epsilon[3]))

    epsilon_b = calc_epsilon_b(indices, B, epsilon, weights)
    assert(epsilon_b.shape == (n_points, n_point_params))
    assert_array_equal(epsilon_b[0],
                       np.dot(np.dot(B[0].T, weights[0]), epsilon[0]) +
                       np.dot(np.dot(B[1].T, weights[1]), epsilon[1]))
    assert_array_equal(epsilon_b[1],
                       np.dot(np.dot(B[2].T, weights[2]), epsilon[2]) +
                       np.dot(np.dot(B[3].T, weights[3]), epsilon[3]))

    Y = calc_Y(indices, W, V_inv)
    assert(Y.shape == (n_visible, n_pose_params, n_point_params))
    assert_array_almost_equal(Y[0], np.dot(W[0], V_inv[0]))  # (i, j) = (0, 0)
    assert_array_almost_equal(Y[1], np.dot(W[1], V_inv[0]))  # (i, j) = (0, 1)
    assert_array_almost_equal(Y[2], np.dot(W[2], V_inv[1]))  # (i, j) = (1, 1)
    assert_array_almost_equal(Y[3], np.dot(W[3], V_inv[1]))  # (i, j) = (1, 2)

    e = calc_e(indices, Y, epsilon_a, epsilon_b)
    assert(e.shape == (n_viewpoints, n_pose_params))
    assert_array_equal(e[0], -np.dot(Y[0], epsilon_b[0]) + epsilon_a[0])
    assert_array_equal(e[1], (-np.dot(Y[1], epsilon_b[0])
                              -np.dot(Y[2], epsilon_b[1]) + epsilon_a[1]))
    assert_array_equal(e[2], -np.dot(Y[3], epsilon_b[1]) + epsilon_a[2])

    S = calc_S(indices, U, Y, W)
    #
    assert(S.shape == (n_pose_params * n_viewpoints,
                       n_pose_params * n_viewpoints))
    # (j, k) == (0, 0)
    assert_array_almost_equal(S[0:4, 0:4], U[0] - np.dot(Y[0], W[0].T))
    # (j, k) == (0, 1)
    assert_array_almost_equal(S[0:4, 4:8], -np.dot(Y[0], W[1].T))
    # (j, k) == (0, 2)
    assert_array_almost_equal(S[0:4, 8:12], np.zeros((4, 4)))
    # (j, k) == (1, 0)
    assert_array_almost_equal(S[4:8, 0:4], -np.dot(Y[1], W[0].T))
    # (j, k) == (1, 1)
    assert_array_almost_equal(S[4:8, 4:8], U[1] - np.dot(Y[1], W[1].T) - np.dot(Y[2], W[2].T))
    # (j, k) == (1, 2)
    assert_array_almost_equal(S[4:8, 8:12], -np.dot(Y[2], W[3].T))
    # (j, k) == (2, 0)
    assert_array_almost_equal(S[8:12, 0:4], np.zeros((4, 4)))
    # (j, k) == (2, 1)
    assert_array_almost_equal(S[8:12, 4:8], -np.dot(Y[3], W[2].T))
    # (j, k) == (2, 2)
    assert_array_almost_equal(S[8:12, 8:12], -np.dot(Y[3], W[3].T) + U[2])

    delta_a = calc_delta_a(S, e)
    assert(delta_a.shape == (n_viewpoints, n_pose_params))
    assert_array_almost_equal(np.dot(S, delta_a.flatten()), e.flatten())

    delta_b = calc_delta_b(indices, V_inv, W, epsilon_b, delta_a)

    assert(delta_b.shape == (n_points, n_point_params))
    assert_array_almost_equal(
        delta_b[0],
        np.dot(V_inv[0],
               epsilon_b[0] - (np.dot(W[0].T, delta_a[0]) + np.dot(W[1].T, delta_a[1])))
    )

    assert_array_almost_equal(
        delta_b[1],
        np.dot(V_inv[1],
               epsilon_b[1] - (np.dot(W[2].T, delta_a[1]) + np.dot(W[3].T, delta_a[2])))
    )


def test_calc_epsilon():
    x_true = np.array([
        [1, 3],
        [2, 1],
        [3, 2]
    ])
    x_pred = np.array([
        [4, 5],
        [3, 0],
        [2, 1]
    ])
    expected = np.array([
        [-3, -2],
        [-1, 1],
        [1, 1]
    ])

    assert_array_equal(calc_epsilon(x_true, x_pred), expected)


def test_identities2x2():
    assert_array_equal(
        identities2x2(3),
        np.array([
            [[1, 0],
             [0, 1]],
            [[1, 0],
             [0, 1]],
            [[1, 0],
             [0, 1]]
        ])
    )
