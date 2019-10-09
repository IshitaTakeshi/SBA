def test_calc_epsilon():
    expected = np.array([[-3 -1 1],
                         [-2 1 1]])
    assert_equal(calc_epsilon([[1 2 3], [3 1 2]], [[4 3 2, 5 0 1]]), expected)


A = np.empty((2, 4, 4))

A[:, :, 1] = [ 1  3  0 -1;
               2 -1 -1  4]
A[:, :, 2] = [ 3  0  4  1;
               2  1 -1 -3]
A[:, :, 3] = [-1  4  2  0;
               3  1  3  3]
A[:, :, 4] = [ 1  0  3  2;
               0  2  1  1]

AtA = np.empty(4, 4, 4)
for index in 1:size(A, 3)
    AtA[:, :, index] = np.dot(A[:, :, index].T, A[:, :, index])

B = Array{Float64}(undef, 2, 3, 4)

B[:, :, 1] = [ 2  3  1;
               1  0  2]
B[:, :, 2] = [ 1  0  2;
               1  1  3]
B[:, :, 3] = [ 1  2  3;
               1  0  1]
B[:, :, 4] = [ 0  0  5;
               4  0  3]

BtB = np.empty((3, 3, 4))
for index in 1:size(B, 3)
    BtB[:, :, index] = np.dot(B[:, :, index].T, B[:, :, index])

# 2 points, 3 viewpoints
# i = 1:2, j = 1:3
# N = 4 (number of visible points)
# X    = [x_11 x_12 x_22 x_23]
# mask = [   1    1    0;
#            0    1    1]

viewpoint_indices = [1, 2, 2, 3]
point_indices = [1, 1, 2, 2]

indices = Indices(viewpoint_indices, point_indices)


def test_calc_delta():
    U = calc_U(indices, A)
    assert(U.shape == (4, 4, 3))  # (n_pose_paramas, n_pose_paramas, n_viewpoints)
    assert_array_equal(U[:, :, 1], AtA[:, :, 1])
    assert_array_equal(U[:, :, 2], AtA[:, :, 2] + AtA[:, :, 3])
    assert_array_equal(U[:, :, 3], AtA[:, :, 4])

    V_inv = calc_V_inv(indices, B)
    assert(V_inv.shape == (3, 3, 2))  # (n_point_params, n_point_params, n_points)
    assert_array_equal(V_inv[:, :, 1], inv(BtB[:, :, 1] + BtB[:, :, 2]))
    assert_array_equal(V_inv[:, :, 2], inv(BtB[:, :, 3] + BtB[:, :, 4]))

    W = calc_W(indices, A, B)
    assert(W.shape == (4, 3, 4))  # (n_pose_params, n_point_params, N)
    assert_array_equal(W[:, :, 1], np.dot(A[:, :, 1].T, B[:, :, 1]))
    assert_array_equal(W[:, :, 2], np.dot(A[:, :, 2].T, B[:, :, 2]))
    assert_array_equal(W[:, :, 3], np.dot(A[:, :, 3].T, B[:, :, 3]))
    assert_array_equal(W[:, :, 4], np.dot(A[:, :, 4].T, B[:, :, 4]))

    epsilon = np.array([[1, 2, -3, 1],
                        [3, -1, 3, 0]])

    epsilon_a = calc_epsilon_a(indices, A, epsilon)
    assert(epsilon_a.shape == (4, 3))  # (n_pose_params, n_viewpoints)
    assert_array_equal(epsilon_a[:, 1], np.dot(A[:, :, 1].T, epsilon[:, 1]))
    assert_array_equal(epsilon_a[:, 2], (np.dot(A[:, :, 2].T, epsilon[:, 2]) +
                                         np.dot(A[:, :, 3].T, epsilon[:, 3])))
    assert_array_equal(epsilon_a[:, 3], np.dot(A[:, :, 4].T, epsilon[:, 4]))


    epsilon_b = calc_epsilon_b(indices, B, epsilon)
    assert(epsilon_b.shape == (3, 2))  # (n_point_params, n_points)
    assert_array_equal(epsilon_b[:, 1], (np.dot(B[:, :, 1].T, epsilon[:, 1]) +
                                         np.dot(B[:, :, 2].T, epsilon[:, 2])))
    assert_array_equal(epsilon_b[:, 2], (np.dot(B[:, :, 3].T, epsilon[:, 3]) +
                                         np.dot(B[:, :, 4].T, epsilon[:, 4])))

    Y = calc_Y(indices, W, V_inv)
    assert(Y.shape == (4, 3, 4))  # (n_pose_params, n_point_params, N)
    assert_array_equal(Y[:, :, 1], np.dot(W[:, :, 1], V_inv[:, :, 1]))  # (i, j) = (1, 1)
    assert_array_equal(Y[:, :, 2], np.dot(W[:, :, 2], V_inv[:, :, 1]))  # (i, j) = (1, 2)
    assert_array_equal(Y[:, :, 3], np.dot(W[:, :, 3], V_inv[:, :, 2]))  # (i, j) = (2, 2)
    assert_array_equal(Y[:, :, 4], np.dot(W[:, :, 4], V_inv[:, :, 2]))  # (i, j) = (2, 3)

    e = calc_e(indices, Y, epsilon_a, epsilon_b)
    assert(e.shape == (4, 3))  # (n_pose_params, n_viewpoints)
    assert_array_equal(e[:, 1], (-np.dot(Y[:, :, 1], epsilon_b[:, 1]) +
                                 epsilon_a[:, 1]))
    assert_array_equal(e[:, 2], (-np.dot(Y[:, :, 2], epsilon_b[:, 1])
                                 -np.dot(Y[:, :, 3], epsilon_b[:, 2]) +
                                 epsilon_a[:, 2]))
    assert_array_equal(e[:, 3], (-np.dot(Y[:, :, 4], epsilon_b[:, 2]) +
                                 epsilon_a[:, 3]))

    S = calc_S(indices, U, Y, W)
    # (n_pose_params * n_viewpoints, n_pose_params * n_viewpoints)
    assert(size(S) == (4 * 3, 4 * 3))
    # (j, k) == (1, 1)
    assert_array_equal(S[1: 4, 1: 4], (np.dot(-Y[:, :, 1], W[:, :, 1].T) + U[:, :, 1]))
    # (j, k) == (1, 2)
    assert_array_equal(S[1: 4, 5: 8], -np.dot(Y[:, :, 1], W[:, :, 2].T))
    # (j, k) == (1, 3)
    assert_array_equal(S[1: 4, 9:12], np.zeros((4, 4)))
    # (j, k) == (2, 1)
    assert_array_equal(S[5: 8, 1: 4], -np.dot(Y[:, :, 2], W[:, :, 1].T))
    # (j, k) == (2, 2)
    assert_array_equal(S[5: 8, 5: 8], (-np.dot(Y[:, :, 2], W[:, :, 2].T)
                                       -np.dot(Y[:, :, 3], W[:, :, 3].T) +
                                       U[:, :, 2]))
    # (j, k) == (2, 3)
    assert_array_equal(S[5: 8, 9:12], -np.dot(Y[:, :, 3], W[:, :, 4].T))

    # (j, k) == (3, 1)
    assert_array_equal(S[9:12, 1:4], np.zeros((4, 4)))

    # (j, k) == (3, 2)
    assert_array_equal(S[9:12, 5:8], -np.dot(Y[:, :, 4], W[:, :, 3].T))

    # (j, k) == (3, 3)
    assert_array_equal(S[9:12, 9:12], (-np.dot(Y[:, :, 4], W[:, :, 4].T) +
                                       U[:, :, 3]))

    delta_a = calc_delta_a(S, e)
    assert(delta_a.shape == (4, 3))  # (n_pose_params, n_viewpoints)
    assert_array_almost_equal(np.dot(S, delta_a.flatten()), e.flatten())

    delta_b = calc_delta_b(indices, V_inv, W, epsilon_b, delta_a)

    assert(delta_b.shape == (3, 2))  # (n_point_params, n_points)
    assert_array_almost_equal(
        delta_b[:, 1],
        np.dot(V_inv[:, :, 1],
               epsilon_b[:, 1] - (np.dot(W[:, :, 1].T, delta_a[:, 1]) +
                                  np.dot(W[:, :, 2].T, delta_a[:, 2])))
    )

    assert_array_almost_equal(
        delta_b[:, 2],
        np.dot(V_inv[:, :, 2],
               epsilon_b[:, 2] - (np.dot(W[:, :, 3].T, delta_a[:, 2]) +
                                  np.dot(W[:, :, 4].T, delta_a[:, 3])))
    )
