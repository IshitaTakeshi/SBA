import itertools


def create_jacobian(mask::BitArray, A::Array, B::Array):
    assert(A.shape[3] == B.shape[3])

    N = np.sum(mask)
    n_points, n_viewpoints = mask.shape
    n_pose_params = A.shape[2]
    n_point_params = B.shape[2]

    n_rows = 2 * N
    n_cols_a = n_pose_params * n_viewpoints
    n_cols_b = n_point_params * n_points
    JA = np.zeros((n_rows, n_cols_a))
    JB = np.zeros((n_rows, n_cols_b))

    # J' * J should be invertible
    # n_rows(J) >= n_cols(J)
    assert(n_rows >= n_cols_a + n_cols_b)

    viewpoint_indices = np.empty(N, dtype=np.int64)
    point_indices = np.empty(N, dtype=np.int64)

    index = 1
    for i, j in itertools.product(range(n_points), range(n_viewpoints)):
        if not mask[i, j]:
            continue

        viewpoint_indices[index] = j
        point_indices[index] = i

        row = (index - 1) * 2 + 1

        col = (j-1) * n_pose_params + 1
        JA[row:row+1, col:col+n_pose_params-1] = A[:, :, index]

        col = (i-1) * n_point_params + 1
        JB[row:row+1, col:col+n_point_params-1] = B[:, :, index]

        index += 1

    indices = Indices(viewpoint_indices, point_indices)
    J = np.hstack((JA, JB))
    return indices, J


# there shouldn't be an empty row / column
# (empty means that all row elements / column elements = 0)
# and it seems that at least two '1' elements must be
# found per one row / column
mask = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 1, 0, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 0, 1, 0],
    [1, 0, 0, 1, 1, 1, 0, 1, 1],
    [0, 0, 1, 0, 0, 0, 0, 0, 1]
], dtype=np.bool)

N = np.sum(mask)

x_true = np.random.uniform(-9, 9, (2, N))
x_pred = np.random.uniform(-9, 9, (2, N))
A = np.random.random((2, 4, N))
B = np.random.random((2, 3, N))

indices, J = create_jacobian(mask, A, B)
delta_a, delta_b = sba(indices, x_true, x_pred, A, B)

delta = np.linalg.solve(np.dot(J.T, J), np.dot(J.T, (x_true - x_pred).flatten()))

n_pose_params = A.shape[2]
n_viewpoints = mask.shape[2]
size_A = n_pose_params * n_viewpoints

assert_array_almost_equal(delta[1:size_A], vec(delta_a))
assert_array_almost_equal(delta[size_A+1:end], vec(delta_b))

