from sba.indices import Indices


# number of visible keypoints = 13
# mask = [1 1 1 1;
#         1 0 1 1;
#         1 0 1 1;
#         1 1 1 0]

viewpoint_indices = [1, 2, 3, 4, 1, 3, 4, 1, 3, 4, 1, 2, 3]
point_indices     = [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
indices = Indices(viewpoint_indices, point_indices)
N = len(viewpoint_indices)
A = np.random.normal((2, 4, N))  # n_pose_params = 4
B = np.random.normal((2, 3, N))  # n_point_params = 3
x_true = x_pred = randn(Float64, 2, N)

# n_rows(J) = 2 * n_visible_keypoints
#           = 2 * 16 = 26
# n_cols(J) = n _point_params * n_points + n_pose_params * n_viewpoints
#           = 3 * 4 + 4 * 4 = 12 + 16 = 28
# n_rows(J) < n_cols(J)
# J' * J can not be invertible. ArgumentError should be thrown
with pytest.raises(ValueError):
    sba(indices, x_true, x_pred, A, B)

# number of visible keypoints = 14
# mask = [1 1 1 1;
#         1 1 1 1;
#         1 1 1 0;
#         1 1 1 0]
viewpoint_indices = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3]
point_indices     = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4]
indices = Indices(viewpoint_indices, point_indices)
N = len(viewpoint_indices)
A = np.random.normal((2, 4, N))  # n_pose_params = 4
B = np.random.normal((2, 3, N))  # n_point_params = 3
x_true = x_pred = np.random.normal((2, N))

# n_rows(J) = 2 * n_visible_keypoints
#           = 2 * 14 = 28
# n_cols(J) = n _point_params * n_points + n_pose_params * n_viewpoints
#           = 3 * 4 + 4 * 4 = 12 + 16 = 28
# n_rows(J) = n_cols(J)
# J' * J can be invertible. Nothing should be thrown
sba(indices, x_true, x_pred, A, B)
