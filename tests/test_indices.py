import pytest

from sba.indices import Indices

# visibility mask = [
#     1 1 1;  # x_11 x_12 x_13
#     1 0 1;  # x_21      x_23
#     0 1 1;  #      x_32 x_33
#     1 1 0;  # x_41 x_42
# ]

viewpoint_indices = [1, 2, 3, 1, 3, 2, 3, 1, 2]
point_indices = [1, 1, 1, 2, 2, 3, 3, 4, 4]
indices = Indices(viewpoint_indices, point_indices)

assert(n_points(indices) == 4)
assert(n_viewpoints(indices) == 3)

# get array indices of X for x_*1
# where x_*1 are projections of all visible points in the 1st viewpoint
assert_array_equal(points_by_viewpoint(indices, 1), [1, 4, 8])

# get array indices of X for x_*3
assert_array_equal(points_by_viewpoint(indices, 3), [3, 5, 7])

# get array indices of X for x_1*
# where x_1* are projections of the 1st 3D point in all observable viewpoints
assert_array_equal(viewpoints_by_point(indices, 1), [1, 2, 3])

# get array indices of X for x_4*
assert_array_equal(viewpoints_by_point(indices, 4), [8, 9])

# [x_12, x_32] and [x_13, x_33] are shared
viewpoints_j, viewpoints_k = shared_point_indices(indices, 2, 3)
assert_array_equal(viewpoints_j, [2, 6])
assert_array_equal(viewpoints_k, [3, 7])

# [x_11, x_21] and [x_13, x_23] are shared
viewpoints_j, viewpoints_k = shared_point_indices(indices, 1, 3)
assert_array_equal(viewpoints_j, [1, 4])
assert_array_equal(viewpoints_k, [3, 5])


# another case
# n_points = 2, n_viewpoints = 3,
# indices     1    2    3
#   X =   [x_13 x_21 x_22]

# then the corresponding mask should be
# mask = [
#     [0 0 1],  #           x_13
#     [1 1 0]   # x_21 x_22
# ]

viewpoint_indices = [3, 1, 2]
point_indices = [1, 2, 2]
indices = Indices(viewpoint_indices, point_indices)

assert(n_points(indices) == 2)
assert(n_viewpoints(indices) == 3)

# get array indices of X for x_*1
assert_array_equal(points_by_viewpoint(indices, 1), [2])

# get array indices of X for x_*3
assert_array_equal(points_by_viewpoint(indices, 3), [1])

# get array indices of X for x_1*
assert_array_equal(viewpoints_by_point(indices, 1), [1])

# get array indices of X for x_2*
assert_array_equal(viewpoints_by_point(indices, 2), [2, 3])

# [x_21] and [x_22] are shared
viewpoints_j, viewpoints_k = shared_point_indices(indices, 1, 2)
assert_array_equal(viewpoints_j, [2])
assert_array_equal(viewpoints_k, [3])

# no points are shared
assert(shared_point_indices(indices, 1, 3) is None)


# second row has only zero elements
# visibility mask = [
#     1 0 1 0;
#     0 0 0 0;
#     0 1 1 1
# ]
viewpoint_indices = [1, 3, 2, 3, 4]
point_indices = [1, 1, 3, 3, 3]
with pytest.raises(AssertionError):
    Indices(viewpoint_indices, point_indices)

# third column has only zero elements
# visibility mask = [
#     1 0 0 1;
#     0 1 0 0;
#     0 1 0 1
# ]
viewpoint_indices = [1, 4, 2, 2, 4]
point_indices = [1, 1, 2, 3, 3]
with pytest.raises(AssertionError):
    Indices(viewpoint_indices, point_indices)
