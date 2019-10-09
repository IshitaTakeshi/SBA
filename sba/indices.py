
# x_ij is a keypoint corresponding to the i-th 3D point
# observed from the j-th camera (viewpoint).
# Assume we have a keypoint array X = {x_ij} observed in condition
# n_points = 4, n_viewpoints = 3,
# where x_22 x_31 x_43 are missing.
#
# X            [x_11 x_12 x_13 x_21 x_23 x_32 x_33 x_41 x_42]
# indices          1    2    3    4    5    6    7    8    9
# point index      1    1    1    2    2    3    3    4    4
# viewpoint index  1    2    3    1    3    2    3    1    2

# viewpoints_by_point[1] == [1, 2, 3]  # x_11 x_12 x_13
# viewpoints_by_point[2] == [4, 5]     # x_21 x_23
# viewpoints_by_point[3] == [6, 7]     # x_32 x_33
# viewpoints_by_point[4] == [8, 9]     # x_41 x_42

# points_by_viewpoint[1] == [1, 4, 8]  # x_11 x_21 x_41
# points_by_viewpoint[2] == [2, 6, 9]  # x_12 x_32 x_42
# points_by_viewpoint[3] == [3, 5, 7]  # x_13 x_23 x_33
#
# mask = [
#     1 1 1;  # x_11 x_12 x_13
#     1 0 1;  # x_21      x_23
#     0 1 1;  #      x_32 x_33
#     1 1 0;  # x_41 x_42
# ]


def n_points(indices):
    return len(indices.viewpoints_by_point)


def n_viewpoints(indices):
    return len(indices.points_by_viewpoint)


class Indices(object):
    def __init__(self, viewpoint_indices, point_indices):
        assert(len(viewpoint_indices) == len(point_indices))

        n_points = np.max(point_indices)
        n_viewpoints = np.max(viewpoint_indices)
        self.mask = np.empty((n_points, n_viewpoints), dtype=np.bool)

        self.viewpoints_by_point = [[]] * n_points
        self.points_by_viewpoint = [[]] * n_viewpoints

        unique_points = set()
        unique_viewpoints = set()

        for index, (i, j) in enumerate(zip(point_indices, viewpoint_indices))
            self.viewpoints_by_point[i].append(index)
            self.points_by_viewpoint[j].append(index)
            self.mask[i, j] = 1

            unique_points.add(i)
            unique_viewpoints.add(j)

        # they cannot be true if some point / viewpoint indices are missing
        # ex. raises AssertionError if n_viewpoints == 4 and
        # unique_viewpoints == [1, 2, 4]  (3 is missing)
        assert(len(unique_viewpoints) == n_viewpoints)
        assert(len(unique_points) == n_points)


def points_by_viewpoint(indices, j):
    """
    'points_by_viewpoint(j)' should return indices of 3D points
    observable from a viewpoint j
    """

    return indices.points_by_viewpoint[j]


def viewpoints_by_point(indices, i):
    """
    'viewpoints_by_point(i)' should return indices of viewpoints
    that can observe a point i
    """

    return indices.viewpoints_by_point[i]


def shared_point_indices(indices, j, k):
    """
    j, k: viewpoint indices
    This function returns two indices of points commonly observed from both viewpoints.
    These two indices are corresponding to the first and second view respectively
    """
    mask_j = indices.mask[:, j]
    mask_k = indices.mask[:, k]

    indices_j = []
    indices_k = []

    index_j = 0
    index_k = 0
    for bit_j, bit_k in zip(mask_j, mask_k)
        if bit_j == 1
            index_j += 1

        if bit_k == 1
            index_k += 1

        if bit_j & bit_k == 1
            indices_j.append(index_j)
            indices_k.append(index_k)

    if len(indices_j) == 0  # (== len(indices_k))
        return None  # no shared points found between j and k

    (indices.points_by_viewpoint[j][indices_j],
     indices.points_by_viewpoint[k][indices_k])
