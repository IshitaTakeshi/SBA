import numpy as np


# x_ij is a keypoint corresponding to the i-th 3D point
# observed from the j-th camera (viewpoint).
# Assume we have a keypoint array X = {x_ij} observed in condition
# n_points = 4, n_viewpoints = 3,
# where x_11 x_20 x_32 are missing.
#
# X            [x_00 x_01 x_02 x_10 x_12 x_21 x_22 x_30 x_31]
# indices          0    1    2    3    4    5    6    7    8
# point index      0    0    0    1    1    2    2    3    3
# viewpoint index  0    1    2    0    2    1    2    0    1

# viewpoints_by_point[0] == [0, 1, 2]  # x_00 x_01 x_02
# viewpoints_by_point[1] == [3, 4]     # x_10      x_12
# viewpoints_by_point[2] == [5, 6]     #      x_21 x_22
# viewpoints_by_point[3] == [7, 8]     # x_30 x_31

# points_by_viewpoint[0] == [0, 3, 7]  # x_00 x_10      x_30
# points_by_viewpoint[1] == [1, 5, 8]  # x_01      x_21 x_31
# points_by_viewpoint[2] == [2, 4, 6]  # x_02 x_12 x_22
#
# mask = [
#     1 1 1;  # x_00 x_01 x_02
#     1 0 1;  # x_10      x_12
#     0 1 1;  #      x_21 x_22
#     1 1 0;  # x_30 x_31
# ]


def indices_are_unique(viewpoint_indices, point_indices):
    indices = np.vstack((point_indices, viewpoint_indices))
    unique = np.unique(indices, axis=1)
    return unique.shape[1] == len(point_indices)


class Indices(object):
    def __init__(self, viewpoint_indices, point_indices):
        assert(len(viewpoint_indices) == len(point_indices))
        if not indices_are_unique(viewpoint_indices, point_indices):
            raise ValueError("Found non-unique (i, j) pair")

        self.n_visible = len(viewpoint_indices)

        n_viewpoints = np.max(viewpoint_indices) + 1
        n_points = np.max(point_indices) + 1

        self.mask = np.zeros((n_points, n_viewpoints), dtype=np.bool)

        self._viewpoints_by_point = [[] for i in range(n_points)]
        self._points_by_viewpoint = [[] for j in range(n_viewpoints)]

        unique_viewpoints = set()
        unique_points = set()

        for index, (i, j) in enumerate(zip(point_indices, viewpoint_indices)):
            self._viewpoints_by_point[i].append(index)
            self._points_by_viewpoint[j].append(index)
            self.mask[i, j] = 1

            unique_viewpoints.add(j)
            unique_points.add(i)

        # unique_points are accumulated over all viewpoints.
        # The condition below cannot be true if some point indices
        # are missing ex. raises AssertionError if n_points == 4 and
        # unique_points == {0, 1, 3}  (2 is missing)
        assert(len(unique_points) == n_points)
        # do the same to 'unique_viewpoints'
        assert(len(unique_viewpoints) == n_viewpoints)

        for i, viewpoints in enumerate(self._viewpoints_by_point):
            self._viewpoints_by_point[i] = np.array(viewpoints)

        for j, points in enumerate(self._points_by_viewpoint):
            self._points_by_viewpoint[j] = np.array(points)

    @property
    def n_points(self):
        return len(self._viewpoints_by_point)

    @property
    def n_viewpoints(self):
        return len(self._points_by_viewpoint)

    def shared_point_indices(self, j, k):
        """
        j, k: viewpoint indices
        Returns two point indices commonly observed from both viewpoints.
        These two indices represent the first and second view respectively.
        """

        # points_j = [1, 5, 8]
        # points_j = [2, 4, 6]
        # mask_j       = [1, 0, 1, 1]
        # mask_k       = [1, 1, 1, 0]
        # mask         = [1, 0, 1, 0]
        # mask[mask_j] = [1, 1, 0]
        # mask[mask_k] = [1, 0, 1]
        # points_j[mask[mask_j]] = [1, 5]
        # points_k[mask[mask_k]] = [2, 6]

        points_j = self._points_by_viewpoint[j]
        points_k = self._points_by_viewpoint[k]
        mask_j, mask_k = self.mask[:, j], self.mask[:, k]
        mask = mask_j & mask_k
        return (points_j[mask[mask_j]], points_k[mask[mask_k]])

    def points_by_viewpoint(self, j):
        """
        'points_by_viewpoint(j)' should return indices of 3D points
        observable from a viewpoint j
        """

        return self._points_by_viewpoint[j]

    def viewpoints_by_point(self, i):
        """
        'viewpoints_by_point(i)' should return indices of viewpoints
        that can observe a point i
        """

        return self._viewpoints_by_point[i]
