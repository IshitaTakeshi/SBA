import pytest
from numpy.testing import assert_array_equal

from sparseba.indices import Indices, indices_are_unique


def test_indices_are_unique():
    # unique
    Indices([0, 1, 2, 0, 2, 1, 2, 0, 1],
            [0, 0, 0, 1, 1, 2, 2, 3, 3])

    with pytest.raises(ValueError):
        # second and third are duplicated
        #        0  1  2  3  4  5  6  7  8
        Indices([0, 1, 2, 2, 2, 1, 2, 0, 1],
                [0, 0, 0, 1, 1, 2, 2, 3, 3])


def test_case_1():
    # row indicates keypoints generated from the same point
    # column indicates keypoints observed in the same viewpoint
    # visibility mask = [
    #     mask          keypoints         indices
    #     [1, 1, 1],  # x_00 x_01 x_02  # 0   1   2
    #     [1, 0, 1],  # x_10      x_12  # 3       4
    #     [0, 1, 1],  #      x_21 x_22  #     5   6
    #     [1, 1, 0]   # x_30 x_31       # 7   8
    # ]

    indices = Indices(viewpoint_indices=[0, 1, 2, 0, 2, 1, 2, 0, 1],
                      point_indices=[0, 0, 0, 1, 1, 2, 2, 3, 3])

    assert(indices.n_visible == 9)
    assert(indices.n_points == 4)
    assert(indices.n_viewpoints == 3)

    # get array indices of X for x_*0
    # where x_*0 are projections of all visible points in the 0th viewpoint
    assert_array_equal(indices.points_by_viewpoint(0), [0, 3, 7])

    # get array indices of X for x_*2
    assert_array_equal(indices.points_by_viewpoint(2), [2, 4, 6])

    # get array indices of X for x_0*
    # where x_0* are projections of the 0st 2D point in all observable viewpoints
    assert_array_equal(indices.viewpoints_by_point(0), [0, 1, 2])

    # get array indices of X for x_3*
    assert_array_equal(indices.viewpoints_by_point(3), [7, 8])

    # [x_01, x_21] and [x_02, x_22] are shared
    indices_j, indices_k = indices.shared_point_indices(1, 2)
    assert_array_equal(indices_j, [1, 5])
    assert_array_equal(indices_k, [2, 6])

    # [x_00, x_10] and [x_02, x_12] are shared
    indices_j, indices_k = indices.shared_point_indices(0, 2)
    assert_array_equal(indices_j, [0, 3])
    assert_array_equal(indices_k, [2, 4])

    # [x_00, x_30] and [x_01, x_31] are shared
    indices_j, indices_k = indices.shared_point_indices(0, 1)
    assert_array_equal(indices_j, [0, 7])
    assert_array_equal(indices_k, [1, 8])

    indices_j, indices_k = indices.shared_point_indices(1, 1)
    assert_array_equal(indices_j, [1, 5, 8])
    assert_array_equal(indices_k, [1, 5, 8])


def test_case_2():
    # n_points = 2, n_viewpoints = 3,
    # indices     0    1    2
    #   X =   [x_02 x_10 x_11]

    # then the corresponding mask should be
    # mask = [
    #     [0 0 1],  #           x_02  #       0
    #     [1 1 0]   # x_10 x_11       # 1  2
    # ]

    indices = Indices(viewpoint_indices=[2, 0, 1], point_indices=[0, 1, 1])

    assert(indices.n_visible == 3)
    assert(indices.n_points == 2)
    assert(indices.n_viewpoints == 3)

    # get array indices of X for x_*0
    assert_array_equal(indices.points_by_viewpoint(0), [1])

    # get array indices of X for x_*2
    assert_array_equal(indices.points_by_viewpoint(2), [0])

    # get array indices of X for x_0*
    assert_array_equal(indices.viewpoints_by_point(0), [0])

    # get array indices of X for x_1*
    assert_array_equal(indices.viewpoints_by_point(1), [1, 2])

    # [x_10] and [x_11] are shared
    indices_j, indices_k = indices.shared_point_indices(0, 1)
    assert_array_equal(indices_j, [1])
    assert_array_equal(indices_k, [2])

    # no points are shared
    indices_j, indices_k = indices.shared_point_indices(0, 2)
    assert_array_equal(indices_j, [])
    assert_array_equal(indices_k, [])

    # second row has only zero elements
    # visibility mask = [
    #     1 0 1 0;
    #     0 0 0 0;
    #     0 1 1 1
    # ]
    viewpoint_indices = [0, 2, 1, 2, 3]
    point_indices = [0, 0, 2, 2, 2]
    with pytest.raises(AssertionError):
        Indices(viewpoint_indices, point_indices)

    # third column has only zero elements
    # visibility mask = [
    #     1 0 0 1;
    #     0 1 0 0;
    #     0 1 0 1
    # ]
    viewpoint_indices = [0, 3, 1, 1, 3]
    point_indices = [0, 0, 1, 2, 2]
    with pytest.raises(AssertionError):
        Indices(viewpoint_indices, point_indices)
