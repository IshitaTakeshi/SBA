import pytest

import numpy as np
from numpy.testing import assert_array_equal

from sba.indices import Indices
from sba.utils import identities2x2
from sba.core import SBA


def test_check_args():
    def case1():
        # number of visible keypoints = 13
        # mask = [1 1 1 1;
        #         1 0 1 1;
        #         1 0 1 1;
        #         1 1 1 0]
        viewpoint_indices = [0, 1, 2, 3, 0, 2, 3, 0, 2, 3, 0, 1, 2]
        point_indices = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]

        n_visible = len(viewpoint_indices)
        A = np.random.random(size=(n_visible, 2, 4))  # n_pose_params = 4
        B = np.random.random(size=(n_visible, 2, 3))  # n_point_params = 3
        x_true = x_pred = np.random.random(size=(n_visible, 2))

        # n_rows(J) = 2 * n_visible_keypoints
        #           = 2 * 16 = 26
        # n_cols(J) = n _point_params * n_points + n_pose_params * n_viewpoints
        #           = 3 * 4 + 4 * 4 = 12 + 16 = 28
        # n_rows(J) < n_cols(J)
        # J' * J can not be invertible. ValueError should be thrown
        sba = SBA(viewpoint_indices, point_indices)
        with pytest.raises(ValueError):
            sba.compute(x_true, x_pred, A, B)

    def case2():
        # number of visible keypoints = 14
        # mask = [1 1 1 1;
        #         1 1 1 1;
        #         1 1 1 0;
        #         1 1 1 0]
        viewpoint_indices = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 0, 1, 2]
        point_indices = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3]
        indices = Indices(viewpoint_indices, point_indices)

        n_visible = len(viewpoint_indices)
        A = np.random.random(size=(n_visible, 2, 4))  # n_pose_params = 4
        B = np.random.random(size=(n_visible, 2, 3))  # n_point_params = 3
        x_true = x_pred = np.random.random(size=(n_visible, 2))

        # n_rows(J) = 2 * n_visible_keypoints
        #           = 2 * 14 = 28
        # n_cols(J) = n _point_params * n_points + n_pose_params * n_viewpoints
        #           = 3 * 4 + 4 * 4 = 12 + 16 = 28
        # n_rows(J) = n_cols(J)
        # J' * J can be invertible. Nothing should be thrown
        sba = SBA(viewpoint_indices, point_indices)
        sba.compute(x_true, x_pred, A, B)

        # weights are non symmetric
        weights = np.arange(n_visible * 2 * 2).reshape(n_visible, 2, 2)
        with pytest.raises(ValueError):
            sba.compute(x_true, x_pred, A, B, weights)

        # make them symmetric
        weights = np.array([np.dot(w.T, w) for w in weights])
        # nothing should be raised
        sba.compute(x_true, x_pred, A, B, weights)

    case1()
    case2()


def test_all_symmmetric():
    # identity arrays created
    assert_array_equal(
        identities2x2(4),
        np.array([
            [[1, 0],
             [0, 1]],
            [[1, 0],
             [0, 1]],
            [[1, 0],
             [0, 1]],
            [[1, 0],
             [0, 1]]
        ])
    )
