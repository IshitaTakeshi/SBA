import pytest
import numpy as np

from sba.indices import Indices
from sba.core import sba


def test_check_args():
    def case1():
        # number of visible keypoints = 13
        # mask = [1 1 1 1;
        #         1 0 1 1;
        #         1 0 1 1;
        #         1 1 1 0]
        viewpoint_indices = [0, 1, 2, 3, 0, 2, 3, 0, 2, 3, 0, 1, 2]
        point_indices = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
        indices = Indices(viewpoint_indices, point_indices)

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
        with pytest.raises(ValueError):
            sba(indices, x_true, x_pred, A, B)

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
        sba(indices, x_true, x_pred, A, B)

    case1()
    case2()
