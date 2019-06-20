import torch
import numpy as np

from pytorch_toolbox.losses import soft_f_score_loss

# def test_1d_input():
#
#     pred = torch.from_numpy(
#         np.array([
#             [-2, 3, 1, 3],
#             [3, 1, 2, -3]
#         ])
#     )
#     target = torch.from_numpy(
#         np.array([
#             [0, 1, 0, 0],
#             [0, 0, 0, 1]
#         ])
#     )
#
#     loss = soft_f_score_loss(pred, target)
#     print(loss)
#
#
# test_1d_input()