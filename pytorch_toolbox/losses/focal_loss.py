import logging
import torch
from torch.nn import functional as F
import numpy as np

from pytorch_toolbox.losses import BaseLoss
from pytorch_toolbox.defaults import default_hardware
from pytorch_toolbox.utils import timeit, to_numpy


class FocalLoss(BaseLoss):
    def __init__(self, gamma=2, one_hot_encoding=False):
        self.gamma = gamma
        self.one_hot_encoding = one_hot_encoding
        if not one_hot_encoding:
            logging.warning(
                """
                In FocalLoss:
                One hot encoding is not used for the target
                so the number of classes will be assumed to be:
                <output_of_model>.shape[1]"
                """
            )

    @property
    def unreduced_loss(self):
        return self._unreduced_loss

    @property
    def per_sample_loss(self):
        return self._per_sample_loss

    @property
    def reduced_loss(self):
        return self._reduced_loss

    def __call__(self, out, *yb):
        prediction = out
        target = yb[0]
        # This returns B x ... (same shape as input)
        self._unreduced_loss = focal_loss(prediction, target, self.gamma, one_hot_encoding=self.one_hot_encoding)
        self._per_sample_loss = self.reshape_to_batch_size_x_minus_one_and_sum_over_last_dimension(
            self._unreduced_loss)
        self._reduced_loss = self._per_sample_loss.mean()
        return self._reduced_loss


def focal_loss(input, target, gamma=2, one_hot_encoding=False):
    """

    :param input: B x N_classes x ...
    :param target: B x N_classes x ... # one hot encoding
    :param gamma: the higher the value, the greater the loss for uncertain classes
    :return:
    """
    if not one_hot_encoding:
        target = _make_one_hot(input, target)

    if not (target.size() == input.size()):
        raise ValueError(f"Target size ({target.size()}) must be the same as input size ({input.size()})")

    target = target.float()
    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + \
           ((-max_val).exp() + (-input - max_val).exp()).log()

    inv_probs = F.logsigmoid(-input * (target * 2.0 - 1.0))
    loss = (inv_probs * gamma).exp() * loss
    return loss


def _make_one_hot(input, target):
    one_hot = torch.FloatTensor(*input.shape).zero_().to(default_hardware.device)
    target = one_hot.scatter_(1, target.long(), 1)
    return target

jj