from functools import partial

import torch
from torch.nn import functional as F
from pytorch_toolbox.losses import BaseLoss


class SoftFScoreLoss(BaseLoss):
    def __init__(self, beta=1, per_sample_loss_aggregate_method="SUM"):
        self.beta = beta
        self.per_sample_loss_aggregate_method = per_sample_loss_aggregate_method

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

        # This returns B x n_classes
        loss = soft_f_score_loss(prediction, target, beta=self.beta)
        self._unreduced_loss = loss
        self._per_sample_loss = self.reshape_to_batch_size_x_minus_one_aggregate_over_last_dimension(
            self._unreduced_loss, aggregate_method=self.per_sample_loss_aggregate_method
        )
        self._reduced_loss = self._per_sample_loss.mean()
        return self._reduced_loss


class SoftF1Loss(SoftFScoreLoss):
    def __init__(self, per_sample_loss_aggregate_method):
        super().__init__(
            beta=1, per_sample_loss_aggregate_method=per_sample_loss_aggregate_method
        )


class SoftF2Loss(SoftFScoreLoss):
    def __init__(self, per_sample_loss_aggregate_method):
        super().__init__(
            beta=2, per_sample_loss_aggregate_method=per_sample_loss_aggregate_method
        )


def soft_f_score_loss(logits, labels, beta=1):
    """
    The formula to calculate F_{beta} score is:

        F_{beta} = (1 + beta^{2}) * (precision * recall) / (beta^{2} * precision + recall)

    F_{beta} measure the effectiveness of retrieval with respect to a user who attaches beta times as much importance
    to recall as precision (https://en.wikipedia.org/wiki/F1_score)
    :param logits: B x ... for example B x n_classes x H x W for segmentation or B x n_classes for classification
    :param labels: same as above
    :param beta:
    :return:
    """
    __small_value = 1e-6

    probs = F.sigmoid(logits)

    true_positive = torch.sum(labels * probs, 1)
    soft_tp_plus_fp = torch.sum(probs, 1) + __small_value
    precision = true_positive / soft_tp_plus_fp

    soft_tp_plus_fn = torch.sum(labels, 1) + __small_value
    recall = true_positive / soft_tp_plus_fn
    f1_soft = (
        (1 + beta ** 2)
        * precision
        * recall
        / (beta ** 2 * precision + recall + __small_value)
    )
    return 1 - f1_soft
