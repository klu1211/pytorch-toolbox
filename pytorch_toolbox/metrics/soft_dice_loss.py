from pytorch_toolbox.losses import SoftDiceLoss, dice_loss


def soft_dice_loss_metric(
    preds, targs, n_classes, dice_loss_weights=None, aggregate_method="MEAN"
):
    loss = SoftDiceLoss.reshape_to_batch_size_x_minus_one_aggregate_over_last_dimension(
        dice_loss(preds, targs, n_classes=n_classes, dice_loss_weights=dice_loss_weights),
        aggregate_method=aggregate_method,
    )
    return loss.mean()
