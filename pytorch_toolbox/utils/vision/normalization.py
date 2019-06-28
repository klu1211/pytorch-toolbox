from functools import partial

import torch


def normalize(tensor, *, mean, std):
    mean = torch.Tensor(mean)
    std = torch.Tensor(std)
    tensor = (tensor - mean[..., None, None]) / std[..., None, None]
    return tensor


def denormalize(tensor, *, mean, std):
    mean = torch.Tensor(mean)
    std = torch.Tensor(std)
    tensor = tensor * std[..., None, None] + mean[..., None, None]
    return tensor


IMAGE_NET_STATS = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

FOUR_CHANNEL_PNASNET5LARGE_STATS = {
    "mean": [0.5, 0.5, 0.5, 0.5],
    "std": [0.5, 0.5, 0.5, 0.5],
}

FOUR_CHANNEL_IMAGE_NET_STATS = {
    "mean": [0.485, 0.456, 0.406, 0.485],
    "std": [0.229, 0.224, 0.224, 0.229],
}
image_net_normalize = partial(normalize, **IMAGE_NET_STATS)
image_net_denormalize = partial(denormalize, **IMAGE_NET_STATS)

four_channel_image_net_normalize = partial(normalize, **FOUR_CHANNEL_IMAGE_NET_STATS)
four_channel_image_net_denormalize = partial(denormalize, **FOUR_CHANNEL_IMAGE_NET_STATS)

four_channel_pnasnet5large_normalize = partial(
    normalize, **FOUR_CHANNEL_PNASNET5LARGE_STATS
)
four_channel_pnasnet5large_denormalize = partial(
    denormalize, **FOUR_CHANNEL_PNASNET5LARGE_STATS
)

normalize_fn_lookup = {
    "image_net_normalize": image_net_normalize,
    "four_channel_image_net_normalize": four_channel_image_net_normalize,
    "four_channel_pnasnet5large_normalize": four_channel_image_net_normalize,
    "normalize": normalize,
    "identity": lambda x: x,
}

denormalize_fn_lookup = {
    "image_net_denormalize": image_net_denormalize,
    "four_channel_image_net_denormalize": four_channel_image_net_denormalize,
    "four_channel_pnasnet5large_denormalize": four_channel_image_net_denormalize,
    "denormalize": denormalize,
    "identity": lambda x: x,
}
