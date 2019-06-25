from functools import partial

import cv2
import torch
import numpy as np


def create_u_net_weight_map(mask, w_c=0.5, w_0=10, sigma=5):
    """

    :param mask:
    :param w_c:
    :param w_0:
    :param sigma:
    :return:
    """

    mask_contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    bbox = find_bounding_box_for_mask(mask, mask_contours)
    background_points = find_background_points_in_bbox(mask, *bbox)
    weights_for_background_points = calculate_weights_for_background_points(background_points,
                                                                            mask_contours, w_c, w_0,
                                                                            sigma)
    weight_map = create_weight_map(background_points, weights_for_background_points, mask, w_c)
    return weight_map


def find_bounding_box_for_mask(mask, contours, padding=15):
    max_x = 0
    min_x = mask.shape[1]
    max_y = 0
    min_y = mask.shape[0]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        min_x_, max_x_ = min(x, min_x), max(x + w, max_x)
        min_y_, max_y_ = min(y, min_y), max(y + h, max_y)
        if min_x_ < min_x:
            min_x = min_x_
        if max_x_ > max_x:
            max_x = max_x_
        if min_y_ < min_y:
            min_y = min_y_
        if max_y_ > max_y:
            max_y = max_y_
    return min_x - padding, max_x + padding, min_y - padding, max_y + padding


def find_background_points_in_bbox(mask, min_x, max_x, min_y, max_y):
    y, x = np.where(mask[min_y:max_y, min_x:max_x] == 0)
    y += min_y
    x += min_x
    points = list(zip(y, x))
    return points


def calculate_weights_for_background_points(background_points, mask_contours, w_c, w_0, sigma):
    weights = []
    for point in background_points:
        distances_to_contours = sorted(calculate_distances_to_contour_from_point(point, mask_contours),
                                       reverse=True)
        weight = calculate_weight_from_distances(distances_to_contours, w_c, w_0, sigma)
        weights.append(weight)
    return weights


def calculate_distances_to_contour_from_point(point, contours):
    distances_to_contours = []
    for contour in contours:
        open_cv_point = (point[1], point[0])
        distance_from_point_to_contour = cv2.pointPolygonTest(contour, open_cv_point, measureDist=True)
        distances_to_contours.append(distance_from_point_to_contour)
    return distances_to_contours


def calculate_weight_from_distances(distances, w_c, w_0, sigma):
    if len(distances) == 1:
        p1 = distances[0]
        return calculate_weight(p1, w_c=w_c, w_0=w_0, sigma=sigma)
    else:
        p1, p2, *_ = distances
        return calculate_weight(p1 + p2, w_c=w_c, w_0=w_0, sigma=sigma)


def calculate_weight(distances, w_c=0.5, w_0=10, sigma=5):
    exponent = np.exp(-(np.power(distances, 2) / (2 * sigma ** 2)))
    return w_c + w_0 * exponent


def create_weight_map(points, weights, mask, w_c):
    weight_map = np.zeros_like(mask).astype(np.float32) + w_c
    for point, weight in zip(points, weights):
        weight_map[point] = weight
    return weight_map


# def create_u_net_weight_map(mask_slices, w_c=0.5, w_0=10, sigma=5):
#     """
#
#     :param mask_slices: each element in mask_slices is a numpy array of type expects np.uint8
#     :param w_c:
#     :param w_0:
#     :param sigma:
#     :return:
#     """
#     all_weights = []
#     all_points = []
#     weight_maps = []
#     #     print(f"creating weight maps with: w_c={w_c}, w_0={w_0}, sigma={sigma}")
#     for mask in mask_slices:
#         mask_contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#         bbox = find_bounding_box_for_mask(mask, mask_contours)
#         background_points = find_background_points_in_bbox(mask, *bbox)
#         weights = []
#         for point in background_points:
#             distances_to_contours = sorted(calculate_distances_to_contour_from_point(point, mask_contours),
#                                            reverse=True)
#             weight = calculate_weight_from_distances(distances_to_contours, w_c, w_0, sigma)
#             weights.append(weight)
#         all_weights.append(weights)
#         all_points.append(background_points)
#     for points, weights in zip(all_points, all_weights):
#         weight_map = np.zeros_like(mask).astype(np.float32) + w_c
#         for point, weight in zip(points, weights):
#             weight_map[point] = weight
#         weight_maps.append(weight_map)
#     return np.array(weight_maps)


def tensor2img(image_tensor, imtype=np.uint8, denormalize_fn=None, scale_factor=255.0):
    shape = image_tensor.shape
    image_tensor = image_tensor.cpu()
    if denormalize_fn is not None:
        image_tensor = denormalize_fn(image_tensor)
    np_img = image_tensor.float().numpy()
    if len(shape) == 4:
        ret = []
        for img in np_img:
            if img.shape[0] == 1:
                img = np.tile(img, (3, 1, 1))
            ret.append(img)
        ret = np.transpose(np.array(ret), (0, 2, 3, 1))
    elif len(shape) == 3:
        if shape[0] == 1:
            np_img = np.tile(np_img, (3, 1, 1))
        ret = np.transpose(np_img, (1, 2, 0))
    elif len(shape) == 2:
        ret = np_img
    else:
        print("Expected a Tensor of C x H x W or B x C x H x W")
        return
    return (ret * scale_factor).astype(imtype)


# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.transpose().flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T.astype(np.uint8)  # Needed to align to RLE direction


def rle_decode_with_nans(rle, shape):
    return rle_decode(str(rle[0])) if str(rle[0]) != 'nan' else np.zeros(shape).astype(np.uint8)


def normalize(tensor, *, mean, std):
    mean = torch.Tensor(mean)
    std = torch.Tensor(std)
    tensor = ((tensor - mean[..., None, None]) / std[..., None, None])
    return tensor


def denormalize(tensor, *, mean, std):
    mean = torch.Tensor(mean)
    std = torch.Tensor(std)
    tensor = tensor * std[..., None, None] + mean[..., None, None]
    return tensor


IMAGE_NET_STATS = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

FOUR_CHANNEL_PNASNET5LARGE_STATS = {
    'mean': [0.5, 0.5, 0.5, 0.5],
    'std': [0.5, 0.5, 0.5, 0.5]
}

FOUR_CHANNEL_IMAGE_NET_STATS = {
    'mean': [0.485, 0.456, 0.406, 0.485],
    'std': [0.229, 0.224, 0.224, 0.229]
}
image_net_normalize = partial(normalize, **IMAGE_NET_STATS)
image_net_denormalize = partial(denormalize, **IMAGE_NET_STATS)

four_channel_image_net_normalize = partial(normalize, **FOUR_CHANNEL_IMAGE_NET_STATS)
four_channel_image_net_denormalize = partial(denormalize, **FOUR_CHANNEL_IMAGE_NET_STATS)

four_channel_pnasnet5large_normalize = partial(normalize, **FOUR_CHANNEL_PNASNET5LARGE_STATS)
four_channel_pnasnet5large_denormalize = partial(denormalize, **FOUR_CHANNEL_PNASNET5LARGE_STATS)

normalize_fn_lookup = {
    "image_net_normalize": image_net_normalize,
    "four_channel_image_net_normalize": four_channel_image_net_normalize,
    "four_channel_pnasnet5large_normalize": four_channel_image_net_normalize,
    "normalize": normalize,
    "identity": lambda x: x
}

denormalize_fn_lookup = {
    "image_net_denormalize": image_net_denormalize,
    "four_channel_image_net_denormalize": four_channel_image_net_denormalize,
    "four_channel_pnasnet5large_denormalize": four_channel_image_net_denormalize,
    "denormalize": denormalize,
    "identity": lambda x: x
}
