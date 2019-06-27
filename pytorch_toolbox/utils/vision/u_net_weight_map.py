import logging
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BoundingBox:
    x1: int
    x2: int
    y1: int
    y2: int


def create_u_net_weight_map(mask, w_c=0.5, w_0=10, sigma=5):
    """

    :param mask:
    :param w_c:
    :param w_0:
    :param sigma:
    :return:
    """

    mask_contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    logging.debug(f"Number of contours for mask: {len(mask_contours)}")
    bbox = find_bounding_box_for_mask(mask, mask_contours)
    logging.debug(
        f"Bounding box coordinates for mask: ({bbox[0]}, {bbox[2]}), ({bbox[1]}, {bbox[3]})"
    )
    background_points = find_background_points_in_bbox(mask, *bbox)
    logging.debug(f"Number of background points: {len(background_points)}")
    weights_for_background_points = calculate_weights_for_background_points(
        background_points, mask_contours, w_c, w_0, sigma
    )
    weight_map = create_weight_map(
        background_points, weights_for_background_points, mask, w_c
    )
    return weight_map


def find_bounding_box_for_mask(mask, contours, padding=15):
    mask_bbox = create_mask_bbox(mask)
    mask_width, mask_height = reversed(list(mask.shape[:2]))
    for contour in contours:
        contour_bbox = create_contour_bbox(contour)
        mask_bbox = update_mask_bbox(
            mask_bbox, contour_bbox, mask_width, mask_height, padding
        )
    return mask_bbox.x1, mask_bbox.x2, mask_bbox.y1, mask_bbox.y2


def create_mask_bbox(mask):
    return BoundingBox(mask.shape[1], 0, mask.shape[0], 0)


def create_contour_bbox(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return BoundingBox(x, x + w, y, y + h)


def update_mask_bbox(mask_bbox, contour_bbox, mask_width, mask_height, padding):
    if mask_bbox.x1 > contour_bbox.x1:
        mask_bbox.x1 = max(contour_bbox.x1 - padding, 0)
    if mask_bbox.x2 < contour_bbox.x2:
        mask_bbox.x2 = min(contour_bbox.x2 + padding, mask_width)
    if mask_bbox.y1 > contour_bbox.y1:
        mask_bbox.y1 = max(contour_bbox.y1 - padding, 0)
    if mask_bbox.y2 < contour_bbox.y2:
        mask_bbox.y2 = min(contour_bbox.y2 + padding, mask_height)
    return mask_bbox


def find_background_points_in_bbox(mask, min_x, max_x, min_y, max_y):
    y, x = np.where(mask[min_y:max_y, min_x:max_x] == 0)
    y += min_y
    x += min_x
    points = list(zip(y, x))
    return points


def calculate_weights_for_background_points(
    background_points, mask_contours, w_c, w_0, sigma
):
    weights = []
    for point in background_points:
        distances_to_contours = sorted(
            calculate_distances_to_contour_from_point(point, mask_contours), reverse=True
        )
        weight = calculate_weight_from_distances(distances_to_contours, w_c, w_0, sigma)
        weights.append(weight)
    return weights


def calculate_distances_to_contour_from_point(point, contours):
    distances_to_contours = []
    for contour in contours:
        open_cv_point = (point[1], point[0])
        distance_from_point_to_contour = cv2.pointPolygonTest(
            contour, open_cv_point, measureDist=True
        )
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
