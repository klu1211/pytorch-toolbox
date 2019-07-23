from functools import partial

import cv2
import torch
import numpy as np


class Image:
    def __init__(
        self, data=None, path=None, load_fn=partial(cv2.imread, cv2.IMREAD_UNCHANGED)
    ):
        self._data = data
        self.path = path
        assert not (
            data is None and path is None
        ), "Both the data and the path to the data can't be none, you must define one"
        self.load_fn = load_fn

    @property
    def data(self):
        return self._load_image_if_needed_then_cache()

    def _load_image_if_needed_then_cache(self):
        if self._data is None:
            data = self.load_fn(self.path)
            self._data = data
            return np.array(self._data)
        else:
            return self._data


class Label:
    def __init__(self, data=None, path=None, load_fn=np.load):
        self.data = data
        self.path = path
        assert not (
            data is None and path is None
        ), "Both the data and the path to the data can't be none, you must define one"
        self.load_fn = load_fn

    @property
    def data(self):
        return self._load_label_if_needed_then_cache()

    def _load_label_if_needed_then_cache(self):
        if self._data is None:
            data = self.load_fn(self.path)
            self._data = data
            return np.array(self._data)
        else:
            return self._data
