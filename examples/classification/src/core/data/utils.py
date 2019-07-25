from functools import partial

import cv2
from abc import ABC, abstractmethod
import torch
import numpy as np


class ClassificationImageSample:
    def __init__(self, image, target):
        self.image = image
        self.target = target

    @classmethod
    def create(
        cls,
        image_data=None,
        image_path=None,
        load_image_fn=partial(cv2.imread, cv2.IMREAD_UNCHANGED),
        label_data=None,
        label_path=None,
        load_label_fn=np.load,
    ):
        image = Image(image_data, image_path, load_image_fn)
        label = Label(label_data, label_path, load_label_fn)
        return cls(image, label)


## TODO: need better name..
class BaseDataSample(ABC):
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

    def _load_if_needed_then_cache(self):
        if self._data is None:
            data = self.load_fn(self.path)
            self._data = data
            return np.array(self._data)
        else:
            return self._data


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


class Mask:
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
