from abc import ABC, abstractmethod

import cv2
import numpy as np

## TODO: need better name..
class BaseDataSample(ABC):
    def __init__(
        self,
        data=None,
        path=None,
        should_cache=True,
        load_fn=partial(cv2.imread, cv2.IMREAD_UNCHANGED),
    ):
        self._data = data
        self.path = path
        self.should_cache = should_cache
        assert not (
            data is None and path is None
        ), "Both the data and the path to the data can't be none, you must define one"
        self.load_fn = load_fn

    @property
    def data(self):
        return self._load_if_needed_then_cache()

    def _load_if_needed_then_cache(self):
        if self._data is None:
            data = self.load_fn(self.path)
            if self.should_cache:
                self._data = data
            return np.array(data)
        else:
            return self._data


class Image(BaseDataSample):
    pass


class Label(BaseDataSample):
    pass


class Mask(BaseDataSample):
    pass
