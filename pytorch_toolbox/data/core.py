from abc import ABC, abstractmethod
from typing import Callable, Any, Optional
from functools import partial
from enum import Enum
from dataclasses import dataclass

import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from pytorch_toolbox.utils import to_device
from pytorch_toolbox.defaults import default_collate, default_hardware


class DataBunch:
    def __init__(
        self,
        train_dl: DataLoader,
        valid_dl: DataLoader,
        test_dl: Optional[DataLoader] = None,
        device: torch.device = None,
        collate_fn: Callable = default_collate,
        train_collate_fn: Callable = default_collate,
        val_collate_fn: Callable = default_collate,
        test_collate_fn: Callable = default_collate,
    ):
        self.device = default_hardware.device if device is None else device
        self.train_dl = DeviceDataLoader(train_dl, self.device, train_collate_fn)
        self.valid_dl = DeviceDataLoader(valid_dl, self.device, val_collate_fn)
        self.test_dl = (
            DeviceDataLoader(test_dl, self.device, test_collate_fn) if test_dl else None
        )

    @classmethod
    def create(
        cls,
        train_ds: Dataset,
        valid_ds: Dataset,
        test_ds: Dataset = None,
        train_bs: int = 64,
        val_bs: int = None,
        test_bs: int = None,
        sampler=None,
        num_workers: int = default_hardware.cpus,
        pin_memory: bool = False,
        device: torch.device = None,
        collate_fn: Callable = default_collate,
        train_collate_fn: Callable = None,
        val_collate_fn: Callable = None,
        test_collate_fn: Callable = None,
        train_shuffle=True,
        val_shuffle=False,
        test_shuffle=False,
    ):

        if val_bs is None:
            val_bs = train_bs * 2
        if test_bs is None:
            test_bs = train_bs * 2
        train_dl = DataLoader(
            train_ds,
            train_bs,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            shuffle=train_shuffle,
        )
        val_dl = DataLoader(valid_ds, val_bs, shuffle=val_shuffle, num_workers=num_workers)

        if test_ds is None:
            test_dl = None
        else:
            test_dl = DataLoader(
                test_ds, test_bs, shuffle=test_shuffle, num_workers=num_workers
            )
        dls = [train_dl, val_dl, test_dl]
        return cls(
            *dls,
            device=device,
            train_collate_fn=collate_fn if train_collate_fn is None else train_collate_fn,
            val_collate_fn=collate_fn if val_collate_fn is None else val_collate_fn,
            test_collate_fn=collate_fn if test_collate_fn is None else test_collate_fn
        )


class DeviceDataLoader:
    @classmethod
    def create(
        cls,
        dataset: Dataset,
        batch_size: int = 64,
        shuffle: bool = False,
        device: torch.device = default_hardware.device,
        num_workers: int = default_hardware.cpus,
        collate_fn: Callable = default_collate,
        **kwargs: Any
    ):
        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )
        return cls(dl, device=device, collate_fn=collate_fn)

    @property
    def batch_size(self):
        return self.dl.batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        self.dl.batch_size = batch_size

    @property
    def num_workers(self):
        return self.dl.num_workers

    @num_workers.setter
    def num_workers(self, num_workers):
        self.dl.num_workers = num_workers

    def __init__(
        self, dl: DataLoader, device: torch.device, collate_fn: Callable = default_collate
    ):
        self.dl = dl
        self.device = device
        self.dl.collate_fn = collate_fn

    def proc_batch(self, batch):
        input_ = to_device(batch[0], self.device)
        output = {}
        for k, v in batch[1].items():
            if isinstance(v, torch.Tensor):
                output[k] = to_device(v, self.device)
            else:
                output[k] = v
        return input_, output

    def __iter__(self):
        for b in self.dl:
            yield self.proc_batch(b)

    def __len__(self):
        return len(self.dl)


class Data(ABC):
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


class Image(Data):
    pass


class Label(Data):
    pass


class Mask(Data):
    pass


class ElementType(Enum):
    INPUT = 1
    OUTPUT = 2
    AUXILLARY = 3


@dataclass
class BatchElement:
    name: str
    data: Data
    element_type: ElementType
    cudarizable: bool = False
