import time
import logging
import re
from enum import Enum

from pytorch_toolbox.defaults import *


def to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.cpu().data.numpy()
    else:
        return np.array(t)


def listify(p: OptionalListOrItem = None, q: OptionalListOrItem = None):
    "Make `p` same length as `q`"
    if p is None:
        p = []
    elif isinstance(p, str):
        p = [p]
    elif not isinstance(p, Iterable):
        p = [p]
    n = q if type(q) == int else len(p) if q is None else len(q)
    if len(p) == 1:
        p = p * n
    assert len(p) == n, f"List len mismatch ({len(p)} vs {n})"
    return list(p)


def if_none(a: Any, b: Any) -> Any:
    return b if a is None else a


def is_listy(x: Any) -> bool:
    return isinstance(x, (tuple, list, np.ndarray))


def is_tuple(x: Any) -> bool:
    return isinstance(x, tuple)


def str_to_float(x):
    if isinstance(x, str):
        return float(x)
    elif is_listy(x):
        return [str_to_float(s) for s in x]
    else:
        return x


def range_of(x):
    return list(range(len(x)))


def camel2snake(name: str) -> str:
    _camel_re1 = re.compile("(.)([A-Z][a-z]+)")
    _camel_re2 = re.compile("([a-z0-9])([A-Z])")
    s1 = re.sub(_camel_re1, r"\1_\2", name)
    return re.sub(_camel_re2, r"\1_\2", s1).lower()


def even_mults(start: float, stop: float, n: int) -> np.ndarray:
    "Build evenly stepped schedule from `start` to `stop` in `n` steps."
    mult = stop / start
    step = mult ** (1 / (n - 1))
    return np.array([start * (step ** i) for i in range(n)])


def make_one_hot(labels, n_classes=28):
    one_hots = []
    for label in labels:
        one_hot = np.zeros(n_classes)
        for label_idx in label:
            one_hot[label_idx] = 1
        one_hots.append(one_hot.astype(np.float32))
    return one_hots


class Phase(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3


def timeit(name):
    def wrapper(f):
        def wrapped_f(*args, **kwargs):
            time_start = time.time()
            ret = f(*args, **kwargs)
            time_end = time.time()
            logging.info(f"{name} took {round((time_end - time_start), 2)} seconds")
            return ret

        return wrapped_f

    return wrapper


import time
from pathlib import Path


def create_time_stamped_save_path(save_path, state_dict):
    try:
        save_path = state_dict["save_path"]
    except KeyError:
        state_dict["save_path"] = Path(save_path)
    try:
        current_time = state_dict["start_time"]
    except KeyError:
        current_time = f"{time.strftime('%Y%m%d-%H%M%S')}"
        state_dict["start_time"] = current_time

    current_fold = state_dict.get("current_fold")
    path = Path(save_path, current_time)
    if current_fold is not None:
        path = path / f"Fold_{current_fold}"
    return path


lookup = {"create_time_stamped_save_path": create_time_stamped_save_path}
