import importlib
import albumentations
from albumentations.augmentations.transforms import __all__ as albumentation_transforms

albumentation_lookup = {
    name: getattr(importlib.import_module("albumentations.augmentations.transforms"), name)
    for name in albumentation_transforms
}

lookup = {"Compose": albumentations.Compose, **albumentation_lookup}
