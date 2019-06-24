import importlib
from albumentations.augmentations.transforms import __all__ as albumentation_transforms

albumentation_lookup = {
    name: importlib.import_module("albumentations.augmentations.transforms", name) for name in albumentation_transforms
}

lookup = {
    **albumentation_lookup
}

