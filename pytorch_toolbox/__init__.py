import importlib
from pathlib import Path
from .utils import *

from pytorch_toolbox.utils.pipeline import create_lookup


create_toolbox_lookup = lambda: create_lookup(
    Path(__file__).parent, ignore_paths=[Path(__file__)]
)
