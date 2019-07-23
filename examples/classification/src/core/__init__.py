import importlib
from pathlib import Path

module_names = (
    p.name
    for p in Path(__file__).parent.glob("*")
    if (p.name not in [Path(__file__).name, "__pycache__"])
)
lookup = {}

for name in module_names:
    lookup.update(**importlib.import_module(f".{name}", __name__).lookups)
