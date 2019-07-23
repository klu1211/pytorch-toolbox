import inspect
import importlib.util
from pathlib import Path


def create_lookup_with_relative_modules_from_path(path):
    lookup = {}
    relative_module_paths = get_module_paths_from_path(path)
    for path in relative_module_paths:
        lookup = {**lookup, **create_lookup_from_path(path)}
    return lookup


def get_module_paths_from_path(path):
    return (p for p in path.glob("*") if (p.stem not in ["__init__", "__pycache__"]))


def create_lookup_from_path(path):
    module = load_module_from_path(path)
    lookup = create_lookup_from_module(module)
    return lookup


def load_module_from_path(path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_lookup_from_module(module):
    function_names, function_pointers = load_functions_from_modules(module)
    classes_names, classes_pointers = load_classes_from_modules(module)
    return {
        name: pointer
        for (name, pointer) in zip(
            function_names + classes_names, function_pointers + classes_pointers
        )
    }


def load_functions_from_modules(module):
    try:
        function_names, function_pointers = zip(
            *[
                member
                for member in inspect.getmembers(module, inspect.isfunction)
                if member[1].__module__ == module.__name__
            ]
        )
        return function_names, function_pointers
    except ValueError:  # if there are no functions defined
        return (), ()


def load_classes_from_modules(module):
    try:
        class_names, class_pointers = zip(
            *[
                member
                for member in inspect.getmembers(module, inspect.isclass)
                if member[1].__module__ == module.__name__
            ]
        )
        return class_names, class_pointers
    except ValueError:  # if there are no classes defined
        return (), ()

