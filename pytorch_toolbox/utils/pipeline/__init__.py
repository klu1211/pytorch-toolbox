import inspect
import logging
import importlib.util
import pprint
from pathlib import Path


def create_lookup(path, ignore_paths=[]):
    lookup = {}
    for p in path.glob("*"):
        if p in ignore_paths:
            lookup = {**lookup}
            logging.debug(f"Ignoring file: {str(p)}")
        elif p.suffix == ".py":
            logging.debug(f"Found file: {str(p)} and creating lookup")
            lookup = {**lookup, **create_lookup_from_path(p)}
        elif p.is_dir():
            logging.debug(f"Found folder: {str(p)}")
            lookup = {**lookup, **create_lookup(p, ignore_paths)}
        else:
            logging.debug(f"Nothing here: {str(p)}")
            lookup = {**lookup}
    logging.debug(f"Current lookup is: {pprint.pformat(lookup)}")
    return lookup


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
