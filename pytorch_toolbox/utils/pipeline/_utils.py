import time
from pathlib import Path


def create_lookup(path, ignore_paths=[]):
    """Start at `path` and traverses the directory tree and creates a key, value pair lookup for all the classes and functions found in *.py files, except for files in ignore_paths
    
    :param path: root path to start the traversal
    :type path: str or Path
    :param ignore_paths: paths to ignore for lookup creation, defaults to []
    :type ignore_paths: list, optional
    :return: a lookup dictionary that can be used by the Pipeline
    :rtype: dict
    """
    lookup = {}
    for p in Path(path).glob("*"):
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
    try:
        module = load_module_from_path(path)
        lookup = create_lookup_from_module(module)
        return lookup
    except ImportError:  # When import fails
        logging.warning(f"Importing of module from path: {path} has failed")
        return {}


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
