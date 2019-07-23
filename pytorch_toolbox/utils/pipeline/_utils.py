import time
from pathlib import Path


def find_relative_best_model_save_path(root_save_path, model_save_path):
    """
    There are two scenarios here:
    1. root_save_path isn't the root directory, therefore the relative_model_save_path is found we will get:
       root_save_path = ~/foo/bar
       model_save_path = ~/foo/bar/model_weights.p
       relative_model_save_path = ~/model_weights.p

       After replacement:
       relative_model_save_path = relative_model_save_path[1:] = model_weights.p

       The first "/" has to be removed from the relative_model_save_path so that the path can be constructed correctly
       This can then be used to construct the full path via the LocalRootSavePath or the DockerRootSavePath

       E.g in our example:
       the model path can be constructed via:
       root_save_path / relative_model_save_path = ~/foo/bar/model_weights.p

    2. root_save_path is the root directory, therefore the relative_model_save_path is found we will get:
       root_save_path = /
       model_save_path = /model_weights.p
       relative_model_save_path = model_weights.p

       After replacement:
       model_save_path = relative_model_save_path = model_weights.p

       Hence there is no need to remove the first "/" from the relative_model_save_path
    """
    relative_model_save_path = str(model_save_path).replace(str(root_save_path), "")
    root_save_path_is_root_directory = Path(root_save_path) == Path("/")
    if root_save_path_is_root_directory:
        relative_model_save_path = relative_model_save_path
    else:
        relative_model_save_path = relative_model_save_path[1:]
    return relative_model_save_path


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
