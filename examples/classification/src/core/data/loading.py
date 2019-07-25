from pathlib import Path
from pytorch_toolbox.data.image.types import Image, Label


def load_images_and_labels(root_data_path):
    hot_dog_paths, not_hot_dog_paths = load_data_paths(root_data_path)
    hot_dog_labels, not_hot_dog_labels = create_labels(hot_dog_paths, not_hot_dog_paths)

    images = [Image(path) for path in hot_dog_paths + not_hot_dog_paths]
    labels = [Label(value) for value in hot_dog_labels + not_hot_dog_labels]
    return images, labels


def load_data_paths(data_path):
    hot_dogs = Path(data_path).glob("hot_dog/*")
    not_hot_dogs = Path(data_path).glob("not_hot_dog/*")
    return hot_dogs, not_hot_dogs


def create_labels(hot_dog_paths, not_hot_dog_paths):
    return [1 for p in hot_dog_paths], [0 for p in not_hot_dog_paths]
