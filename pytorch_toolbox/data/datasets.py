from . import Image, Label


class ClassificationImageSample:
    def __init__(self, image, target):
        self.image = image
        self.target = target

    @classmethod
    def create(
        cls,
        image_data=None,
        image_path=None,
        load_image_fn=partial(cv2.imread, cv2.IMREAD_UNCHANGED),
        label_data=None,
        label_path=None,
        load_label_fn=np.load,
    ):
        image = Image(image_data, image_path, load_image_fn)
        label = Label(label_data, label_path, load_label_fn)
        return cls(image, label)
