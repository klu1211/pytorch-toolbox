from .mixins import AugmentMixin, NormalizeMixin, TTAMixin


class ImageClassificationDataset(
    torch.utils.data.Dataset, AugmentMixin, NormalizeTensorMixin, TTAMixin
):
    def __init__(self, images, labels, augment_fn=None, normalize_fn=None, tta_fn=None):
        super().__init(augment_fn, normalize_fn, tta_fn)
        self.images = images
        self.labels = labels
        self.augment_fn = augment_fn
        self.normalize_fn = normalize_fn
        self.tta_fn = tta_fn

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = self.images[i].data
        path = self.images[i].path
        if self.labels is None:
            image_data, _ = self._augment(image_data, target=None)
            image_data = self._normalize_tensor(image_data, target=None)
            image_data, _ = self._tta(image_data, target=None)
            return self._create_sample(path, image_data, label_data=None)
        else:
            label_data = self.labels[i].data
            image_data, label_data = self._augment(image_data, label_data)
            image_data = self._normalize_tensor(image_data)
            image_data, label_data = self._tta(image_data, label_data)
            return self._create_sample(path, image_data, label_data)
        pass

    def _create_sample(self, path, image_data, label_data):
        sample = dict(path=path, input=image_data)
        if label_data is not None:
            sample["target"] = label_data
        return sample


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
