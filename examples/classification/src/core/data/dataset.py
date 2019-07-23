import torch
import torch.utils.data


class NormalizeTensorMixin:
    def normalize_tensor(self, tensor):
        if self.normalize_fn is not None:
            tensor = self.normalize_fn(tensor)
        return tensor


class AugmentMixin:
    def augment(self, image, target=None):
        if target is not None:
            if self.augment_fn is not None:
                image, target = self.augment_fn(image, target)
            else:
                pass
            return image, target
        else:
            if self.augment_fn is not None:
                image.px, _ = self.augment_fn(image, target)
            else:
                pass
            return image.tensor, None


class TTAMixin:
    def tta(self, image, target=None):
        if target is not None:
            if self.tta_fn is not None:
                image, target = self.tta_fn(image, target)
            else:
                pass
            return image, target
        else:
            if self.tta_fn is not None:
                image, _ = self.tta_fn(image, target)
            else:
                pass
            return image, None


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

