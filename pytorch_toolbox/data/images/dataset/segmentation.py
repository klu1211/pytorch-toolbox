from ... import BatchElement, ElementType
from .mixins import NormalizeMixin, TTAMixin

## TODO: how can I make the get element more extendable?
## TODO: 1. Allow users to add their own data other than weight maps
## TODO: 2. Don't hard code the augmentation outputs
class ImageSegmentationDataset(
    torch.utils.data.Dataset, SegmentationAugmentMixin, NormalizeTensorMixin, TTAMixin
):
    def __init__(
        self, image_segmentation_samples, augment_fn=None, normalize_fn=None, tta_fn=None
    ):
        super().__init()
        self.image_segmentation_samples = image_segmentation_samples
        self.augment_fn = augment_fn
        self.normalize_fn = normalize_fn
        self.tta_fn = tta_fn

    def __len__(self):
        return len(self.image_segmentation_samples)

    def __getitem__(self, i):
        sample = self.image_segmentation_samples[i]
        image = sample.image
        mask = sample.mask
        weight_map = sample.weight_map
        if mask is None:
            image, _ = self._augment(image_data, target=None)
            image = self._normalize_tensor(image_data, target=None)
            image, _ = self._tta(image_data, target=None)
            sample.image = image
            return self._create_batch_element(sample)
        else:
            if weight_map is None:
                augmented = self._augment(image, masks=[mask])
                image = augmented["image"]
                mask = augmented["masks"][0]
                weight_map = augmented["masks"][1]
            else:
                augmented = self._augment(image, masks=[mask, weight_map])
                image = augmented["image"]
                mask = augmented["masks"][0]
            image = self._normalize_tensor(image)
            sample.image = image
            sample.mask = mask
            sample.weight_map = weight_map
            return self._create_batch_element(sample)

    def _create_batch_element(self, sample):
        image = BatchElement("")
        return ()

        pass
        batch_element = dict(path=path, input=image_data)
        if label_data is not None:
            sample["target"] = label_data
        return sample


class SegmentationAugmentationMixin:
    def augment(self, image, masks=None):
        if masks is not None:
            if self.augment_fn is not None:
                image, masks = self.augment_fn(image, masks)
            else:
                pass
            return image, masks
        else:
            if self.augment_fn is not None:
                image, _ = self.augment_fn(image, masks)
            else:
                pass
            return image, None


class ImageSegmentationSample:
    def __init__(self, image, mask=None, weight_map=None):
        self.image = image
        self.mask = mask
        self.weight_map = weight_map

    @property
    def has_mask(self):
        return self.mask is not None

    @property
    def has_weight_map(self):
        return self.weight_map is not None

    @property
    def augmentables(self):
        augmentables = []
        if self.has_mask:
            augmentables.append(self.mask.data)
        if self.has_weight_map:
            augmentables.append(self.weight_map.data)
        return self.image.data, augmentables

    @classmethod
    def create(
        cls,
        image_data=None,
        image_path=None,
        load_image_fn=partial(cv2.imread, cv2.IMREAD_UNCHANGED),
        mask_data=None,
        mask_path=None,
        load_mask_fn=np.load,
        weight_map_data=None,
        weight_map_path=None,
        load_weight_map_fn=partial(cv2.imread, cv2.IMREAD_UNCHANGED),
    ):
        image = Image(image_data, image_path, load_image_fn)
        mask = Mask(label_data, label_path, load_label_fn)
        weight_map = Mask(weight_map_data, weight_map_path, load_weight_map_fn)
        return cls(image, mask, weight_map)
