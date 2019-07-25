class NormalizeTensorMixin:
    def normalize_tensor(self, tensor):
        if self.normalize_fn is not None:
            tensor = self.normalize_fn(tensor)
        return tensor


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
