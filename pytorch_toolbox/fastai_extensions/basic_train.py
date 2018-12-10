import sys
sys.path.append("../fastai")

import torch

from fastai import *
import fastai

class Learner(fastai.Learner):

    def model_gradients(self):
        for lg in self.layer_groups:
            for l in lg:
                print(l)
                for p in l.parameters():
                    print(p.shape)
                    print(p.requires_grad)

    def predict_on_dl(self, dl, pbar=None, callbacks=None, metrics=None):
        assert dl is not None
        metrics = ifnone(metrics, self.metrics)
        cb_handler = CallbackHandler(self.callbacks + ifnone(callbacks, []), metrics)
        with torch.no_grad():
            self.model.eval()
            for xb, yb in progress_bar(dl, parent=pbar, leave=(pbar is not None)):
                if cb_handler: xb, yb = cb_handler.on_batch_begin(xb, yb, train=False)
                cb_handler = ifnone(cb_handler, CallbackHandler())
                if not is_listy(xb): xb = [xb]
                out = self.model(*xb)
                _ = cb_handler.on_loss_begin(out)

    def predict_on_test_dl(self, pbar=None, callbacks=None, metrics=None):
        """Test with callbacks"""
        dl = ifnone(dl, self.data.test_dl)
        predict_on_dl(dl, pbar, callbacks, metrics)


    def freeze_layer_groups(self, layer_group_idxs):
        if not is_listy(layer_group_idxs): layer_group_idxs = [layer_group_idxs]
        super().unfreeze()
        for i in layer_group_idxs:
            for l in self.layer_groups[i]:
                if not self.train_bn or not isinstance(l, bn_types): requires_grad(l, False)

    def unfreeze_layer_groups(self, layer_group_idxs):
        if not is_listy(layer_group_idxs): layer_group_idxs = [layer_group_idxs]
        layer_group_idxs_to_freeze = list(set(list(range(len(self.layer_groups)))) - set(layer_group_idxs))
        self.freeze_layer_groups(layer_group_idxs_to_freeze)