from typing import List
from dataclasses import dataclass

from pytorch_toolbox.callbacks import Callback


@dataclass
class KeysExtractorCallback(Callback):
    keys: str

    def on_batch_begin(self, last_input, last_target, **kwargs):
        if not isinstance(last_target, dict):
            return last_input, last_target
        else:
            return last_input, [last_target.get(key) for key in self.keys]
