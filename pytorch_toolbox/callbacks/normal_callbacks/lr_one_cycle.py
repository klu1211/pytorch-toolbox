import numpy as np

from pytorch_toolbox.defaults import Floats, StartOptEnd, Any
from pytorch_toolbox.callbacks import Callback, annealing_linear, annealing_cos, Scheduler
from pytorch_toolbox.utils import listify, is_listy
from pytorch_toolbox.utils.training import Phase


class OneCycleScheduler(Callback):
    "Manage 1-Cycle style training as outlined in Leslie Smith's [paper](https://arxiv.org/pdf/1803.09820.pdf)."

    def __init__(
        self,
        learn,
        lr_max: float,
        moms: Floats = (0.95, 0.85),
        div_factor: float = 25.0,
        pct_start: float = 0.3,
    ):
        super().__init__()
        self.learn = learn
        self.lr_max = lr_max
        self.moms = moms
        self.div_factor = div_factor
        self.pct_start = pct_start
        self.moms = tuple(listify(self.moms, 2))
        if is_listy(self.lr_max):
            self.lr_max = np.array(self.lr_max)

    def steps(self, *steps_cfg: StartOptEnd):
        "Build anneal schedule for all of the parameters."
        return [
            Scheduler(step, n_iter, func=func)
            for (step, (n_iter, func)) in zip(steps_cfg, self.phases)
        ]

    def on_train_begin(self, n_epochs: int, **kwargs: Any) -> None:
        "Initialize our optimization params based on our annealing schedule."
        n = len(self.learn.data.train_dl) * n_epochs
        a1 = int(n * self.pct_start)
        a2 = n - a1
        self.phases = ((a1, annealing_linear), (a2, annealing_cos))
        low_lr = self.lr_max / self.div_factor
        self.lr_scheds = self.steps((low_lr, self.lr_max), (self.lr_max, low_lr / 1e4))
        self.mom_scheds = self.steps(self.moms, (self.moms[1], self.moms[0]))
        self.opt = self.learn.opt
        self.opt.lr, self.opt.mom = self.lr_scheds[0].start, self.mom_scheds[0].start
        self.idx_s = 0

    def on_batch_end(self, phase, **kwargs: Any) -> None:
        "Take one step forward on the annealing schedule for the optim params."
        if phase == Phase.TRAIN:
            if self.idx_s >= len(self.lr_scheds):
                return True
            self.opt.lr = self.lr_scheds[self.idx_s].step()
            self.opt.mom = self.mom_scheds[self.idx_s].step()
            # when the current schedule is complete we move onto the next
            # schedule. (in 1-cycle there are two schedules)
            if self.lr_scheds[self.idx_s].is_done:
                self.idx_s += 1
