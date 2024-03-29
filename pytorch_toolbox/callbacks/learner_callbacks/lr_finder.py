from pytorch_toolbox.defaults import TensorOrNumber, Any
from pytorch_toolbox.callbacks import LearnerCallback, annealing_exp, Scheduler


class LRFinder(LearnerCallback):
    """Causes `learn` to go on a mock training from `start_lr` to `end_lr` for `num_it` iterations.
       Training is interrupted if the loss diverges. Weights changes are reverted after run complete."""

    def __init__(
        self,
        learn,
        start_lr: float = 1e-7,
        end_lr: float = 10,
        num_it: int = 100,
        stop_div: bool = True,
    ):
        "Initialize schedule of learning rates"
        super().__init__(learn)
        self.data, self.stop_div = learn.data, stop_div
        self.scheduler = Scheduler((start_lr, end_lr), num_it, annealing_exp)
        # To avoid validating if the train_dl has less than num_it batches, we put aside the valid_dl and remove it
        # during the call to fit.
        self.valid_dl = learn.data.valid_dl
        self.data.valid_dl = None

    def on_train_begin(self, pbar, **kwargs: Any) -> None:
        "Initialize optimizer and learner hyperparameters."
        setattr(pbar, "clean_on_interrupt", True)
        self.learn.save_model_with_name("tmp")
        self.opt = self.learn.opt
        self.opt.lr = self.scheduler.start
        self.stop, self.best_loss = False, 0.0

    def on_batch_end(
        self, iteration: int, smooth_loss: TensorOrNumber, **kwargs: Any
    ) -> None:
        "Determine if loss has runaway and we should stop."
        if iteration == 0 or smooth_loss < self.best_loss:
            self.best_loss = smooth_loss
        self.opt.lr = self.scheduler.step()
        if self.scheduler.is_done or (self.stop_div and smooth_loss > 4 * self.best_loss):
            # We use the smoothed loss to decide on the stopping since it's less shaky.
            self.stop = True
            return True

    def on_epoch_end(self, **kwargs: Any) -> None:
        "Tell Learner if we need to stop."
        return self.stop

    def on_train_end(self, **kwargs: Any) -> None:
        "Cleanup learn model weights disturbed during LRFind exploration."
        # restore the valid_dl we turned off on `__init__`
        self.data.valid_dl = self.valid_dl
        self.learn.load_model_with_name("tmp")
        if hasattr(self.learn.model, "reset"):
            self.learn.model.reset()
        print(
            "LR Finder is complete, type {learner_name}.recorder.plot() to see the graph."
        )
