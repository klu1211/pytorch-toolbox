from torch._utils import _unflatten_dense_tensors
from torch.nn.utils import parameters_to_vector

from pytorch_toolbox.callbacks import Callback
from pytorch_toolbox.defaults import *
from pytorch_toolbox.utils.training import split_layers_into_batch_norm_and_non_batch_norm


def get_master(
    layer_groups: ModuleList, flat_master: bool = False
) -> Tuple[List[List[Tensor]], List[List[Tensor]]]:
    "Return two lists, one for the model parameters in FP16 and one for the master parameters in FP32."
    split_groups = split_layers_into_batch_norm_and_non_batch_norm(layer_groups)
    model_params = [
        [param for param in lg.parameters() if param.requires_grad] for lg in split_groups
    ]
    if flat_master:
        master_params = []
        for lg in model_params:
            if len(lg) != 0:
                mp = parameters_to_vector([param.data.float() for param in lg])
                mp = torch.nn.Parameter(mp, requires_grad=True)
                if mp.grad is None:
                    mp.grad = mp.new(*mp.size())
                master_params.append([mp])
            else:
                master_params.append([])
        return model_params, master_params
    else:
        master_params = [
            [param.clone().float().detach() for param in lg] for lg in model_params
        ]
        for mp in master_params:
            for param in mp:
                param.requires_grad = True
        return model_params, master_params


def model_g2master_g(
    model_params: Sequence[Tensor],
    master_params: Sequence[Tensor],
    flat_master: bool = False,
) -> None:
    "Copy the model gradients to the master parameters for the optimizer step."
    if flat_master:
        for model_group, master_group in zip(model_params, master_params):
            if len(master_group) != 0:
                master_group[0].grad.data.copy_(
                    parameters_to_vector([p.grad.data.float() for p in model_group])
                )
    else:
        for model_group, master_group in zip(model_params, master_params):
            for model, master in zip(model_group, master_group):
                if model.grad is not None:
                    if master.grad is None:
                        master.grad = master.data.new(*master.data.size())
                    master.grad.data.copy_(model.grad.data)
                else:
                    master.grad = None


def master2model(
    model_params: Sequence[Tensor],
    master_params: Sequence[Tensor],
    flat_master: bool = False,
) -> None:
    "Copy master parameters to model parameters."
    if flat_master:
        for model_group, master_group in zip(model_params, master_params):
            if len(model_group) != 0:
                for model, master in zip(
                    model_group, _unflatten_dense_tensors(master_group[0].data, model_group)
                ):
                    model.data.copy_(master)
    else:
        for model_group, master_group in zip(model_params, master_params):
            for model, master in zip(model_group, master_group):
                model.data.copy_(master.data)


class MixedPrecision(Callback):
    "Callback that handles mixed-precision training."

    def __init__(self, learn, loss_scale=512.0, flat_master=False):
        self.learn = learn
        self.loss_scale = loss_scale
        self.flat_master = flat_master
        assert torch.backends.cudnn.enabled, "Mixed precision training requires cudnn."

    def on_train_begin(self, **kwargs) -> None:
        # Get a copy of the model params in FP32
        self.model_params, self.master_params = get_master(
            self.learn.layer_groups, self.flat_master
        )
        # Changes the optimizer so that the optimization step is done in FP32.
        opt = self.learn.opt
        mom, wd, beta = opt.mom, opt.wd, opt.beta
        lrs = [lr for lr in self.learn.opt._lr for _ in range(2)]
        opt_params = [{"params": mp, "lr": lr} for mp, lr in zip(self.master_params, lrs)]
        self.learn.opt.opt = self.learn.opt_func(opt_params)
        opt.mom, opt.wd, opt.beta = mom, wd, beta

    def on_batch_begin(self, last_input, last_target, **kwargs):
        return last_input.half(), last_target

    def on_loss_begin(self, last_output: Tensor, **kwargs: Any) -> Tensor:
        "Convert half precision output to FP32 to avoid reduction overflow."
        return last_output.float()

    def on_backward_begin(self, last_loss: Tensor, **kwargs: Any) -> Tensor:
        "Scale gradients up by `loss_scale` to prevent underflow."
        # To avoid gradient underflow, we scale the gradients
        return last_loss * self.loss_scale

    def on_backward_end(self, **kwargs: Any):
        "Convert the gradients back to FP32 and divide them by the scale."
        model_g2master_g(self.model_params, self.master_params, self.flat_master)
        for group in self.master_params:
            for param in group:
                param.grad.div_(self.loss_scale)

    def on_step_end(self, **kwargs: Any) -> None:
        "Update the params from master to model and zero grad."
        # Zeros the gradients of the model since the optimizer is disconnected.
        self.learn.model.zero_grad()
        # Update the params from master to model.
        master2model(self.model_params, self.master_params, self.flat_master)

    def on_train_end(self, **kwargs):
        return
