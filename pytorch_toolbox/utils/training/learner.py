from pytorch_toolbox.utils import listify, str_to_float
from pytorch_toolbox.losses import LossWrapper
from pytorch_toolbox.defaults import default_wd

### Learner creation


def create_learner(
    data,
    model_creator,
    loss_funcs=[],
    metrics=None,
    callbacks_creator=None,
    callback_fns_creator=None,
    weight_decay=default_wd,
    to_fp16=False,
    model_path=None,
):
    """Function to create a learner
    
    :param data: The data bunch to be used by the Learner
    :type data: DataBunch
    :param model_creator: Any pre-initialized torch model
    :type model_creator: parital(nn.Module)
    :param loss_funcs: A list of loss functions, defaults to []
    :type loss_funcs: BaseLoss, optional
    :param metrics: A list of metrics, defaults to None
    :type metrics: look at the metrics in pytorch_toolbox/metrics.py, optional
    :param callbacks_creator: function to create the callbacks when called, defaults to None
    :type callbacks_creator: look at function `create_callbacks` in pytorch_toolbox/training/learner.py, optional
    :param callback_fns_creator: function to create the learner callbacks when called, defaults to None
    :type callback_fns_creator: look at function `create_learner_callbacks` in pytorch_toolbox/training/learner.py, optional
    :param weight_decay: weight decay to be used, defaults to default_wd
    :type weight_decay: float, optional
    :param to_fp16: whether or not to use float16 for training, defaults to False
    :type to_fp16: bool, optional
    :param model_path: path to a saved model, defaults to None
    :type model_path: str or Path, optional
    :return: wrapper to create a Learner class
    :rtype: Learner
    """
    model = model_creator()
    callbacks = callbacks_creator() if callbacks_creator is not None else None
    callback_fns = callback_fns_creator() if callback_fns_creator is not None else None

    from pytorch_toolbox.training import Learner

    learner = Learner(
        data=data,
        model=model,
        weight_decay=weight_decay,
        layer_groups=get_layer_groups(model),
        loss_func=LossWrapper(loss_funcs),
        metrics=metrics,
        callbacks=callbacks,
        callback_fns=callback_fns,
    )
    if model_path is not None:
        learner.load_model_with_path(model_path)
    if to_fp16:
        learner = learner.to_fp16()
    return learner


def get_layer_groups(model):
    try:
        return model.layer_groups
    except AttributeError:
        return None


def create_callbacks(callback_references):
    callbacks = []
    for cb_ref in callback_references:
        try:
            callbacks.append(cb_ref())
        except TypeError:
            callbacks.append(cb_ref)
    return callbacks


def create_learner_callbacks(learner_callback_references):
    callback_fns = []
    for learn_cb_ref in learner_callback_references:
        try:
            callback_fns.append(learn_cb_ref())
        except TypeError:
            callback_fns.append(learn_cb_ref)
    return callback_fns


### Training scheme wrappers:


def training_scheme_one_cycle(learner, lr, epochs, div_factor=25):
    lr = [float(lr_) for lr_ in listify(lr)]
    learner.unfreeze()
    learner.fit_one_cycle(cyc_len=epochs, max_lr=lr, div_factor=div_factor)


def training_scheme_multi_step(
    learner,
    epochs_for_step_for_hyperparameters,
    hyperparameter_names,
    hyperparameter_values,
    start_epoch=None,
    end_epoch=None,
):
    learner.unfreeze()
    learner.fit_multi_step(
        epochs_for_step_for_hyperparameters,
        hyperparameter_names,
        str_to_float(hyperparameter_values),
        start_epoch,
        end_epoch,
    )
