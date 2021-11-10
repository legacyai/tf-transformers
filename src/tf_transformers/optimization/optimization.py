import tensorflow as tf
from absl import logging

from tf_transformers.optimization.adafactor_optimization import AdafactorOptimizer
from tf_transformers.optimization.adam_weighted import AdamWeightDecay
from tf_transformers.optimization.learning_rate_utils import WarmUp


def get_learning_rate_fn(init_lr, num_train_steps, num_warmup_steps, decay_function, end_learning_rate):
    """Get learning rate function

    Args:
        num_train_steps ([int]): Train Steps
        num_warmup_steps ([int]): Warmup Steps
        decay_function (str, optional): [description]. Defaults to 'linear'.

    Returns:
        [type]: [description]
    """
    # if decay_function == "linear":
    #     logging.info("Using linear optimization warmup")
    #     learning_rate_fn = WarmUp_Linear(
    #         initial_learning_rate=init_lr, num_training_steps=num_train_steps, warmup_steps=num_warmup_steps
    #     )
    #     return learning_rate_fn

    if decay_function == "linear":
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=init_lr,
            decay_steps=num_train_steps,
            end_learning_rate=end_learning_rate,
            name='linear_lr',
        )
    if decay_function == "cosine":
        logging.info("Using Cosine decay optimization")
        learning_rate_fn = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=init_lr, decay_steps=num_warmup_steps, alpha=0.1, name="cosine_lr"
        )
        if num_warmup_steps > 0.0:
            logging.info("Using linear optimization warmup")
            learning_rate_fn = WarmUp(
                initial_learning_rate=init_lr,
                decay_schedule_fn=learning_rate_fn,
                warmup_steps=num_warmup_steps,
                name="warmup",
            )
        return learning_rate_fn
    logging.info("Not using learning rate fn : Using initial learning rate {}".format(init_lr))
    return init_lr


def create_optimizer(
    init_lr,
    num_train_steps,
    num_warmup_steps=0,
    decay_function="linear",
    adam_beta_1=0.9,
    adam_beta_2=0.999,
    adam_epsilon=1e-06,
    weight_decay_rate=0.0,
    end_learning_rate=0.0,
    optimizer_type="adamw",
):
    r"""
    All optimization functions has to be start from here.

    Args:
        init_lr (:obj:`int`): Intial Learning rate.
        num_train_steps  (:obj:`int`): Total number of training steps (including batch size).
        num_warmup_steps (:obj:`int`): If num_warmup_steps > 0, warmup will be enabled.
        decay_function (:obj:`str`): decay function.
        adam_beta_1 (:obj:`float`):
        adam_beta_2 (:obj:`float`):
        adam_epsilon (:obj:`float`):
        weight_decay_rate (:obj:`float`):
        end_learning_rate (:obj:`float`):
        optimizer_type (:obj:`str`):

    """

    if decay_function not in ["linear", "cosine"]:
        raise ValueError("Invalid decay function {}".format(decay_function))

    if optimizer_type not in ["adamw", "lamb", "adafactor"]:
        raise ValueError("Invalid optimizer type {}".format(optimizer_type))

    if optimizer_type == "adafactor":
        logging.info("Using AdaFactor optimizer")
        return AdafactorOptimizer(learning_rate=init_lr)

    learning_rate_fn = get_learning_rate_fn(
        init_lr, num_train_steps, num_warmup_steps, decay_function, end_learning_rate
    )

    if optimizer_type == "adamw":
        logging.info("Using Adamw optimizer")
        optimizer = AdamWeightDecay(
            learning_rate=learning_rate_fn,
            beta_1=adam_beta_1,
            beta_2=adam_beta_2,
            epsilon=adam_epsilon,
            weight_decay_rate=weight_decay_rate,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
        )
    elif optimizer_type == "lamb":
        import tensorflow_addons.optimizers as tfa_optimizers

        logging.info("using Lamb optimizer")
        optimizer = tfa_optimizers.LAMB(
            learning_rate=learning_rate_fn,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6,
            weight_decay_rate=weight_decay_rate,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
        )
    else:
        raise ValueError("Unsupported optimizer type: ", optimizer_type)

    return optimizer, learning_rate_fn
