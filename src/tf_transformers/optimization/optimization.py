import tensorflow as tf
from absl import logging

from tf_transformers.optimization.adafactor_optimization import AdafactorOptimizer
from tf_transformers.optimization.adam_weighted import AdamWeightDecay
from tf_transformers.optimization.learning_rate_utils import WarmUp, WarmUp_Linear


def get_learning_rate_fn(init_lr, num_train_steps, num_warmup_steps, learning_rate_type):
    """Get learning rate function

    Args:
        num_train_steps ([int]): Train Steps
        num_warmup_steps ([int]): Warmup Steps
        learning_rate_type (str, optional): [description]. Defaults to 'linear'.

    Returns:
        [type]: [description]
    """
    if learning_rate_type == "linear":
        logging.info("Using linear optimization warmup")
        learning_rate_fn = WarmUp_Linear(
            initial_learning_rate=init_lr, num_training_steps=num_train_steps, warmup_steps=num_warmup_steps
        )
        return learning_rate_fn

    if learning_rate_type == "polynomial":
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=init_lr, decay_steps=num_train_steps, end_learning_rate=0.0
        )
        if num_warmup_steps > 0.0:
            logging.info("Using linear optimization warmup")
            learning_rate_fn = WarmUp(
                initial_learning_rate=init_lr, decay_schedule_fn=learning_rate_fn, warmup_steps=num_warmup_steps
            )
        return learning_rate_fn

    if learning_rate_type == "cosine":
        logging.info("Using Cosine decay optimization")
        learning_rate_fn = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=init_lr, decay_steps=num_warmup_steps, alpha=0.0, name="cosine_lr"
        )
        return learning_rate_fn

    logging.info("Not using learning rate fn : Using initial learning rate {}".format(init_lr))
    return init_lr


def create_optimizer(
    init_lr,
    num_train_steps,
    num_warmup_steps,
    learning_rate_type="polynomial",
    adam_beta_2=0.999,
    adam_epsilon=1e-06,
    weight_decay_rate=0.0,
    optimizer_type="adamw",
):
    """Create optimizer based on learning rate.

    Args:
        init_lr ([float]): Initial learning rate
        num_train_steps ([int]): Train steps
        num_warmup_steps ([int]): Warmup Steps
        learning_rate_type (str, optional): ['polynomial' or 'linear']
        adam_beta_2 (float, optional): [description]. Defaults to 0.999.
        adam_epsilon ([type], optional): [description]. Defaults to 1e-06.
        weight_decay_rate (float, optional): [description]. Defaults to 0.0.
        optimizer_type (str, optional): [description]. Defaults to "adamw".

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """

    if optimizer_type == "adafactor":
        logging.info("Using AdaFactor optimizer")
        return AdafactorOptimizer(learning_rate=init_lr)

    learning_rate_fn = get_learning_rate_fn(init_lr, num_train_steps, num_warmup_steps, learning_rate_type)

    if optimizer_type == "adamw":
        logging.info("Using Adamw optimizer")
        optimizer = AdamWeightDecay(
            learning_rate=learning_rate_fn,
            beta_1=0.9,
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
