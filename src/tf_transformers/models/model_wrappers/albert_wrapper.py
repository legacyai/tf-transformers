import tensorflow as tf
from absl import logging

from tf_transformers.models import AlbertEncoder
from tf_transformers.utils import get_config, validate_model_name

logging.set_verbosity("INFO")

allowed_model_names = ["albert_base_v2", "albert_large_v2"]


def modelWrapper(model_name, **kwargs):
    """Wrapper for Model

    Args:
        model_name ([type]): [description]

    Returns:
        [type]: [description]
    """

    name = "albert"

    model_name = model_name.replace("-", "_")  # replace - with _
    validate_model_name(model_name, allowed_model_names)
    config = get_config("tf_transformers.models.model_configs.albert", model_name)

    config_kwargs = []
    for _kwarg in kwargs:
        if _kwarg in config:
            config[_kwarg] = kwargs[_kwarg]
            logging.info("Overwride {} with {}".format(_kwarg, kwargs[_kwarg]))
            config_kwargs.append(_kwarg)

    # If it is in kwargs, Keras Layer will throw error
    if config_kwargs:
        for _kwarg in config_kwargs:
            del kwargs[_kwarg]

    if "is_training" not in kwargs:
        kwargs["is_training"] = False
        kwargs["pipeline_mode"] = None

    if "mask_mode" not in kwargs:
        kwargs["mask_mode"] = config["mask_mode"]

    checkpoint_dir = None
    if "checkpoint_dir" in kwargs:
        checkpoint_dir = kwargs["checkpoint_dir"]
        del kwargs["checkpoint_dir"]
    kwargs["name"] = name
    tf.keras.backend.clear_session()
    model_layer = AlbertEncoder(config, **kwargs)
    model = model_layer.get_model()
    if checkpoint_dir:
        model.load_checkpoint(checkpoint_dir)
    return model_layer, model, config
