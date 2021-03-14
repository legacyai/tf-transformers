import tensorflow as tf
from absl import logging

from tf_transformers.models import BERTEncoder
from tf_transformers.utils import get_config, validate_model_name

logging.set_verbosity("INFO")
allowed_model_names = ["bert_base_cased", "bert_base_uncased", "bert_large_cased", "bert_large_uncased"]


def modelWrapper(model_name, **kwargs):
    """Wrapper for Model

    Args:
        model_name ([type]): [description]

    Returns:
        [type]: [description]
    """

    name = "bert"

    model_name = model_name.replace("-", "_")  # replace - with _
    validate_model_name(model_name, allowed_model_names)
    config = get_config("tf_transformers.models.model_configs.bert", model_name)

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
    model_layer = BERTEncoder(config, **kwargs)
    model = model_layer.get_model()
    if checkpoint_dir:
        model.load_checkpoint(checkpoint_dir)
    return model_layer, model, config
