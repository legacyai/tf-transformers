def get_config(module_name, config_name):
    """Load config from .py based on importlib

    Args:
        module_name ([type]): [description]
        config_name ([type]): [description]

    Raises:
        ValueError: [description]

    """
    import importlib

    my_module = importlib.import_module(module_name)
    config = getattr(my_module, config_name).config
    return config


def get_model_wrapper(model_name):

    import importlib

    model_name = model_name.split("_")[0].strip()  # roberta_base --> roberta
    model_cls = importlib.import_module("tf_transformers.models.model_wrappers.{}_wrapper".format(model_name))
    return model_cls.modelWrapper


def validate_model_name(model_name, allowed_model_names):
    """Validate model_name

    Args:
        model_name ([type]): [description]

    Raises:
        ValueError: [description]
    """
    if model_name not in allowed_model_names:
        raise ValueError("{} not in allowed names {}".format(model_name, allowed_model_names))
