import tensorflow as tf
from absl import logging

from tf_transformers.models import EncoderDecoder, T5Encoder
from tf_transformers.utils import get_config, get_model_wrapper, validate_model_name

logging.set_verbosity("INFO")
allowed_model_names = ["t5_small", "t5_base"]


def modelWrapper(model_name, **kwargs):
    """Wrapper for Model

    Args:
        model_name ([type]): [description]

    Returns:
        [type]: [description]
    """

    name = "t5"
    kwargs["name"] = name + "_encoder"

    model_name = model_name.replace("-", "_")  # replace - with _
    validate_model_name(model_name, allowed_model_names)
    config = get_config("tf_transformers.models.model_configs.t5", model_name)

    for _kwarg in kwargs:
        if _kwarg in config:
            config["_kwarg"] = kwargs[_kwarg]
            logging.info("Overwride {} with {}".format(_kwarg, kwargs[_kwarg]))

    config["bidirectional"] = True  # default False

    if "is_training" not in kwargs:
        kwargs["is_training"] = False
        kwargs["pipeline_mode"] = None
    checkpoint_dir = None
    if "checkpoint_dir" in kwargs:
        checkpoint_dir = kwargs["checkpoint_dir"]
        del kwargs["checkpoint_dir"]
    if "mask_mode" not in kwargs:
        # default for gpt2
        kwargs["mask_mode"] = config["mask_mode"]

    # Decoder mode
    if "is_decoder" in kwargs:
        # Hard set, In the decoder model (bidirectional has to be false)
        if kwargs["is_decoder"] is True:
            config["bidirectional"] = False
            # kwargs["mask_mode"] = "causal"
            kwargs["name"] = name + "_decoder"

    tf.keras.backend.clear_session()
    model_layer = T5Encoder(config, **kwargs)
    model = model_layer.get_model()
    if checkpoint_dir:
        model.load_checkpoint(checkpoint_dir)
    return model_layer, model, config


def T5Model(
    model_name,
    is_training=False,
    use_dropout=False,
    pipeline_mode=None,
    model_checkpoint_dir=None,
    batch_size=None,
    encoder_sequence_length=None,
    decoder_sequence_length=None,
    decoder_mask_mode="causal",
):
    """Wrapper for Model

    Args:
        model_name ([type]): [description]

    Returns:
        [type]: [description]
    """

    encoder_kwargs = {}
    decoder_kwargs = {}

    if is_training is False:
        if pipeline_mode == "auto-regressive":
            decoder_kwargs["pipeline_mode"] = "auto-regressive"
        else:
            # Hard set (Normal pipleline in test mode for non Auto regressive tasks)
            is_training = True
            use_dropout = False

    encoder_kwargs["is_training"] = is_training
    decoder_kwargs["is_training"] = is_training
    encoder_kwargs["batch_size"] = batch_size
    decoder_kwargs["batch_size"] = batch_size
    encoder_kwargs["sequence_length"] = encoder_sequence_length
    encoder_kwargs["mask_mode"] = "user_defined"

    encoder_kwargs["use_dropout"] = use_dropout
    decoder_kwargs["use_dropout"] = use_dropout
    decoder_kwargs["mask_mode"] = decoder_mask_mode
    decoder_kwargs["sequence_length"] = decoder_sequence_length
    decoder_kwargs["is_decoder"] = True

    model_name = model_name.replace("-", "_")  # replace - with _
    encoder_class = get_model_wrapper(model_name)
    encoder_layer, encoder_model, encoder_config = encoder_class(model_name, **encoder_kwargs)
    del encoder_model
    decoder_kwargs["share_encoder_embeddings"] = True
    decoder_kwargs["encoder_embedding_layer"] = encoder_layer._embedding_layer
    # same class with different decoder kwargs
    decoder_layer, decoder_model, decoder_config = encoder_class(model_name, **decoder_kwargs)
    del decoder_model
    tf.keras.backend.clear_session()
    model_layer = EncoderDecoder(
        encoder=encoder_layer,
        decoder=decoder_layer,
        is_training=is_training,
        name=model_name,
        encoder_sequence_length=encoder_sequence_length,
    )
    config = {}
    config["encoder"] = encoder_config
    config["decoder"] = decoder_config
    model = model_layer.get_model()
    if model_checkpoint_dir:
        model = model.load_checkpoint(model_checkpoint_dir)
        logging.info("Model loaded succesfully from {}".format(model_checkpoint_dir))
    return model_layer, model, config
