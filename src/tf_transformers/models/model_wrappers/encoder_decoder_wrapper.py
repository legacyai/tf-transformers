import tensorflow as tf
from absl import logging

from tf_transformers.models import EncoderDecoder
from tf_transformers.utils import get_config, get_model_wrapper, validate_model_name

logging.set_verbosity("INFO")


def EncoderDecoderModel(
    encoder_model_name=None,
    decoder_model_name=None,
    model_name=None,
    encoder_checkpoint_dir=None,
    model_checkpoint_dir=None,
    is_training=True,
    use_dropout=True,
    encoder_mask_mode="user_defined",
    warm_start_decoder=True,
    share_attention_layers=True,
    share_encoder_embeddings=False,
    name=None,
    pipeline_mode=None,
    return_all_layer_token_embeddings=False,
    **kwargs,
):
    """Wrapper for Model

    Args:
        model_name ([type]): [description]

    Returns:
        [type]: [description]
    """

    # If model name is given, encoder and decoder
    # will be initiated from the same model
    if model_name:
        if encoder_model_name or decoder_model_name:
            raise ValueError(
                "When `model_name` is set, EncoderDecoderModel will be initialized using  `model_name`. \
                    Please use `encoder_model_name` and `decoder_model_name` \
                    to use 2 different architecture for EncoderDecodeModel"
            )
    if model_checkpoint_dir:
        if encoder_checkpoint_dir:
            raise ValueError("When `model_checkpoint_dir` is set, `encoder_checkpoint_dir` should be None")

    if model_name:
        model_name = model_name.replace("-", "_")  # replace - with _
        encoder_model_name = model_name
        decoder_model_name = model_name
        share_encoder_embeddings = True  # hard set
    else:
        encoder_model_name = encoder_model_name.replace("-", "_")  # replace - with _
        decoder_model_name = decoder_model_name.replace("-", "_")  # replace - with _

    # We cannot share encoder embeddings
    # if models are different. Yes, there is a possibility
    # that two diferent models share same embedding
    if share_encoder_embeddings:
        if encoder_model_name != decoder_model_name:
            raise ValueError("We can `share_encoder_embeddings` only if encoder and decoder are same models")

    encoder_kwargs = {}
    decoder_kwargs = {}

    # pipeline-mode is reqired only for decoder
    # in the case of encoder decoder models
    if is_training is False:
        if pipeline_mode == "auto-regressive":
            decoder_kwargs["pipeline_mode"] = "auto-regressive"
        else:
            # Hard set
            is_training = True
            use_dropout = False

    if "encoder_sequence_length" not in kwargs:
        encoder_sequence_length = None
    else:
        encoder_sequence_length = kwargs["encoder_sequence_length"]
    if "decoder_use_mlm_layer" not in kwargs:
        decoder_kwargs["use_mlm_layer"] = True

    encoder_kwargs["is_training"] = is_training
    decoder_kwargs["is_training"] = is_training
    encoder_kwargs["mask_mode"] = encoder_mask_mode
    encoder_kwargs["return_all_layer_token_embeddings"] = False

    encoder_kwargs["use_dropout"] = use_dropout
    decoder_kwargs["use_dropout"] = use_dropout
    decoder_kwargs["mask_mode"] = "causal"
    decoder_kwargs["return_all_layer_token_embeddings"] = False
    decoder_kwargs["is_decoder"] = True

    if share_attention_layers is False:
        decoder_kwargs["share_attention_layers"] = False

    # Now Lets iterate over kwargs and separate
    # encoder and decoder kwargs because same attributes
    # are avilable for both . So, anything that starts with
    # encoder is encoder kwargs and decoder is decoder kwargs

    for _kwarg in kwargs:
        if _kwarg.startswith("encoder_"):
            _k = _kwarg.split("encoder_")[1]
            encoder_kwargs[_k] = kwargs[_kwarg]
            continue
        if _kwarg.startswith("decoder_"):
            _k = _kwarg.split("decoder_")[1]
            decoder_kwargs[_k] = kwargs[_kwarg]
            continue

    encoder_class = get_model_wrapper(encoder_model_name)
    decoder_class = get_model_wrapper(decoder_model_name)
    encoder_layer, encoder_model, encoder_config = encoder_class(encoder_model_name, **encoder_kwargs)
    if share_encoder_embeddings:
        decoder_kwargs["share_encoder_embeddings"] = True
        decoder_kwargs["encoder_embedding_layer"] = encoder_layer._embedding_layer
        if encoder_layer.use_type_embeddings:
            decoder_kwargs["encoder_type_embedding_layer"] = encoder_layer._type_embeddings
        if encoder_layer.use_positonal_embeddings:
            decoder_kwargs["encoder_positional_embedding_layer"] = encoder_layer._position_embedding_layer

    decoder_layer, decoder_model, decoder_config = decoder_class(decoder_model_name, **decoder_kwargs)

    if encoder_checkpoint_dir:
        encoder_model.load_checkpoint(encoder_checkpoint_dir)
        logging.info("Encoder loaded succesfully from {}".format(encoder_checkpoint_dir))

        del encoder_model
        del decoder_model

        if encoder_model_name == decoder_model_name:
            if warm_start_decoder:
                encoder_layer_dict = {}
                for var in encoder_layer.variables:
                    encoder_layer_dict[var.name] = var

                assigned_values = 0
                for var in decoder_layer.variables:
                    if var.name in encoder_layer_dict:
                        assigned_values += 1
                        var.assign(encoder_layer_dict[var.name])

                logging.info(
                    "Warm started decoder {}/{} variables".format(assigned_values, len(decoder_layer.variables))
                )
                del encoder_layer_dict

    if not name:
        name = "{}_{}".format(encoder_model_name, decoder_model_name)

    tf.keras.backend.clear_session()
    model_layer = EncoderDecoder(
        encoder=encoder_layer,
        decoder=decoder_layer,
        is_training=is_training,
        name=name,
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
