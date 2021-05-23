import tensorflow as tf
from tf_transformers.utils import get_config
from tf_transformers.core import ModelWrapper
from tf_transformers.models.encoder_decoder import EncoderDecoder
from tf_transformers.models.mt5 import MT5Encoder
from tf_transformers.models.mt5.convert import convert_mt5_tf, convert_mt5_pt
from absl import logging

DEFAULT_CONFIG = config = {
    "attention_probs_dropout_prob": 0.1,
    "hidden_act": "gelu",
    "intermediate_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "embedding_size": 512,
    "initializer_range": 0.02,
    "intermediate_size": 1024,
    "max_position_embeddings": -1,
    "num_attention_heads": 6,
    "attention_head_size": 64,
    "num_hidden_layers": 8,
    "vocab_size": 250112,
    "type_vocab_size": -1,
    "layer_norm_epsilon": 1e-06,
    "bidirectional": True,
    "positional_buckets": 32,
}


def normalize_model_name(model_name):
    return model_name.lower().replace("-", "_").strip()


def update_config(tft_config, hf_config):
    """Update tft config with hf config.

    Args:
        tft_config ([type]): [description]
        hf_config ([type]): [description]
    """

    tft_config["vocab_size"] = hf_config["vocab_size"]
    tft_config["embedding_size"] = hf_config["d_model"]
    tft_config["intermediate_size"] = hf_config["d_ff"]
    # tft_config["type_vocab_size"] = hf_config["type_vocab_size"]
    # tft_config["max_position_embeddings"] = hf_config["max_position_embeddings"]

    tft_config["num_attention_heads"] = hf_config["num_heads"]
    tft_config["num_hidden_layers"] = hf_config["num_layers"]
    tft_config["positional_buckets"] = hf_config["relative_attention_num_buckets"]

    return tft_config


class MT5Model(ModelWrapper):
    """MT5 Encoder Wrapper"""

    def __init__(self, model_name, cache_dir):
        """
        Args:
            model_name (str): Model name
            cache_dir (str): cache dir to save the mode checkpoints
        """
        super(MT5Model, self).__init__(cache_dir=cache_dir, model_name=model_name)

    @classmethod
    def get_model(
        cls,
        model_name,
        config=None,
        use_auto_regressive=False,
        cache_dir=None,
        model_checkpoint_dir=None,
        convert_from_hf=True,
        return_layer=False,
        convert_fn_type="both",
        **kwargs,
    ):
        """Get Model will reurn a tf.keras.Model / LegacyModel .


        Args:
            model_name (str): Name of the model
            config (dict): Model config
            cache_dir ([type], optional): [description]. Defaults to None.
            model_checkpoint_dir ([type], optional): [description]. Defaults to None.
            convert_from_hf (bool, optional): [description]. Defaults to True.
            return_layer (bool, optional): [description]. Defaults to False.
            convert_fn_type: ['both' , 'tf', 'pt'] . If both , we use both functions to fallback to another if
            one fails.

        Returns:
            [type]: [description]
        """
        module_name = "tf_transformers.models.model_configs.mt5"
        tft_model_name = normalize_model_name(model_name)

        if not config:
            try:
                config = get_config(module_name, tft_model_name)
            except:
                # Load a base config and then overwrite it
                config = DEFAULT_CONFIG
                from transformers import PretrainedConfig

                # See if it present in HF
                try:
                    hf_config = PretrainedConfig.from_pretrained(model_name)
                    hf_config = hf_config.to_dict()
                    config = update_config(config, hf_config)
                except:
                    pass

        else:
            # if a config is provided, we wont be doing any extra .
            # Just create a model and return it with random_weights
            tf.keras.backend.clear_session()
            encoder_layer = MT5Encoder(config=config, name="mt5_encoder")

            config["bidirectional"] = False
            decoder_layer = MT5Encoder(
                config=config,
                name="mt5_decoder",
                use_decoder=True,
                mask_mode="causal",
                use_auto_regressive=use_auto_regressive,
            )
            model_layer = EncoderDecoder(encoder_layer, decoder_layer, share_embeddings=True)
            model = model_layer.get_model()
            logging.info("Create model from config")
            if return_layer:
                return model_layer, config
            return model, config

        cls_ref = cls(model_name, cache_dir)
        # if we allow names other than
        # whats in the class, we might not be able
        # to convert from hf properly.
        if "name" in kwargs:
            del kwargs["name"]

        tf.keras.backend.clear_session()
        config["bidirectional"] = True
        encoder_layer = MT5Encoder(config=config, name="mt5_encoder")

        config["bidirectional"] = False
        decoder_layer = MT5Encoder(
            config=config,
            name="mt5_decoder",
            use_decoder=True,
            mask_mode="causal",
            use_auto_regressive=use_auto_regressive,
        )
        model_layer = EncoderDecoder(encoder_layer, decoder_layer, share_embeddings=True)
        model = model_layer.get_model()

        # Give preference to model_checkpoint_dir
        if model_checkpoint_dir:
            model.load_checkpoint(model_checkpoint_dir)
        else:
            load_succesfuly = False
            if cls_ref.model_path.exists():
                try:
                    model.load_checkpoint(str(cls_ref.model_path))
                    load_succesfuly = True
                except:
                    pass
            if convert_from_hf and not load_succesfuly:
                if convert_fn_type == "both":
                    cls_ref.convert_hf_to_tf(
                        model,
                        config,
                        convert_tf_fn=convert_mt5_tf,
                        convert_pt_fn=convert_mt5_pt,
                    )
                if convert_fn_type == "tf":
                    cls_ref.convert_hf_to_tf(model, config, convert_tf_fn=convert_mt5_tf, convert_pt_fn=None)
                if convert_fn_type == "pt":
                    cls_ref.convert_hf_to_tf(model, config, convert_tf_fn=None, convert_pt_fn=convert_mt5_pt)
        if return_layer:
            return model_layer, config
        return model, config
