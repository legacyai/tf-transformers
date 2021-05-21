import tensorflow as tf
from tf_transformers.utils import get_config
from tf_transformers.core import ModelWrapper
from tf_transformers.models.gpt2 import GPT2Encoder
from tf_transformers.models.gpt2.convert import convert_gpt2_tf, convert_gpt2_pt
from absl import logging

DEFAULT_CONFIG = config = {
    "attention_probs_dropout_prob": 0.1,
    "hidden_act": "gelu",
    "intermediate_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "embedding_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "max_position_embeddings": 1024,
    "num_attention_heads": 12,
    "attention_head_size": 64,
    "num_hidden_layers": 12,
    "type_vocab_size": -1,
    "vocab_size": 50257,
    "layer_norm_epsilon": 1e-05,
    "mask_mode": "causal"
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
    tft_config["embedding_size"] = hf_config["n_embd"]
    tft_config["intermediate_size"] = hf_config["n_ctx"]
    # tft_config["type_vocab_size"] = hf_config["type_vocab_size"]
    # tft_config["max_position_embeddings"] = hf_config["max_position_embeddings"]

    tft_config["num_attention_heads"] = hf_config["n_head"]
    tft_config["num_hidden_layers"] = hf_config["n_layer"]

    if "attention_head_size" in tft_config:
        tft_config["attention_head_size"] = tft_config["embedding_size"] // tft_config["num_attention_heads"]
    return tft_config


class GPT2Model(ModelWrapper):
    """GPT2 Encoder Wrapper"""

    def __init__(self, model_name, cache_dir):
        """
        Args:
            model_name (str): Model name
            cache_dir (str): cache dir to save the mode checkpoints
        """
        super(GPT2Model, self).__init__(cache_dir=cache_dir, model_name=model_name)

    @classmethod
    def get_model(
        cls,
        model_name,
        config=None,
        cache_dir=None,
        model_checkpoint_dir=None,
        convert_from_hf=True,
        return_layer=False,
        convert_fn_type='both',
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
        module_name = "tf_transformers.models.model_configs.gpt2"
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
            model_layer = GPT2Encoder(config, **kwargs)
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
        model_layer = GPT2Encoder(config, **kwargs)
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
                if convert_fn_type == 'both':
                    cls_ref.convert_hf_to_tf(
                        model, convert_tf_fn=convert_gpt2_tf(model, config), convert_pt_fn=convert_gpt2_pt(model, config)
                    )
                if convert_fn_type == 'tf':
                    cls_ref.convert_hf_to_tf(
                        model, convert_tf_fn=convert_gpt2_tf(model, config), convert_pt_fn=None
                    )
                if convert_fn_type == 'pt':
                    cls_ref.convert_hf_to_tf(
                        model, convert_tf_fn=None, convert_pt_fn=convert_gpt2_pt(model, config)
                    )
        if return_layer:
            return model_layer, config
        return model, config
