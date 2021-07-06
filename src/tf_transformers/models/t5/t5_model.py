# coding=utf-8
# Copyright 2021 TF-Transformers Authors and The TensorFlow Authors.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from absl import logging

from tf_transformers.core import ModelWrapper
from tf_transformers.models.encoder_decoder import EncoderDecoder
from tf_transformers.models.t5 import T5Encoder as Encoder
from tf_transformers.models.t5.convert import convert_t5_pt as convert_pt
from tf_transformers.models.t5.convert import convert_t5_tf as convert_tf
from tf_transformers.utils import get_config

DEFAULT_CONFIG = {
    "attention_probs_dropout_prob": 0.1,
    "hidden_act": "gelu",
    "intermediate_act": "relu",
    "hidden_dropout_prob": 0.1,
    "embedding_size": 512,
    "initializer_range": 0.02,
    "intermediate_size": 2048,
    "max_position_embeddings": -1,
    "num_attention_heads": 8,
    "attention_head_size": 64,
    "num_hidden_layers": 6,
    "vocab_size": 32128,
    "type_vocab_size": -1,
    "layer_norm_epsilon": 1e-06,
    "bidirectional": True,
    "positional_buckets": 32,
}


def normalize_model_name(model_name):
    return model_name.lower().replace("-", "_").strip()


class t5Model(ModelWrapper):
    """t5 Encoder Wrapper"""

    def __init__(self, model_name='alt5', cache_dir=None):
        """
        Args:
            model_name (str): Model name
            cache_dir (str): cache dir to save the mode checkpoints
        """
        super(t5Model, self).__init__(model_name=model_name, cache_dir=cache_dir)

    def update_config(self, tft_config, hf_config):
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

    @classmethod
    def from_config(cls, config, return_layer=False, encoder_kwargs=None, decoder_kwargs=None, **kwargs):

        config = config.copy()
        cls_ref = cls()
        # if we allow names other than
        # whats in the class, we might not be able
        # to convert from hf properly.
        if "name" in kwargs:
            del kwargs["name"]

        encoder_kwargs_copy = {}
        if encoder_kwargs:
            if not isinstance(encoder_kwargs, dict):
                raise ValueError("encoder kwargs should be dict")
            encoder_kwargs_copy = cls_ref._update_kwargs_and_config(encoder_kwargs, config)

        # if a config is provided, we wont be doing any extra .
        # Just create a model and return it with random_weights
        #  (Distribute strategy fails)
        config["bidirectional"] = True
        encoder_layer = Encoder(config=config, name="t5_encoder", **encoder_kwargs_copy)

        decoder_kwargs_copy = {}
        if decoder_kwargs:
            if not isinstance(decoder_kwargs, dict):
                raise ValueError("decoder kwargs should be dict")
            decoder_kwargs_copy = cls_ref._update_kwargs_and_config(decoder_kwargs, config)

        config["bidirectional"] = False
        decoder_layer = Encoder(
            config=config, name="t5_decoder", use_decoder=True, mask_mode="causal", **decoder_kwargs_copy
        )
        model_layer = EncoderDecoder(encoder_layer, decoder_layer, share_embeddings=True)
        model = model_layer.get_model()
        logging.info("Create model from config")
        if return_layer:
            return model_layer
        return model

    @classmethod
    def from_pretrained(
        cls,
        model_name,
        cache_dir=None,
        model_checkpoint_dir=None,
        convert_from_hf=True,
        return_layer=False,
        return_config=False,
        convert_fn_type="both",
        encoder_kwargs=None,
        decoder_kwargs=None,
        **kwargs,
    ):
        """Return tf.keras.Model / LegacyModel .


        Args:
            model_name (str): Name of the model
            cache_dir ([type], optional): [description]. Defaults to None.
            model_checkpoint_dir ([type], optional): [description]. Defaults to None.
            convert_from_hf (bool, optional): [description]. Defaults to True.
            return_layer (bool, optional): [description]. Defaults to False.
            convert_fn_type: ['both' , 'tf', 'pt'] . If both , we use both functions to fallback to another if
            one fails.

        Returns:
            [type]: [description]
        """
        module_name = "tf_transformers.models.model_configs.t5"
        tft_model_name = normalize_model_name(model_name)

        # Load a base config and then overwrite it
        config = DEFAULT_CONFIG.copy()
        cls_ref = cls(model_name, cache_dir)
        try:
            # If a config present as a part of tft load it
            config = get_config(module_name, tft_model_name)
        except Exception as e:
            logging.warn(e)

        try:
            from transformers import PretrainedConfig

            hf_config = PretrainedConfig.from_pretrained(model_name)
            hf_config = hf_config.to_dict()
            config = cls_ref.update_config(config, hf_config)
        except Exception as e:
            logging.info("Error: {}".format(e))
            logging.info("Failed loading config from HuggingFace")

        # if we allow names other than
        # whats in the class, we might not be able
        # to convert from hf properly.
        if "name" in kwargs:
            del kwargs["name"]

        encoder_kwargs_copy = {}
        if encoder_kwargs:
            if not isinstance(encoder_kwargs, dict):
                raise ValueError("encoder kwargs should be dict")
            encoder_kwargs_copy = cls_ref._update_kwargs_and_config(encoder_kwargs, config)

        # if a config is provided, we wont be doing any extra .
        # Just create a model and return it with random_weights
        #  (Distribute strategy fails)
        config["bidirectional"] = True
        encoder_layer = Encoder(config=config, name="t5_encoder", **encoder_kwargs_copy)

        decoder_kwargs_copy = {}
        if decoder_kwargs:
            if not isinstance(decoder_kwargs, dict):
                raise ValueError("decoder kwargs should be dict")
            decoder_kwargs_copy = cls_ref._update_kwargs_and_config(decoder_kwargs, config)

        config["bidirectional"] = False
        decoder_layer = Encoder(
            config=config, name="t5_decoder", use_decoder=True, mask_mode="causal", **decoder_kwargs_copy
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
                except Exception as e:
                    logging.warn(e)
            if convert_from_hf and not load_succesfuly:
                if convert_fn_type == "both":
                    cls_ref.convert_hf_to_tf(
                        model,
                        config,
                        convert_tf_fn=convert_tf,
                        convert_pt_fn=convert_pt,
                    )
                if convert_fn_type == "tf":
                    cls_ref.convert_hf_to_tf(model, config, convert_tf_fn=convert_tf, convert_pt_fn=None)
                if convert_fn_type == "pt":
                    cls_ref.convert_hf_to_tf(model, config, convert_tf_fn=None, convert_pt_fn=convert_pt)

        if return_layer:
            if return_config:
                return model_layer, config
            return model_layer
        if return_config:
            return model, config
        return model
