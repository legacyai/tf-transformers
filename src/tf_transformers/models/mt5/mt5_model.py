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
from typing import Dict, Optional, Union

from absl import logging

from tf_transformers.core import ModelWrapper
from tf_transformers.models.encoder_decoder import EncoderDecoder
from tf_transformers.models.mt5 import MT5Encoder as Encoder
from tf_transformers.models.mt5.configuration_mt5 import MT5Config as ModelConfig
from tf_transformers.models.mt5.convert import convert_mt5_pt as convert_pt
from tf_transformers.models.mt5.convert import convert_mt5_tf as convert_tf
from tf_transformers.utils.docstring_file_utils import add_start_docstrings
from tf_transformers.utils.docstring_utils import (
    ENCODER_MODEL_CONFIG_DOCSTRING,
    ENCODER_PRETRAINED_DOCSTRING,
)

code_example = r'''

        >>> from tf_transformers.models import  MT5Model
        >>> model = MT5Model.from_pretrained("google/mt5-small")
        >>> batch_size = 5
        >>> encoder_sequence_length = 64 # Encoder
        >>> decoder_sequence_length = 37 # Decoder
        >>> encoder_input_ids = tf.random.uniform(shape=(batch_size, encoder_sequence_length), dtype=tf.int32)
        >>> decoder_input_ids = tf.random.uniform(shape=(batch_size, decoder_sequence_length), dtype=tf.int32)
        >>> encoder_input_mask = tf.ones_like(encoder_input_ids)
        >>> inputs = {{'encoder_input_ids': input_ids, \
            'encoder_input_mask': encoder_input_mask, "decoder_input_ids": decoder_input_ids}
        >>> outputs = model(inputs)

'''


class MT5Model(ModelWrapper):
    """MT5 Encoder Wrapper"""

    def __init__(self, model_name='mt5', cache_dir=None, save_checkpoint_cache=True):
        """
        Args:
            model_name (str): Model name
            cache_dir (str): cache dir to save the mode checkpoints
        """
        super(MT5Model, self).__init__(
            model_name=model_name, cache_dir=cache_dir, save_checkpoint_cache=save_checkpoint_cache
        )

    def update_config(self, tft_config: Dict, hf_config: Dict):
        """Update tft config with hf config. Useful while converting.
        Args:
            tft_config: Dict of TFT configuration.
            hf_config: Dict of HF configuration.
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
    @add_start_docstrings(
        "MT5 Model from config :",
        ENCODER_MODEL_CONFIG_DOCSTRING.format("transformers.models.MT5Encoder", "tf_transformers.models.mt5.MT5Config"),
    )
    def from_config(cls, config, return_layer=False, encoder_kwargs=None, decoder_kwargs=None, **kwargs):

        config_dict = config.to_dict()
        # Dummy call to cls, as we need `_update_kwargs_and_config` function to be used here.
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
            encoder_kwargs_copy = cls_ref._update_kwargs_and_config(encoder_kwargs, config_dict)

        # if a config is provided, we wont be doing any extra .
        # Just create a model and return it with random_weights
        # (Distribute strategy fails)
        config_dict["bidirectional"] = True
        encoder_layer = Encoder(config=config_dict, name="mt5_encoder", **encoder_kwargs_copy)

        decoder_kwargs_copy = {}
        if decoder_kwargs:
            if not isinstance(decoder_kwargs, dict):
                raise ValueError("decoder kwargs should be dict")
            decoder_kwargs_copy = cls_ref._update_kwargs_and_config(decoder_kwargs, config_dict)

        config_dict["bidirectional"] = False
        decoder_layer = Encoder(
            config=config_dict, name="mt5_decoder", use_decoder=True, mask_mode="causal", **decoder_kwargs_copy
        )
        model_layer = EncoderDecoder(encoder_layer, decoder_layer, share_embeddings=True)
        model = model_layer.get_model()
        logging.info("Create model from config")
        if return_layer:
            return model_layer
        return model

    @classmethod
    @add_start_docstrings(
        "MT5 Model Pretrained with example :",
        ENCODER_PRETRAINED_DOCSTRING.format(
            "tf_transformers.models.T5Model", "tf_transformers.models.T5Encoder", "google/mt5-small", code_example
        ),
    )
    def from_pretrained(
        cls,
        model_name: str,
        cache_dir: Union[str, None] = None,
        model_checkpoint_dir: Optional[str] = None,
        convert_from_hf: bool = True,
        return_layer: bool = False,
        return_config: bool = False,
        convert_fn_type: Optional[str] = "both",
        encoder_kwargs: Optional[Dict] = None,
        decoder_kwargs: Optional[Dict] = None,
        save_checkpoint_cache: bool = True,
        load_from_cache: bool = True,
        use_auto_regressive: bool = False,
        **kwargs,
    ):
        # Load a base config and then overwrite it
        cls_ref = cls(model_name, cache_dir, save_checkpoint_cache)
        config = ModelConfig()
        config_dict = config.to_dict()

        try:
            from transformers import PretrainedConfig

            hf_config = PretrainedConfig.from_pretrained(model_name)
            hf_config = hf_config.to_dict()
            config_dict = cls_ref.update_config(config_dict, hf_config)
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
            encoder_kwargs_copy = cls_ref._update_kwargs_and_config(encoder_kwargs, config_dict)

        # if a config is provided, we wont be doing any extra .
        # Just create a model and return it with random_weights
        #  (Distribute strategy fails)
        config_dict["bidirectional"] = True
        encoder_layer = Encoder(config=config_dict, name="mt5_encoder", **encoder_kwargs_copy)

        decoder_kwargs_copy = {}
        if decoder_kwargs:
            if not isinstance(decoder_kwargs, dict):
                raise ValueError("decoder kwargs should be dict")
            decoder_kwargs_copy = cls_ref._update_kwargs_and_config(decoder_kwargs, config_dict)

        config_dict["bidirectional"] = False
        if "use_auto_regressive" in decoder_kwargs_copy:
            del decoder_kwargs_copy["use_auto_regressive"]
        decoder_layer = Encoder(
            config=config_dict,
            name="mt5_decoder",
            use_decoder=True,
            mask_mode="causal",
            use_auto_regressive=use_auto_regressive,
            **decoder_kwargs_copy,
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
                    if load_from_cache:
                        model.load_checkpoint(str(cls_ref.model_path))
                        load_succesfuly = True
                except Exception as e:
                    logging.warn(e)
            if convert_from_hf and not load_succesfuly:
                if convert_fn_type == "both":
                    cls_ref.convert_hf_to_tf(
                        model,
                        config_dict,
                        convert_tf_fn=convert_tf,
                        convert_pt_fn=convert_pt,
                    )
                if convert_fn_type == "tf":
                    cls_ref.convert_hf_to_tf(model, config_dict, convert_tf_fn=convert_tf, convert_pt_fn=None)
                if convert_fn_type == "pt":
                    cls_ref.convert_hf_to_tf(model, config_dict, convert_tf_fn=None, convert_pt_fn=convert_pt)

        if return_layer:
            if return_config:
                return model_layer, config_dict
            return model_layer
        if return_config:
            return model, config_dict
        return model
