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
# flake8: noqa
from typing import Dict, Optional, Union

from absl import logging

from tf_transformers.core import ModelWrapper
from tf_transformers.core.read_from_hub import (
    get_config_cache,
    get_config_only,
    load_pretrained_model,
)
from tf_transformers.models.bart import BartEncoder as Encoder
from tf_transformers.models.bart.configuration_bart import BartConfig as ModelConfig
from tf_transformers.models.bart.convert import convert_bart_pt as convert_pt
from tf_transformers.models.bart.convert import convert_bart_tf as convert_tf
from tf_transformers.models.encoder_decoder import EncoderDecoder
from tf_transformers.utils.docstring_file_utils import add_start_docstrings
from tf_transformers.utils.docstring_utils import (
    ENCODER_MODEL_CONFIG_DOCSTRING,
    ENCODER_PRETRAINED_DOCSTRING,
)

MODEL_TO_HF_URL = {
    "bart-base": "tftransformers/bart-base",
    "bart-medium": "tftransformers/bart-medium",
    "facebook/bart-base": "tftransformers/bart-base",
    "facebook/bart-medium": "tftransformers/bart-medium",
}
code_example = r'''

        >>> from tf_transformers.models import  BartModel
        >>> model = BartModel.from_pretrained("bart-small")
        >>> batch_size = 5
        >>> encoder_sequence_length = 64 # Encoder
        >>> decoder_sequence_length = 37 # Decoder
        >>> encoder_input_ids = tf.random.uniform(shape=(batch_size, encoder_sequence_length), dtype=tf.int32)
        >>> decoder_input_ids = tf.random.uniform(shape=(batch_size, decoder_sequence_length), dtype=tf.int32)
        >>> encoder_input_mask = tf.ones_like(encoder_input_ids)
        >>> inputs = {{'encoder_input_ids': input_ids, 'encoder_input_mask': encoder_input_mask, "decoder_input_ids": decoder_input_ids}
        >>> outputs = model(inputs)

'''

code_example_summarize = r'''
        >>> import tensorflow as tf
        >>> from tf_transformers.models import BartModel
        >>> from tf_transformers.text import TextDecoder
        >>> model = BartModel.from_pretrained('facebook/bart-large-cnn') # Convert
        >>> model = BartModel.from_pretrained('facebook/bart-large-cnn', use_auto_regressive=True)
        >>> decoder = TextDecoder(model)
        >>> text = 'Text to summarize'
        >>> inputs = tokenizer([text],
                        max_length=1024,
                        return_tensors='tf',
                        padding=True, add_special_tokens=True)
        >>> outputs = decoder.decode(inputs_tf, mode='beam', num_beams=4, max_iterations=96, do_sample=False, eos_id=2)
        >>> outputs = outputs['predicted_ids'][:, 0, :] # Select best
        >>> summaries = tokenizer.batch_decode(outputs.numpy())

'''


class BartModel(ModelWrapper):
    """Bart Encoder Wrapper

    Args:
        model_name (:obj:`str`): Name of the model
        cache_dir  (:obj:`str`): Directory to where model caches. default (:obj:`None`).
        save_checkpoint_cache  (:obj:`bool`): To save model or not.
    """

    def __init__(
        self, model_name: str = 'bart', cache_dir: Union[str, None] = None, save_checkpoint_cache: bool = True
    ):

        super(BartModel, self).__init__(
            model_name=model_name, cache_dir=cache_dir, save_checkpoint_cache=save_checkpoint_cache
        )

    def update_config(self, tft_config: Dict, hf_config: Dict):
        r"""
        Update tft config with hf config

        Args:
            tft_config (:obj:`dict`): Dictionary of tft model config
            hf_config  (:obj:`dict`): Dictionary of hf model config
        Returns:

        """
        tft_config["vocab_size"] = hf_config["vocab_size"]
        tft_config["embedding_size"] = hf_config["d_model"]

        # Assert encoder and decoder has same number of attention heads
        assert hf_config['encoder_ffn_dim'] == hf_config['decoder_ffn_dim']
        tft_config["intermediate_size"] = hf_config["encoder_ffn_dim"]

        # tft_config["type_vocab_size"] = hf_config["type_vocab_size"]
        tft_config["max_position_embeddings"] = hf_config["max_position_embeddings"]

        # Assert encoder and decoder has same number of attention heads
        assert hf_config['encoder_attention_heads'] == hf_config['decoder_attention_heads']
        tft_config["num_attention_heads"] = hf_config["encoder_attention_heads"]
        tft_config["num_hidden_layers"] = hf_config["num_hidden_layers"]

        return tft_config

    @classmethod
    def get_config(cls, model_name: str):
        r"""
        Get config from model name.
        Args:
            model_name (:obj:`str`): Name of the model
        Returns:
            Config (:obj:`dict`)

        """
        # Check if it is under tf_transformers
        if model_name in MODEL_TO_HF_URL:
            URL = MODEL_TO_HF_URL[model_name]
            config_dict = get_config_only(URL)
            return config_dict
        else:
            # Check inside huggingface
            config = ModelConfig()
            config_dict = config.to_dict()
            cls_ref = cls()
            try:
                from transformers import PretrainedConfig

                hf_config = PretrainedConfig.from_pretrained(model_name)
                hf_config = hf_config.to_dict()
                config_dict = cls_ref.update_config(config_dict, hf_config)
                return config_dict
            except Exception as e:
                logging.info("Error: {}".format(e))
                logging.info("Failed loading config from HuggingFace")

    @classmethod
    @add_start_docstrings(
        "Bart Model from config :",
        ENCODER_MODEL_CONFIG_DOCSTRING.format(
            "transformers.models.BartEncoder", "tf_transformers.models.bart.BartConfig"
        ),
    )
    def from_config(cls, config, return_layer=False, encoder_kwargs=None, decoder_kwargs=None, **kwargs):
        if isinstance(config, ModelConfig):
            config_dict = config.to_dict()
        else:
            config_dict = config  # Dummy call to cls, as we need `_update_kwargs_and_config` function to be used here.
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
        encoder_layer = Encoder(config=config_dict, name="bart_encoder", **encoder_kwargs_copy)

        decoder_kwargs_copy = {}
        if decoder_kwargs:
            if not isinstance(decoder_kwargs, dict):
                raise ValueError("decoder kwargs should be dict")
            decoder_kwargs_copy = cls_ref._update_kwargs_and_config(decoder_kwargs, config_dict)

        decoder_layer = Encoder(
            config=config_dict, name="bart_decoder", use_decoder=True, mask_mode="causal", **decoder_kwargs_copy
        )
        # Share embeddings
        decoder_layer._embedding_layer = encoder_layer._embedding_layer

        model_layer = EncoderDecoder(encoder_layer, decoder_layer)
        model = model_layer.get_model()
        logging.info("Create model from config")
        if return_layer:
            return model_layer
        return model

    @classmethod
    @add_start_docstrings(
        "Bart Model Pretrained with example :",
        ENCODER_PRETRAINED_DOCSTRING.format(
            "tf_transformers.models.BartModel", "tf_transformers.models.BartEncoder", "bart-small", code_example
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
        # Check if model is in out Huggingface cache
        if model_name in MODEL_TO_HF_URL:
            URL = MODEL_TO_HF_URL[model_name]
            config_dict, local_cache = get_config_cache(URL)

            encoder_kwargs_copy = {}
            if encoder_kwargs:
                if not isinstance(encoder_kwargs, dict):
                    raise ValueError("encoder kwargs should be dict")
                encoder_kwargs_copy = cls_ref._update_kwargs_and_config(encoder_kwargs, config_dict)

            # if a config is provided, we wont be doing any extra .
            # Just create a model and return it with random_weights
            #  (Distribute strategy fails)
            encoder_layer = Encoder(config=config_dict, name="bart_encoder", **encoder_kwargs_copy)

            decoder_kwargs_copy = {}
            if decoder_kwargs:
                if not isinstance(decoder_kwargs, dict):
                    raise ValueError("decoder kwargs should be dict")
                decoder_kwargs_copy = cls_ref._update_kwargs_and_config(decoder_kwargs, config_dict)

            if "use_auto_regressive" in decoder_kwargs_copy:
                del decoder_kwargs_copy["use_auto_regressive"]
            decoder_layer = Encoder(
                config=config_dict,
                name="bart_decoder",
                use_decoder=True,
                mask_mode="causal",
                use_auto_regressive=use_auto_regressive,
                **decoder_kwargs_copy,
            )

            # Share embeddings
            decoder_layer._embedding_layer = encoder_layer._embedding_layer

            model_layer = EncoderDecoder(encoder_layer, decoder_layer)
            model = model_layer.get_model()

            # Load Model
            load_pretrained_model(model, local_cache, URL)
            if return_layer:
                if return_config:
                    return model_layer, config_dict
                return model_layer
            if return_config:
                return model, config_dict
            return model

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
        encoder_layer = Encoder(config=config_dict, name="bart_encoder", **encoder_kwargs_copy)

        decoder_kwargs_copy = {}
        if decoder_kwargs:
            if not isinstance(decoder_kwargs, dict):
                raise ValueError("decoder kwargs should be dict")
            decoder_kwargs_copy = cls_ref._update_kwargs_and_config(decoder_kwargs, config_dict)

        if "use_auto_regressive" in decoder_kwargs_copy:
            del decoder_kwargs_copy["use_auto_regressive"]
        decoder_layer = Encoder(
            config=config_dict,
            name="bart_decoder",
            use_decoder=True,
            mask_mode="causal",
            use_auto_regressive=use_auto_regressive,
            **decoder_kwargs_copy,
        )

        # Share embeddings
        decoder_layer._embedding_layer = encoder_layer._embedding_layer

        model_layer = EncoderDecoder(encoder_layer, decoder_layer)
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
