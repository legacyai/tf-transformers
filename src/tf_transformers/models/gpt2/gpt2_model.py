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
from tf_transformers.models.gpt2 import GPT2Encoder as Encoder
from tf_transformers.models.gpt2.configuration_gpt2 import GPT2Config as ModelConfig
from tf_transformers.models.gpt2.convert import convert_gpt2_pt as convert_pt
from tf_transformers.models.gpt2.convert import convert_gpt2_tf as convert_tf
from tf_transformers.utils.docstring_file_utils import add_start_docstrings
from tf_transformers.utils.docstring_utils import (
    ENCODER_MODEL_CONFIG_DOCSTRING,
    ENCODER_PRETRAINED_DOCSTRING,
)

code_example = r'''

        >>> from tf_transformers.models import  GPT2Model
        >>> model = GPT2Model.from_pretrained("gpt2")
        >>> batch_size = 5
        >>> sequence_length = 64
        >>> input_ids = tf.random.uniform(shape=(batch_size, sequence_length), dtype=tf.int32)
        >>> inputs = {{'input_ids': input_ids}
        >>> outputs = model(inputs)

'''


class GPT2Model(ModelWrapper):
    """GPT2 Encoder Wrapper"""

    def __init__(
        self, model_name: str = 'gpt2', cache_dir: Union[str, None] = None, save_checkpoint_cache: bool = True
    ):
        super(GPT2Model, self).__init__(
            model_name=model_name, cache_dir=cache_dir, save_checkpoint_cache=save_checkpoint_cache
        )

    def update_config(self, tft_config: Dict, hf_config: Dict):
        """Update tft config with hf config.

        Args:
            tft_config ([type]): [description]
            hf_config ([type]): [description]
        """
        tft_config["vocab_size"] = hf_config["vocab_size"]
        tft_config["embedding_size"] = hf_config["n_embd"]
        tft_config["intermediate_size"] = hf_config["n_embd"] * 4
        tft_config["max_position_embeddings"] = hf_config["n_ctx"]

        tft_config["num_attention_heads"] = hf_config["n_head"]
        tft_config["num_hidden_layers"] = hf_config["n_layer"]

        return tft_config

    @classmethod
    @add_start_docstrings(
        "GPT2 Model from config :",
        ENCODER_MODEL_CONFIG_DOCSTRING.format(
            "transformers.models.GPT2Encoder", "tf_transformers.models.gpt2.GPT2Config"
        ),
    )
    def from_config(cls, config: ModelConfig, return_layer: bool = False, **kwargs):
        config_dict = config.to_dict()
        # Dummy call to cls, as we need `_update_kwargs_and_config` function to be used here.
        cls_ref = cls()
        # if we allow names other than whats in the class, we might not be able
        # to convert from hf properly.
        if "name" in kwargs:
            del kwargs["name"]

        kwargs_copy = cls_ref._update_kwargs_and_config(kwargs, config_dict)

        # if a config is provided, we wont be doing any extra .
        # Just create a model and return it with random_weights
        # (Distribute strategy fails)
        model_layer = Encoder(config_dict, **kwargs_copy)
        model = model_layer.get_model()
        logging.info("Create model from config")
        if return_layer:
            return model_layer
        return model

    @classmethod
    @add_start_docstrings(
        "GPT2 Model Pretrained with example :",
        ENCODER_PRETRAINED_DOCSTRING.format(
            "tf_transformers.models.GPT2Model", "tf_transformers.models.GPT2Encoder", "gpt2", code_example
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
        save_checkpoint_cache: bool = True,
        load_from_cache: bool = True,
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

        kwargs_copy = cls_ref._update_kwargs_and_config(kwargs, config_dict)
        model_layer = Encoder(config_dict, **kwargs_copy)
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
                return model_layer, config
            return model_layer
        if return_config:
            return model, config
        return model
