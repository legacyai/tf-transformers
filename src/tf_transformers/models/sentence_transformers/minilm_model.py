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
"""The main wrapper around MiniLM"""
from typing import Dict, Optional, Union

import tensorflow as tf
from absl import logging

from tf_transformers.core import LegacyLayer, LegacyModel, ModelWrapper
from tf_transformers.core.read_from_hub import (
    get_config_cache,
    get_config_only,
    load_pretrained_model,
)
from tf_transformers.models.bert import BertEncoder as Encoder
from tf_transformers.models.minilm.configuration_minilm import (
    MiniLMConfig as ModelConfig,
)
from tf_transformers.utils.docstring_file_utils import add_start_docstrings
from tf_transformers.utils.docstring_utils import (
    ENCODER_MODEL_CONFIG_DOCSTRING,
    ENCODER_PRETRAINED_DOCSTRING,
)

convert_tf = None
MODEL_TO_HF_URL = {
    'sentence-transformers/all-MiniLM-L6-v2': 'tftransformers/all-MiniLM-L6-v2-sentence-transformers',
    'sentence-transformers/all-MiniLM-L6-v1': 'tftransformers/all-MiniLM-L6-v1-sentence-transformers',
    'sentence-transformers/all-MiniLM-L12-v1': 'tftransformers/all-MiniLM-L12-v1-sentence-transformers',
    'sentence-transformers/all-MiniLM-L12-v2': 'tftransformers/all-MiniLM-L12-v2-sentence-transformers',
    'sentence-transformers/multi-qa-MiniLM-L6-dot-v1': 'tftransformers/multi-qa-MiniLM-L6-dot-v1-sentence-transformers',
    'sentence-transformers/multi-qa-MiniLM-L6-cos-v1': 'tftransformers/multi-qa-MiniLM-L6-cos-v1-sentence-transformers',
    'sentence-transformers/paraphrase-MiniLM-L3-v2': 'tftransformers/paraphrase-MiniLM-L3-v2-sentence-transformers',
    'sentence-transformers/paraphrase-MiniLM-L6-v2': 'tftransformers/paraphrase-MiniLM-L6-v2-sentence-transformers',
    'sentence-transformers/paraphrase-MiniLM-L12-v2': 'tftransformers/paraphrase-MiniLM-L12-v2-sentence-transformers',
    'sentence-transformers/msmarco-MiniLM-L6-cos-v5': 'tftransformers/msmarco-MiniLM-L6-cos-v5-sentence-transformers',
    'sentence-transformers/msmarco-MiniLM-L12-cos-v5': 'tftransformers/msmarco-MiniLM-L12-cos-v5-sentence-transformers',
}

code_example = r'''
'''


def convert_pt(model, config, model_name):
    """PT converter
    Args:
        model_hf: HuggingFace Model (TF)
        model: tf_transformers model/layer
        config: dict
    Returns:
        a function
    """
    import numpy as np

    # When dropout, use_auto_regressive is enabled assertion won't work
    try:
        # LegacyLayer
        local_config = model._config_dict
    except Exception as e:
        # LegacyModel
        import traceback

        print(traceback.format_exc())
        logging.error(e)
        local_config = model.model_config

    if local_config['use_dropout']:
        logging.warn("Note: As `use_dropout` is True we will skip Assertions, please verify the model.")
    if local_config['use_auto_regressive']:
        raise ValueError(
            "Please save  model checkpoint without `use_auto_regressive` and then reload it with `use_auto_regressive`."
        )

    # From vars (Transformer variables)
    from_model_vars = [
        "encoder.layer.{}.attention.self.query.weight",
        "encoder.layer.{}.attention.self.query.bias",
        "encoder.layer.{}.attention.self.key.weight",
        "encoder.layer.{}.attention.self.key.bias",
        "encoder.layer.{}.attention.self.value.weight",
        "encoder.layer.{}.attention.self.value.bias",
        "encoder.layer.{}.attention.output.dense.weight",
        "encoder.layer.{}.attention.output.dense.bias",
        "encoder.layer.{}.attention.output.LayerNorm.weight",
        "encoder.layer.{}.attention.output.LayerNorm.bias",
        "encoder.layer.{}.intermediate.dense.weight",
        "encoder.layer.{}.intermediate.dense.bias",
        "encoder.layer.{}.output.dense.weight",
        "encoder.layer.{}.output.dense.bias",
        "encoder.layer.{}.output.LayerNorm.weight",
        "encoder.layer.{}.output.LayerNorm.bias",
    ]

    # To vars (Transformer variables)
    to_model_vars = [
        "tf_transformers/bert/transformer/layer_{}/self_attention/query/kernel:0",
        "tf_transformers/bert/transformer/layer_{}/self_attention/query/bias:0",
        "tf_transformers/bert/transformer/layer_{}/self_attention/key/kernel:0",
        "tf_transformers/bert/transformer/layer_{}/self_attention/key/bias:0",
        "tf_transformers/bert/transformer/layer_{}/self_attention/value/kernel:0",
        "tf_transformers/bert/transformer/layer_{}/self_attention/value/bias:0",
        "tf_transformers/bert/transformer/layer_{}/self_attention_output/kernel:0",
        "tf_transformers/bert/transformer/layer_{}/self_attention_output/bias:0",
        "tf_transformers/bert/transformer/layer_{}/self_attention_layer_norm/gamma:0",
        "tf_transformers/bert/transformer/layer_{}/self_attention_layer_norm/beta:0",
        "tf_transformers/bert/transformer/layer_{}/intermediate/kernel:0",
        "tf_transformers/bert/transformer/layer_{}/intermediate/bias:0",
        "tf_transformers/bert/transformer/layer_{}/output/kernel:0",
        "tf_transformers/bert/transformer/layer_{}/output/bias:0",
        "tf_transformers/bert/transformer/layer_{}/output_layer_norm/gamma:0",
        "tf_transformers/bert/transformer/layer_{}/output_layer_norm/beta:0",
    ]

    # Simple Assertion
    assert len(from_model_vars) == len(to_model_vars)

    # This dictionary maps from -> to dict names
    mapping_dict = {}
    for index in range(len(from_model_vars)):
        for i in range(config["num_hidden_layers"]):
            mapping_dict[from_model_vars[index].format(i)] = to_model_vars[index].format(i)

    # Word Embeddings
    mapping_dict["embeddings.word_embeddings.weight"] = "tf_transformers/bert/word_embeddings/embeddings:0"
    # Positional Embedding
    mapping_dict["embeddings.position_embeddings.weight"] = "tf_transformers/bert/positional_embeddings/embeddings:0"
    # Type Embeddings
    mapping_dict["embeddings.token_type_embeddings.weight"] = "tf_transformers/bert/type_embeddings/embeddings:0"
    mapping_dict["embeddings.LayerNorm.weight"] = "tf_transformers/bert/embeddings/layer_norm/gamma:0"
    mapping_dict["embeddings.LayerNorm.bias"] = "tf_transformers/bert/embeddings/layer_norm/beta:0"
    mapping_dict["pooler.dense.weight"] = "tf_transformers/bert/pooler_transform/kernel:0"
    mapping_dict["pooler.dense.bias"] = "tf_transformers/bert/pooler_transform/bias:0"

    from sentence_transformers import SentenceTransformer

    model_hf = SentenceTransformer(model_name)
    # HF model variable name to variable values, for fast retrieval
    from_to_variable_dict = {
        name.replace('0.auto_model.', ''): var.detach().numpy() for name, var in model_hf.named_parameters()
    }

    # We need variable name to the index where it is stored inside tf_transformers model
    tf_transformers_model_index_dict = {}
    for index, var in enumerate(model.variables):
        tf_transformers_model_index_dict[var.name] = index

        # In auto_regressive mode, positional embeddings variable name has
        # cond extra name. So, in case someone converts in that mode,
        # replace above mapping here, only for positional embeddings
        if var.name == "tf_transformers/bert/cond/positional_embeddings/embeddings:0":
            mapping_dict[
                "embeddings.position_embeddings.weight"
            ] = "tf_transformers/bert/cond/positional_embeddings/embeddings:0"
    # Start assigning HF values to tf_transformers
    # assigned_map and assigned_map_values are used for sanity check if needed
    assigned_map = []
    # assigned_map_values = []
    for original_var, legacy_var in mapping_dict.items():
        index = tf_transformers_model_index_dict[legacy_var]

        if "query/kernel:0" in legacy_var or "key/kernel:0" in legacy_var or "value/kernel:0" in legacy_var:

            # huggingface (2D) to tf_transformers (3D)
            model.variables[index].assign(
                np.reshape(
                    np.transpose(from_to_variable_dict.get(original_var)),
                    (
                        config["embedding_size"],
                        config["num_attention_heads"],
                        config["attention_head_size"],
                    ),
                )
            )
            assigned_map.append((original_var, legacy_var))
            continue
        if "query/bias:0" in legacy_var or "key/bias:0" in legacy_var or "value/bias:0" in legacy_var:

            # huggingface (2D) to tf_transformers (3D)
            model.variables[index].assign(
                np.reshape(
                    from_to_variable_dict.get(original_var),
                    (
                        config["num_attention_heads"],
                        config["attention_head_size"],
                    ),
                )
            )
            assigned_map.append((original_var, legacy_var))
            continue

        if "self_attention_output/kernel:0" in legacy_var:
            # huggingface (3D) to tf_transformers (2D)
            model.variables[index].assign(
                np.reshape(
                    np.transpose(from_to_variable_dict.get(original_var)),
                    (config["embedding_size"], config["num_attention_heads"] * config["attention_head_size"]),
                )
            )
            assigned_map.append((original_var, legacy_var))
            continue

        if "self_attention_output/bias:0" in legacy_var:
            # huggingface (3D) to tf_transformers (2D)
            model.variables[index].assign(
                np.reshape(
                    from_to_variable_dict.get(original_var),
                    (-1),
                )
            )
            assigned_map.append((original_var, legacy_var))
            continue

        if (
            "intermediate/kernel:0" in legacy_var
            or "output/kernel:0" in legacy_var
            or 'pooler_transform/kernel:0' in legacy_var
        ):
            # huggingface (torch transpose
            model.variables[index].assign(np.transpose(from_to_variable_dict.get(original_var)))

            assigned_map.append((original_var, legacy_var))
            continue

        model.variables[index].assign(from_to_variable_dict.get(original_var))
        assigned_map.append((original_var, legacy_var))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text = "This is a long sentence to check how close models are."
    outputs_hf = model_hf.encode([text])

    inputs = tokenizer(text, return_tensors="tf")
    inputs_tf = {}
    inputs_tf["input_ids"] = inputs["input_ids"]
    inputs_tf["input_type_ids"] = inputs["token_type_ids"]
    inputs_tf["input_mask"] = inputs["attention_mask"]
    outputs_tf = model(inputs_tf)

    tf.debugging.assert_near(tf.reduce_sum(outputs_hf), tf.reduce_sum(outputs_tf['sentence_vector']), rtol=1.0)


class SentenceMiniLMEncoder(LegacyLayer):
    def __init__(
        self,
        model,
        activation=None,
        is_training=False,
        use_dropout=False,
        use_bias=False,
        kernel_initializer="truncated_normal",
        dropout_rate=0.2,
        return_all_outputs=False,
        **kwargs,
    ):
        r"""
        MiniLMBased Sentence Encoder

        Args:
            model (:obj:`LegacyLayer/LegacyModel`):
                Transformer Model.
                Eg:`~tf_transformers.model.BertModel`.
            num_classes (:obj:`int`):
                Number of classes
            activation (:obj:`str/tf.keras.Activation`, `optional`, defaults to None): Activation
            is_training (:obj:`bool`, `optional`, defaults to False): To train
            use_dropout (:obj:`bool`, `optional`, defaults to False): Use dropout
            use_bias (:obj:`bool`, `optional`, defaults to True): use bias
            dropout_rate (:obj: `float`, defaults to `0.2`)
            key (:obj: `str`, `optional`, defaults to 128): If specified, we use this
            key in model output dict and pass it through classfication layer. If its a list
            we return a list of logits for joint loss.
            return_all_outputs: (:obj:`bool`) to return all model outputs of base model or not.
            kernel_initializer (:obj:`str/tf.keras.intitalizers`, `optional`, defaults to `truncated_normal`): Initializer for
            linear layer

        """
        super(SentenceMiniLMEncoder, self).__init__(
            is_training=is_training, use_dropout=use_dropout, name=model.name, **kwargs
        )

        self.model = model
        if isinstance(model, LegacyModel):
            self.model_config = model.model_config
        elif isinstance(model, tf.keras.layers.Layer):
            self.model_config = model._config_dict
        self._is_training = is_training
        self._use_dropout = use_dropout
        self.return_all_outputs = return_all_outputs
        # Initialize model
        self.model_inputs, self.model_outputs = self.get_model(initialize_only=True)

    def call(self, inputs):
        """Call"""
        model_outputs = self.model(inputs)
        sentence_vector = tf.reduce_mean(model_outputs['token_embeddings'], axis=1)  # over sequences
        if self.return_all_outputs:
            model_outputs['sentence_vector'] = sentence_vector
        else:
            return {'sentence_vector': sentence_vector}

    def get_model(self, initialize_only=False):
        """Get model"""
        inputs = self.model.input
        layer_outputs = self(inputs)
        if initialize_only:
            return inputs, layer_outputs
        model = LegacyModel(inputs=inputs, outputs=layer_outputs, name="gtr_t5")
        model.model_config = self.model_config
        return model


class SentenceMiniLMModel(ModelWrapper):
    """MiniLMModel Encoder Wrapper"""

    def __init__(
        self, model_name: str = 'bert', cache_dir: Union[str, None] = None, save_checkpoint_cache: bool = True
    ):
        """
        Args:
            model_name (str): Model name
            cache_dir (str): cache dir to save the mode checkpoints
        """
        super(SentenceMiniLMModel, self).__init__(
            model_name=model_name, cache_dir=cache_dir, save_checkpoint_cache=save_checkpoint_cache
        )

    def update_config(self, tft_config: Dict, hf_config: Dict):
        """Update tft config with hf config. Useful while converting.
        Args:
            tft_config: Dict of TFT configuration.
            hf_config: Dict of HF configuration.
        """
        tft_config["vocab_size"] = hf_config["vocab_size"]
        tft_config["embedding_size"] = hf_config["hidden_size"]
        tft_config["intermediate_size"] = hf_config["intermediate_size"]
        tft_config["type_vocab_size"] = hf_config["type_vocab_size"]
        tft_config["max_position_embeddings"] = hf_config["max_position_embeddings"]

        tft_config["num_attention_heads"] = hf_config["num_attention_heads"]
        tft_config["num_hidden_layers"] = hf_config["num_hidden_layers"]

        tft_config['attention_head_size'] = tft_config['embedding_size'] // tft_config['num_attention_heads']

        return tft_config

    @classmethod
    def get_config(cls, model_name: str):
        """Get a config from Huggingface hub if present"""

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
        "Bert Model from config :",
        ENCODER_MODEL_CONFIG_DOCSTRING.format(
            "transformers.models.BertEncoder", "tf_transformers.models.bert.BertConfig"
        ),
    )
    def from_config(cls, config: ModelConfig, return_layer: bool = False, return_all_outputs=False, **kwargs):
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

        kwargs_copy = cls_ref._update_kwargs_and_config(kwargs, config_dict)

        # if a config is provided, we wont be doing any extra .
        # Just create a model and return it with random_weights
        # (Distribute strategy fails)
        model_layer = Encoder(config_dict, **kwargs_copy)
        model_layer = SentenceMiniLMEncoder(model_layer, return_all_outputs=return_all_outputs)
        model = model_layer.get_model()
        logging.info("Create model from config")
        if return_layer:
            return model_layer
        return model

    @classmethod
    @add_start_docstrings(
        "Bert Model Pretrained with example :",
        ENCODER_PRETRAINED_DOCSTRING.format(
            "tf_transformers.models.BertModel", "tf_transformers.models.BertEncoder", "bert-base-uncased", code_example
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
        use_mlm_layer=False,
        skip_hub=False,
        return_all_outputs=False,
        **kwargs,
    ):
        # Load a base config and then overwrite it
        cls_ref = cls(model_name, cache_dir, save_checkpoint_cache)
        # Check if model is in out Huggingface cache
        if model_name in MODEL_TO_HF_URL and skip_hub is False:
            if use_mlm_layer:
                model_name = model_name + '-mlm'
            URL = MODEL_TO_HF_URL[model_name]
            config_dict, local_cache = get_config_cache(URL)
            kwargs_copy = cls_ref._update_kwargs_and_config(kwargs, config_dict)
            model_layer = Encoder(config_dict, **kwargs_copy)
            model_layer = SentenceMiniLMEncoder(model_layer, return_all_outputs=return_all_outputs)
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
            logging.info("Error in config: {}".format(e))
            logging.info("Failed loading config from HuggingFace")

        # if we allow names other than
        # whats in the class, we might not be able
        # to convert from hf properly.
        if "name" in kwargs:
            del kwargs["name"]

        kwargs_copy = cls_ref._update_kwargs_and_config(kwargs, config_dict)
        model_layer = Encoder(config_dict, **kwargs_copy)
        model_layer = SentenceMiniLMEncoder(model_layer, return_all_outputs=return_all_outputs)
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
