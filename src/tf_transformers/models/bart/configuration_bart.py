# coding=utf-8
# Copyright 2021 TF-Transformers Authors.
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
# ====================================================================
""" Bart model configuration """
from tf_transformers.core import TransformerConfig


class BartConfig(TransformerConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~tf_transformers.models.BartModel`.
    It is used to instantiate an ALBERT model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the
    ALBERT `base <https://huggingface.co/bert-base-uncased>`__ architecture.

    Configuration objects inherit from :class:`~tf_transformers.models.TransformerConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~tf_transformers.models.TransformerConfig` for more information.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the ALBERT model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~tf_transformers.model.BartModel` or
            :class:`~tf_transformers.models.BartEncoder`.
        embedding_size (:obj:`int`, `optional`, defaults to 128):
            Dimensionality of vocabulary embeddings.
        embedding_projection_size (:obj:`int`):
            Dimensionality of the encoder layers and the pooler layer. Useful for Bart.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        attention_head_size (:obj:`int`):
            Size of attention heads in each layer. Normally (embedding_size//num_attention_heads).
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            The dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and many are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.BartModel` or
            :class:`~transformers.TFBartModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_epsilon (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        classifier_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for attached classifiers.
        position_embedding_type (:obj:`str`, `optional`, defaults to :obj:`"absolute"`):
            Type of position embedding. Choose one of :obj:`"absolute"`, :obj:`"relative_key"`,
            :obj:`"relative_key_query"`. For positional embeddings use :obj:`"absolute"`. For more information on
            :obj:`"relative_key"`, please refer to `Self-Attention with Relative Position Representations (Shaw et al.)
            <https://arxiv.org/abs/1803.02155>`__. For more information on :obj:`"relative_key_query"`, please refer to
            `Method 4` in `Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)
            <https://arxiv.org/abs/2009.13658>`__.
        num_hidden_groups (:obj:`int`, `optional`, defaults to 1):
            Number of groups for the hidden layers, parameters in the same group are shared.

    Examples::

        >>> from tf_transformers.models import BartConfig, BartModel
        >>> # Initializing an bert-base-uncased style configuration
        >>> configuration = BartConfig()

        >>> # Initializing an Bart different style configuration
        >>> configuration_new = BartConfig(
        ...      embedding_size=768,
        ...      num_attention_heads=12,
        ...      intermediate_size=3072,
        ...  )

        >>> # Initializing a model from the original configuration
        >>> model = BartModel.from_config(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model._config_dict # This has more details than original configuration
    """

    def __init__(
        self,
        vocab_size=50265,
        embedding_size=768,
        num_hidden_layers=12,
        num_attention_heads=64,
        attention_head_size=64,
        intermediate_size=3072,
        hidden_act="gelu",
        intermediate_act="gelu",
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
        max_position_embeddings=1024,
        type_vocab_size=-1,
        initializer_range=0.02,
        layer_norm_epsilon=1e-5,
        position_embedding_type="absolute",
    ):
        super().__init__(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            attention_head_size=attention_head_size,
            hidden_act=hidden_act,
            intermediate_act=intermediate_act,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_epsilon=layer_norm_epsilon,
            position_embedding_type=position_embedding_type,
        )
