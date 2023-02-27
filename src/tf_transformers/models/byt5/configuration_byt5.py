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
""" ByT5 model configuration """
from tf_transformers.core import TransformerConfig


class ByT5Config(TransformerConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~tf_transformers.models.T5Model`.
    It is used to instantiate an T5 model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the
    T5 `base <https://huggingface.co/t5-small>`__ architecture.

    Configuration objects inherit from :class:`~tf_transformers.models.TransformerConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~tf_transformers.models.TransformerConfig` for more information.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30000):
            Vocabulary size of the T5 model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~tf_transformers.model.T5Model` or
            :class:`~tf_transformers.models.T5Encoder`.
        embedding_size (:obj:`int`, `optional`, defaults to 128):
            Dimensionality of vocabulary embeddings.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_hidden_layers_decoder (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer decoder, which can be different as in encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        attention_head_size (:obj:`int`):
            Size of attention heads in each layer. Normally (embedding_size//num_attention_heads).
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            The dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        inner_group_num (:obj:`int`, `optional`, defaults to 1):
            The number of inner repetition of attention and ffn.
        hidden_act (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and many are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.T5Model` or
            :class:`~transformers.TFT5Model`.
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
        bidirectional (:obj:`bool`, `optional`, defaults to True):
            For relative positional embeddings, Encoder has :obj:`bidirectional=True`, while Decoder has
            :obj:`bidirectional=False`.
        positional_buckets (:obj:`int`, `optional`, defaults to 32):
            The number of buckets to use for each attention layer. For relative positional embeddings.


    Examples::

        >>> from tf_transformers.models import T5Config, T5Model
        >>> # Initializing an bert-base-uncased style configuration
        >>> configuration = T5Config()

        >>> # Initializing an Bert different style configuration
        >>> configuration_new = T5Config(
        ...      embedding_size=768,
        ...      num_attention_heads=12,
        ...      intermediate_size=3072,
        ...  )

        >>> # Initializing a model from the original configuration
        >>> model = T5Model.from_config(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model._config_dict # This has more details than original configuration
    """

    def __init__(
        self,
        vocab_size=32128,
        embedding_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        attention_head_size=64,
        intermediate_size=2048,
        hidden_act="gelu",
        intermediate_act="relu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=-1,
        type_vocab_size=-1,
        initializer_range=0.02,
        layer_norm_epsilon=1e-6,
        position_embedding_type="relative",
        bidirectional=True,
        positional_buckets=32,
        decoder_start_token_id=0,
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
            bidirectional=bidirectional,
            positional_buckets=positional_buckets,
            decoder_start_token_id=decoder_start_token_id,
        )
