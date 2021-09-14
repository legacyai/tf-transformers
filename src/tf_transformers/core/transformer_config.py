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

"""TF Transformers general config file for Transformer based models"""

from typing import Any, Dict


class TransformerConfig:

    r"""
    This is the configuration class to store the configuration of a :module:`~tf_transformers.models`.
    It is used to instantiate an Transformer model according to the specified
    arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar
    configuration to base models of most architecture.

    Model Configuration objects must be inherit from :class:`~tf_transformers.models.TransformerConfig` and
    can be used to control the model architecture. Read the documentation from :class:`~tf_transformers.models.TransformerConfig`
    for more information.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30000):
            Vocabulary size of the model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.models.XXXModel`.
        embedding_size (:obj:`int`, `optional`, defaults to 128):
            Dimensionality of vocabulary embeddings.
        embedding_projection_size (:obj:`int`):
            Dimensionality of the encoder layers and the pooler layer. Useful for Albert.
        num_hidden_layers (:obj:`int`):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`):
            Number of attention heads for each attention layer in the Transformer encoder.
        attention_head_size (:obj:`int`):
            Size of attention heads in each layer. Normally (embedding_size//num_attention_heads).
        intermediate_size (:obj:`int`, `optional`, defaults to 16384):
            The dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`Callable`, `optional`"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, and many  are supported.
        intermediate_act (:obj:`str` or :obj:`Callable`, `optional`"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, and many  are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.models.XXXModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_epsilon (:obj:`float`):
            The epsilon used by the layer normalization layers.
        position_embedding_type (:obj:`str`, `optional`, defaults to :obj:`"absolute"`):
            Type of position embedding. Choose one of :obj:`"absolute"`, :obj:`"relative_key"`,
            :obj:`"relative_key_query"`. For positional embeddings use :obj:`"absolute"`. For more information on
            :obj:`"relative_key"`, please refer to `Self-Attention with Relative Position Representations (Shaw et al.)
            <https://arxiv.org/abs/1803.02155>`__. For more information on :obj:`"relative_key_query"`, please refer to
            `Method 4` in `Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)
            <https://arxiv.org/abs/2009.13658>`__.
        num_hidden_groups (:obj:`int`, `optional`, defaults to 1):
            Number of groups for the hidden layers, parameters in the same group are shared.
            This is useful only for models like Albert, where we share parameters across layers.
        bidirectional (:obj:`bool`, `optional`, defaults to True):
            For relative positional embeddings, Encoder has :obj:`bidirectional=True`, while Decoder has
            :obj:`bidirectional=False`.
        positional_buckets (:obj:`int`, `optional`, defaults to 32):


    """

    def __init__(
        self,
        vocab_size,
        embedding_size,
        num_hidden_layers,
        attention_head_size=None,
        num_attention_heads=None,
        intermediate_size=None,
        embedding_projection_size=None,
        hidden_act=None,
        intermediate_act=None,
        hidden_dropout_prob=None,
        attention_probs_dropout_prob=None,
        max_position_embeddings=None,
        type_vocab_size=None,
        initializer_range=None,
        layer_norm_epsilon=None,
        position_embedding_type="absolute",
        num_hidden_groups=1,
        positional_buckets=None,
        bidirectional=None,
        cls_token_id=None,
        sep_token_id=None,
        decoder_start_token_id=None,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
    ):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.embedding_projection_size = embedding_projection_size
        self.hidden_act = hidden_act
        self.intermediate_act = intermediate_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_epsilon = layer_norm_epsilon
        self.position_embedding_type = position_embedding_type
        self.num_hidden_groups = num_hidden_groups
        self.bidirectional = bidirectional
        self.positional_buckets = positional_buckets

        # Convert attributes to dict and del "self" from that
        self._inputs = locals()
        del self._inputs['self']

    def to_dict(self) -> Dict[str, Any]:
        """Convert attributes of class to dict"""
        return self._inputs
