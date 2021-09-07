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
# ==============================================================================

"""TF Transformers general config file for Transformer based models"""
class TransformerConfig:
    
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.AlbertModel` or a
    :class:`~transformers.TFAlbertModel`. It is used to instantiate an ALBERT model according to the specified
    arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar
    configuration to that of the ALBERT `xxlarge <https://huggingface.co/albert-xxlarge-v2>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30000):
            Vocabulary size of the ALBERT model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.AlbertModel` or
            :class:`~transformers.TFAlbertModel`.
        embedding_size (:obj:`int`, `optional`, defaults to 128):
            Dimensionality of vocabulary embeddings.
        hidden_size (:obj:`int`, `optional`, defaults to 4096):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_hidden_groups (:obj:`int`, `optional`, defaults to 1):
            Number of groups for the hidden layers, parameters in the same group are shared.
        num_attention_heads (:obj:`int`, `optional`, defaults to 64):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 16384):
            The dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        inner_group_num (:obj:`int`, `optional`, defaults to 1):
            The number of inner repetition of attention and ffn.
        hidden_act (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu_new"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.AlbertModel` or
            :class:`~transformers.TFAlbertModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
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

    """
    
    def __init__(self) -> None:
        pass