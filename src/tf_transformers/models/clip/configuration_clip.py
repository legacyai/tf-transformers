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
""" ViT model configuration """
from tf_transformers.core import TransformerConfig


class CLIPImageConfig(TransformerConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~tf_transformers.models.ViTModel`.
    It is used to instantiate an ViT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the ViT `google/vit-base-patch16-224
    <https://huggingface.co/google/vit-base-patch16-224>`__ architecture.

    Configuration objects inherit from :class:`~tf_transformers.models.TransformerConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~tf_transformers.models.TransformerConfig` for more information.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the ALBERT model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~tf_transformers.model.BertModel` or
            :class:`~tf_transformers.models.BertEncoder`.
        embedding_size (:obj:`int`, `optional`, defaults to 128):
            Dimensionality of vocabulary embeddings.
        embedding_projection_size (:obj:`int`):
            Dimensionality of the encoder layers and the pooler layer. Useful for Bert.
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
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.BertModel` or
            :class:`~transformers.TFBertModel`.
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
        image_size (:obj:`int`, `optional`, defaults to :obj:`224`):
            The size (resolution) of each image.
        patch_size (:obj:`int`, `optional`, defaults to :obj:`16`):
            The size (resolution) of each patch.
        num_channels (:obj:`int`, `optional`, defaults to :obj:`3`):
            The number of input channels.
        num_labels (:obj:`int`, `optional`, defaults to :obj:`1000`):
            Total number of labels by which model has been pre-trained


    Examples::

        >>> from tf_transformers.models import CLIPImageConfig, CLIPImageEncoder
        >>> # Initializing an 'google/vit-base-patch16-224' style configuration
        >>> configuration = CLIPImageConfig()

        >>> # Initializing an ViT different style configuration
        >>> configuration_new = CLIPImageConfig(
        ...      embedding_size=768,
        ...      num_attention_heads=12,
        ...      intermediate_size=3072,
        ...  )

        >>> # Initializing a model from the original configuration
        >>> model = CLIPImageEncoder.from_config(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model._config_dict # This has more details than original configuration

        >>> # To get a config
        >>> model_name = 'openai/clip-vit-base-patch32'
        >>> config = CLIPImage.get_config(model_name)
    """

    def __init__(
        self,
        vocab_size=None,
        embedding_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        attention_head_size=64,
        intermediate_size=3072,
        hidden_act="quick_gelu",
        intermediate_act="quick_gelu",
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
        max_position_embeddings=None,
        type_vocab_size=-1,
        initializer_range=0.02,
        layer_norm_epsilon=1e-5,
        position_embedding_type="absolute",
        image_size=224,
        patch_size=16,
        num_channels=3,
        num_labels=None,
        projection_dim=512,
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
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            num_labels=num_labels,
            projection_dim=projection_dim,
        )


class CLIPTextConfig(TransformerConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~tf_transformers.models.ViTModel`.
    It is used to instantiate an ViT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the ViT `google/vit-base-patch16-224
    <https://huggingface.co/google/vit-base-patch16-224>`__ architecture.

    Configuration objects inherit from :class:`~tf_transformers.models.TransformerConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~tf_transformers.models.TransformerConfig` for more information.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the ALBERT model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~tf_transformers.model.BertModel` or
            :class:`~tf_transformers.models.BertEncoder`.
        embedding_size (:obj:`int`, `optional`, defaults to 128):
            Dimensionality of vocabulary embeddings.
        embedding_projection_size (:obj:`int`):
            Dimensionality of the encoder layers and the pooler layer. Useful for Bert.
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
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.BertModel` or
            :class:`~transformers.TFBertModel`.
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
        image_size (:obj:`int`, `optional`, defaults to :obj:`224`):
            The size (resolution) of each image.
        patch_size (:obj:`int`, `optional`, defaults to :obj:`16`):
            The size (resolution) of each patch.
        num_channels (:obj:`int`, `optional`, defaults to :obj:`3`):
            The number of input channels.
        num_labels (:obj:`int`, `optional`, defaults to :obj:`1000`):
            Total number of labels by which model has been pre-trained


    Examples::

        >>> from tf_transformers.models import CLIPImageConfig, CLIPImageEncoder
        >>> # Initializing an 'openai/clip-vit-base-patch32' style configuration
        >>> configuration = CLIPImageConfig()

        >>> vision_config = configuration['vision_config']
        >>> text_config   = configuration['text_config]
        >>> # Initializing a model from the original configuration
        >>> vision_encoder = CLIPImageEncoder.from_config(vision_config)
        >>> text_encoder = CLIPTextEncoder.from_config(text_config)
        >>> model = CLIPEncoder(vision_encoder, text_encoder, is_training=False, use_dropout=False)
        >>> configuration = model._config_dict # This has more details than original configuration

        >>> # To get a model config
        >>> model_name = 'openai/clip-vit-base-patch32'
        >>> config = CLIPImage.get_config(model_name)
    """

    def __init__(
        self,
        vocab_size=49408,
        embedding_size=512,
        num_hidden_layers=12,
        num_attention_heads=8,
        attention_head_size=64,
        intermediate_size=2048,
        hidden_act="quick_gelu",
        intermediate_act="quick_gelu",
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
        max_position_embeddings=77,
        type_vocab_size=-1,
        initializer_range=0.02,
        layer_norm_epsilon=1e-5,
        position_embedding_type="absolute",
        image_size=None,
        patch_size=None,
        num_channels=None,
        num_labels=None,
        projection_dim=512,
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
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            num_labels=num_labels,
            projection_dim=projection_dim,
        )
