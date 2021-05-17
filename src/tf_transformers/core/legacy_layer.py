import tensorflow as tf

from absl import logging
from tf_transformers.core.legacy_model import LegacyModel
from tf_transformers.layers import OnDeviceEmbedding, PositionEmbedding
from abc import ABC, abstractmethod


class LegacyLayer(tf.keras.layers.Layer, ABC):
    """LegacyLayer extends from tf.keras.layers.Layer.
    Base for all the mdoel in tft

    """

    def __init__(self, name, is_training, use_dropout, **kwargs):
        """

        Args:
            name (str): Name of the layer
            is_training (bool): True/False
            use_dropout (bool]): This will help us to disable dropout even in training
        """
        self.is_training = is_training
        self.use_dropout = use_dropout
        if self.is_training is False:
            self.use_dropout = False
        super(LegacyLayer, self).__init__(name=name, **kwargs)

    def get_embedding_layers(self, config):
        """Initializes Embedding layers

        Args:
            config ([dict]): Model config updated with other kwargs from Model
        Returns:
            [tf.keras.layers.Layer]: Returns Word Embedding, Type Embedding,
            Positional Embeddings
        """
        # Word Embedding Layer
        embedding_layer = OnDeviceEmbedding(
            vocab_size=config["vocab_size"],
            embedding_width=config["embedding_size"],
            initializer=config["initializer"],
            name="word_embeddings",
        )
        type_embedding_layer = None
        positional_embedding_layer = None

        # Check for type vocab size and greater than 0 (atleast 1)
        if "type_vocab_size" in config and config["type_vocab_size"] > 0:
            # Type Embeddings
            type_embedding_layer = OnDeviceEmbedding(
                vocab_size=config["type_vocab_size"],
                embedding_width=config["embedding_size"],
                initializer=config["initializer"],
                name="type_embeddings",
            )
        if "max_position_embeddings" in config and config["max_position_embeddings"] > 0:
            # Positional Embedding
            positional_embedding_layer = PositionEmbedding(
                max_sequence_length=config["max_position_embeddings"],
                embedding_width=config["embedding_size"],
                initializer=config["initializer"],
                name="positional_embeddings",
            )
        return embedding_layer, type_embedding_layer, positional_embedding_layer

    @abstractmethod
    def call_encoder(self):
        return

    @abstractmethod
    def call_decoder(self):
        return

    @abstractmethod
    def call_encoder_auto_regressive(self):
        return

    @abstractmethod
    def call_decoder_auto_regressive(self):
        return

    @abstractmethod
    def get_model(self, initialize_only=False):
        """Convert tf.keras.Layer to a tf.keras.Model/LegacyModel.
        Args:
            initialize_only [bool]: If True, just initialize the model, but wont return model object.
        """
        return

    def get_call_method(self, config):
        """This method helps us to choose the call method.

        Args:
            config (dict): Model configs.

        Returns:
            Function: the method.
        """
        # Training Mode
        if config["is_training"]:
            # Decoder Mode
            if config["use_decoder"]:
                return self.call_decoder
            # Encoder Mode
            else:
                return self.call_encoder
        # Inference Mode
        else:
            if config["use_decoder"]:
                call_fn = self.call_decoder
                if config["use_auto_regressive"]:
                    call_fn = self.call_decoder_auto_regressive
            else:
                call_fn = self.call_encoder
                if config["use_auto_regressive"]:
                    call_fn = self.call_encoder_auto_regressive
            return call_fn
        # input_ids = tf.keras.layers.Input(
        #     shape=(self._sequence_length,),
        #     batch_size=self._batch_size,
        #     dtype=tf.int32,
        #     name="input_ids",
        # )
        # input_mask = tf.keras.layers.Input(
        #     shape=(self._sequence_length,),
        #     batch_size=self._batch_size,
        #     dtype=tf.int32,
        #     name="input_mask",
        # )
        # input_type_ids = tf.keras.layers.Input(
        #     shape=(self._sequence_length,),
        #     batch_size=self._batch_size,
        #     dtype=tf.int32,
        #     name="input_type_ids",
        # )
        # inputs = {}

        # if self.cross_attention_inside_encoder:
        #     inputs["encoder_input_ids"] = tf.keras.layers.Input(
        #         shape=(self.sequence_length,),
        #         batch_size=self.batch_size,
        #         dtype=tf.int32,
        #         name="encoder_input_ids",
        #     )
        #     inputs["decoder_input_ids"] = tf.keras.layers.Input(
        #         shape=(self.sequence_length,),
        #         batch_size=self.batch_size,
        #         dtype=tf.int32,
        #         name="decoder_input_ids",
        #     )
        #     if self.use_type_embeddings:
        #         inputs["encoder_input_type_ids"] = tf.keras.layers.Input(
        #             shape=(self.sequence_length,),
        #             batch_size=self.batch_size,
        #             dtype=tf.int32,
        #             name="encoder_input_type_ids",
        #         )
        #         inputs["decoder_input_type_ids"] = tf.keras.layers.Input(
        #             shape=(self.sequence_length,),
        #             batch_size=self.batch_size,
        #             dtype=tf.int32,
        #             name="decoder_input_type_ids",
        #         )
        #     if self.mask_mode in ["user_defined", "prefix"]:
        #         inputs["encoder_input_mask"] = tf.keras.layers.Input(
        #             shape=(self.sequence_length,),
        #             batch_size=self.batch_size,
        #             dtype=tf.int32,
        #             name="encoder_input_mask",
        #         )
        #     if self.is_training is False:
        #         inputs["decoder_all_cache_key"] = tf.keras.layers.Input(
        #             shape=(
        #                 None,
        #                 self.num_attention_heads,
        #                 None,
        #                 self.self.attention_head_size,
        #             ),
        #             dtype=tf.float32,
        #             name="decoder_all_cache_key",
        #         )
        #         inputs["decoder_all_cache_value"] = tf.keras.layers.Input(
        #             shape=(
        #                 None,
        #                 self.num_attention_heads,
        #                 None,
        #                 self.self.attention_head_size,
        #             ),
        #             dtype=tf.float32,
        #             name="decoder_all_cache_value",
        #         )
        #         # self.num_hidden_layers x batch_size x sequence_length x embedding_size
        #         inputs["encoder_hidden_states"] = tf.keras.layers.Input(
        #             shape=(self.sequence_length, self.embedding_size),
        #             batch_size=self.batch_size,
        #             dtype=tf.float32,
        #             name="encoder_hidden_states",
        #         )

        #     layer_outputs = self(inputs)
        #     # We just want to initialize variables
        #     if initialize_only:
        #         return inputs, layer_outputs
        #     # logging.info("Inputs -->")
        #     # for k, v in inputs.items():
        #     #     logging.info("{} ---> {}".format(k, v))
        #     model = LegacyModel(inputs=inputs, outputs=layer_outputs, name=self.name)
        #     model.model_config = {"decoder": self._config_dict}
        #     return model

        # # Encoder
        # if self.is_decoder is False:
        #     input_ids = tf.keras.layers.Input(
        #         shape=(self.sequence_length,),
        #         batch_size=self.batch_size,
        #         dtype=tf.int32,
        #         name="input_ids",
        #     )
        #     input_mask = tf.keras.layers.Input(
        #         shape=(self.sequence_length,),
        #         batch_size=self.batch_size,
        #         dtype=tf.int32,
        #         name="input_mask",
        #     )
        #     input_type_ids = tf.keras.layers.Input(
        #         shape=(self.sequence_length,),
        #         batch_size=self.batch_size,
        #         dtype=tf.int32,
        #         name="input_type_ids",
        #     )

        #     inputs["input_ids"] = input_ids
        #     # When `mask_mode` is `causal` , input_mask is not required
        #     if self.mask_mode in ["user_defined", "prefix"]:
        #         inputs["input_mask"] = input_mask
        #     # Default True in BERT
        #     if self.use_type_embeddings:
        #         inputs["input_type_ids"] = input_type_ids

        #     if self.is_training is False:

        #         if self.pipeline_mode == "auto-regressive":
        #             # Batch size is None
        #             # (12 , None , 12 , None, 64)
        #             # (self.num_hidden_layers,
        #             # batch_size,
        #             # self.num_attention_heads,
        #             # sequence_length,
        #             # self.embedding_size//self.num_attention_heads)
        #             all_cache_key = tf.keras.layers.Input(
        #                 shape=(
        #                     None,
        #                     self.num_attention_heads,
        #                     None,
        #                     self.attention_head_size,
        #                 ),
        #                 dtype=tf.float32,
        #                 name="all_cache_key",
        #             )
        #             all_cache_value = tf.keras.layers.Input(
        #                 shape=(
        #                     None,
        #                     self.num_attention_heads,
        #                     None,
        #                     self.attention_head_size,
        #                 ),
        #                 dtype=tf.float32,
        #                 name="all_cache_value",
        #             )
        #             # Here batch_size = 1 , means we are dealing with vector for past_length
        #             past_length = tf.keras.layers.Input(shape=(None,), batch_size=1, dtype=tf.int32,
        #                   name="past_length")
        #             inputs["all_cache_key"] = all_cache_key
        #             inputs["all_cache_value"] = all_cache_value
        #             inputs["past_length"] = past_length

        # else:
        #     input_ids = tf.keras.layers.Input(
        #         shape=(self.sequence_length,),
        #         batch_size=self.batch_size,
        #         dtype=tf.int32,
        #         name="decoder_input_ids",
        #     )
        #     input_mask = tf.keras.layers.Input(
        #         shape=(self.sequence_length,),
        #         batch_size=self.batch_size,
        #         dtype=tf.int32,
        #         name="decoder_input_mask",
        #     )
        #     input_type_ids = tf.keras.layers.Input(
        #         shape=(self.sequence_length,),
        #         batch_size=self.batch_size,
        #         dtype=tf.int32,
        #         name="decoder_input_type_ids",
        #     )
        #     encoder_hidden_states = tf.keras.layers.Input(
        #         shape=(self.sequence_length, self.embedding_size),
        #         batch_size=self.batch_size,
        #         dtype=tf.float32,
        #         name="encoder_hidden_states",
        #     )
        #     # batch_size x decoder_input_length x encoder_input_length
        #     decoder_encoder_mask = tf.keras.layers.Input(
        #         shape=(self.sequence_length, None),
        #         batch_size=self.batch_size,
        #         dtype=tf.float32,
        #         name="decoder_encoder_mask",
        #     )

        #     inputs["input_ids"] = input_ids
        #     # When `mask_mode` is `causal` , input_mask is not required
        #     if self.mask_mode in ["user_defined", "prefix"]:
        #         inputs["input_mask"] = input_mask
        #     # Default True in BERT
        #     if self.use_type_embeddings:
        #         inputs["input_type_ids"] = input_type_ids

        #     inputs["encoder_hidden_states"] = encoder_hidden_states
        #     inputs["decoder_encoder_mask"] = decoder_encoder_mask

        #     if self.is_training is False:
        #         if self.pipeline_mode == "auto-regressive":
        #             # Batch size is None
        #             # (12 , None , 12 , None, 64)
        #             # (self.num_hidden_layers,
        #             # batch_size,
        #             # self.num_attention_heads,
        #             # sequence_length,
        #             # self.embedding_size//self.num_attention_heads)
        #             all_cache_key = tf.keras.layers.Input(
        #                 shape=(
        #                     None,
        #                     self.num_attention_heads,
        #                     None,
        #                     self.attention_head_size,
        #                 ),
        #                 dtype=tf.float32,
        #                 name="all_cache_key",
        #             )
        #             all_cache_value = tf.keras.layers.Input(
        #                 shape=(
        #                     None,
        #                     self.num_attention_heads,
        #                     None,
        #                     self.attention_head_size,
        #                 ),
        #                 dtype=tf.float32,
        #                 name="all_cache_value",
        #             )
        #             inputs["all_cache_key"] = all_cache_key
        #             inputs["all_cache_value"] = all_cache_value

        # inputs_spec = {k: v.type_spec for k, v in inputs.items()}
        # layer_outputs = self(inputs_spec)
        # # We just want to initialize variables
        # if initialize_only:
        #     return inputs, layer_outputs
        # # logging.info("Inputs -->")
        # # for k, v in inputs.items():
        # #     logging.info("{} ---> {}".format(k, v))
        # model = LegacyModel(inputs=inputs_spec, outputs=layer_outputs, name=self.name)
        # model.model_config = self._config_dict
        # return model
