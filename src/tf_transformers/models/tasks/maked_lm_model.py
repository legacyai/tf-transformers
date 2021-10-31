import tensorflow as tf

from tf_transformers.core import LegacyLayer, LegacyModel
from tf_transformers.layers import BiasLayer, MaskedLM
from tf_transformers.utils import tf_utils


class MaskedLMModel(LegacyLayer):
    def __init__(
        self,
        model,
        hidden_size,
        layer_norm_epsilon,
        activation='gelu',
        use_all_layers=False,
        is_training=False,
        use_dropout=False,
        use_extra_mlm_layer=True, # BERT has
        **kwargs,
    ):
        super(MaskedLMModel, self).__init__(is_training=is_training, use_dropout=use_dropout, name=model.name, **kwargs)

        self.model = model
        # We need keras layer to access embedding table
        if not isinstance(model, tf.keras.layers.Layer):
            raise TypeError(
                "We expect model to be a tf.keras.layers.Layer/LegacyLayer. \
                But you passed {}".format(
                    type(model)
                )
            )
        self.model_config = model._config_dict
        self.use_all_layers = use_all_layers
        self.use_extra_mlm_layer = use_extra_mlm_layer
        self._masked_lm_layer = MaskedLM(
            hidden_size=hidden_size,
            layer_norm_epsilon=layer_norm_epsilon,
            activation=activation,
            name="mlm",
        )
        self._masked_lm_bias = BiasLayer(name="mlm/transform")

        # Initialize model
        self.model_inputs, self.model_outputs = self.get_model(initialize_only=True)
        
    def _gather_indexes(self, sequence_tensor, positions):
        """Gathers the vectors at the specific positions, for performance.
        Args:
            sequence_tensor: Sequence output of shape
                (`batch_size`, `seq_length`, num_hidden) where num_hidden is number of
                hidden units.
            positions: Positions ids of tokens in sequence to mask for pretraining
                of with dimension (batch_size, num_predictions) where
                `num_predictions` is maximum number of tokens to mask out and predict
                per each sequence.
        Returns:
            Masked out sequence tensor of shape (batch_size * num_predictions,
            num_hidden).
        """
        sequence_shape = tf.shape(sequence_tensor)
        batch_size, seq_length = sequence_shape[0], sequence_shape[1]
        width = sequence_shape[2]

        flat_offsets = tf.reshape(tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
        flat_positions = tf.reshape(positions + flat_offsets, [-1])
        flat_sequence_tensor = tf.reshape(sequence_tensor, [batch_size * seq_length, width])
        output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
        # Make it 3D (b x s x h)
        output_tensor = tf.reshape(output_tensor, (batch_size, -1, sequence_tensor.shape[2]))
        return output_tensor

    def call(self, inputs):
        # Encoder outputs
        masked_lm_positions = inputs['masked_lm_positions']
        del inputs['masked_lm_positions']
        result = self.model(inputs)

        if self.use_all_layers:
            all_token_logits = []
            encoder_outputs = result["all_layer_token_embeddings"]
            for per_layer_token_embeddings in encoder_outputs:
                # token logits per layer
                if self.use_extra_mlm_layer:
                    layer_token_embeddings_mlm = self._masked_lm_layer(per_layer_token_embeddings, masked_lm_positions)
                else:
                    layer_token_embeddings_mlm = self._gather_indexes(per_layer_token_embeddings, masked_lm_positions)
                layer_token_logits = tf.matmul(
                    layer_token_embeddings_mlm,
                    tf.cast(self.get_embedding_table(), dtype=tf_utils.get_dtype()),
                    transpose_b=True,
                )
                layer_token_logits = self._masked_lm_bias(layer_token_logits)
                all_token_logits.append(layer_token_logits)

            result = {"token_embeddings": encoder_outputs, "token_logits": all_token_logits}

        else:
            token_embeddings = result['token_embeddings']
            if self.use_extra_mlm_layer:
                token_embeddings_mlm = self._masked_lm_layer(token_embeddings, masked_lm_positions)
            else:
                token_embeddings_mlm = self._gather_indexes(token_embeddings, masked_lm_positions)
            # MaskedLM layer only project it and normalize (b x s x h)
            token_logits = tf.matmul(
                token_embeddings_mlm,
                tf.cast(self.model.get_embedding_table(), dtype=tf_utils.get_dtype()),
                transpose_b=True,
            )
            token_logits = self._masked_lm_bias(token_logits)

            result = {"token_embeddings": token_embeddings, "token_logits": token_logits}
            return result

    def get_model(self, initialize_only=False):
        inputs = self.model.input
        masked_lm_positions = tf.keras.layers.Input(
            shape=(None,),
            batch_size=self.model_config["batch_size"],
            dtype=tf.int32,
            name="masked_lm_positions",
        )
        inputs['masked_lm_positions'] = masked_lm_positions
        inputs_copy = inputs.copy()  # We keep a copy to pass, otherwise checkpoint save will throw error
        layer_outputs = self(inputs)
        if initialize_only:
            return inputs_copy, layer_outputs

        model = LegacyModel(inputs=inputs, outputs=layer_outputs, name="mlm")
        model.model_config = self.model_config
        return model
