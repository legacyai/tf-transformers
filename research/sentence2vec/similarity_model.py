import tensorflow as tf

from tf_transformers.core import LegacyLayer, LegacyModel
from tf_transformers.utils import tf_utils


def _large_compatible_negative(tensor_type):

    """Large negative number as Tensor.
    This function is necessary because the standard value for epsilon
    in this module (-1e9) cannot be represented using `tf.float16`.
    Args:
      tensor_type: a dtype to determine the type.
    Returns:
      a large negative number.
    """
    if tensor_type == tf.float16:
        return tf.float16.min
    return -1e9


class Similarity_Model_Pretraining(LegacyLayer):
    def __init__(
        self,
        encoder,
        projection_dimension,
        decoder=None,
        is_training=True,
        use_dropout=False,
        initializer="glorot_uniform",
        clip_logits=True,
        siamese=True,
        **kwargs,
    ):
        super(Similarity_Model_Pretraining, self).__init__(
            is_training=is_training, use_dropout=use_dropout, name=encoder.name, **kwargs
        )
        self.is_training = is_training
        if siamese:
            self.encoder = encoder
            self.decoder = encoder
        else:
            if decoder is None:
                raise ValueError("When siamese = False, decoder has to be provided. Provided decoder = None.")
            self.encoder = encoder
            self.decoder = decoder

        self.linear_projection = tf.keras.layers.Dense(
            units=projection_dimension,
            activation=None,
            kernel_initializer=initializer,
            name="linear_projection",
        )

        self.clip_logits = clip_logits
        if self.clip_logits:
            # As per CLIP paper
            self.logits_scale = tf.Variable(tf.math.log(1 / 0.07), name='logits_scale')

    def get_mean_embeddings(self, token_embeddings, input_mask):
        """ """
        # cls_embeddings = token_embeddings[:, 0, :]  # 0 is CLS (<s>)
        # mask PAD tokens
        token_emb_masked = token_embeddings * tf.cast(tf.expand_dims(input_mask, 2), tf.float32)
        total_non_padded_tokens_per_batch = tf.cast(tf.reduce_sum(input_mask, axis=1), tf.float32)
        # Convert to 2D
        total_non_padded_tokens_per_batch = tf.expand_dims(total_non_padded_tokens_per_batch, 1)
        mean_embeddings = tf.reduce_sum(token_emb_masked, axis=1) / total_non_padded_tokens_per_batch
        return mean_embeddings

    def call(self, inputs):
        """Call"""
        centre_inputs = {k.replace("centre_", ""): v for k, v in inputs.items() if k.startswith("centre_")}
        neighbour_inputs = {k.replace("neighbour_", ""): v for k, v in inputs.items() if k.startswith("neighbour_")}

        centre_outputs = self.encoder(centre_inputs)
        neighbour_outputs = self.decoder(neighbour_inputs)

        if 'cls_output' not in centre_outputs:
            centre_outputs['cls_output'] = tf.keras.layers.Lambda(lambda x: tf.squeeze(x[:, 0:1, :], axis=1))(
                centre_outputs['token_embeddings']
            )
        if 'cls_output' not in neighbour_outputs:
            neighbour_outputs['cls_output'] = tf.keras.layers.Lambda(lambda x: tf.squeeze(x[:, 0:1, :], axis=1))(
                neighbour_outputs['token_embeddings']
            )

        if self.clip_logits:
            centre_sentence_embedding = self.linear_projection(centre_outputs['cls_output'])
            neighbour_sentence_embedding = self.linear_projection(neighbour_outputs['cls_output'])

            centre_sentence_embedding_mean = self.linear_projection(
                self.get_mean_embeddings(centre_outputs['token_embeddings'], centre_inputs['input_mask'])
            )
            neighbour_sentence_embedding_mean = self.linear_projection(
                self.get_mean_embeddings(neighbour_outputs['token_embeddings'], neighbour_inputs['input_mask'])
            )

            centre_sentence_embedding_normalized = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(
                centre_sentence_embedding
            )
            neighbour_sentence_embedding_normalized = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(
                neighbour_sentence_embedding
            )

            centre_sentence_embedding_mean_normalized = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(
                centre_sentence_embedding_mean
            )
            neighbour_sentence_embedding_mean_normalized = tf.keras.layers.Lambda(
                lambda x: tf.nn.l2_normalize(x, axis=1)
            )(neighbour_sentence_embedding_mean)
        else:
            centre_sentence_embedding = centre_outputs['cls_output']
            neighbour_sentence_embedding = neighbour_outputs['cls_output']

            centre_sentence_embedding_mean = self.get_mean_embeddings(
                centre_outputs['token_embeddings'], centre_inputs['input_mask']
            )
            neighbour_sentence_embedding_mean = self.get_mean_embeddings(
                neighbour_outputs['token_embeddings'], neighbour_inputs['input_mask']
            )

            centre_sentence_embedding_normalized = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(
                centre_sentence_embedding
            )
            neighbour_sentence_embedding_normalized = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(
                neighbour_sentence_embedding
            )

            centre_sentence_embedding_mean_normalized = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(
                centre_sentence_embedding_mean
            )
            neighbour_sentence_embedding_mean_normalized = tf.keras.layers.Lambda(
                lambda x: tf.nn.l2_normalize(x, axis=1)
            )(neighbour_sentence_embedding_mean)

        if self.clip_logits:
            # Clamp logits to a max of tf.math.log(100) = 4.6051702 as per CLIP model
            logits_scale = tf.math.exp(self.logits_scale)
            logits_scale = tf.clip_by_value(
                logits_scale, clip_value_min=tf.math.log(1 / 0.07), clip_value_max=4.6051752
            )
            logits_scale = tf.cast(logits_scale, dtype=tf_utils.get_dtype())
        else:
            logits_scale = tf.cast(1.0, dtype=tf_utils.get_dtype())

        logits = tf.matmul(
            centre_sentence_embedding_normalized, neighbour_sentence_embedding_normalized, transpose_b=True
        )
        logits = logits_scale * logits

        logits_mean = tf.matmul(
            centre_sentence_embedding_mean_normalized, neighbour_sentence_embedding_mean_normalized, transpose_b=True
        )
        logits_mean = logits_scale * logits_mean

        scores = tf.matmul(centre_sentence_embedding_normalized, centre_sentence_embedding_normalized, transpose_b=True)
        scores_mask = tf.where(
            tf.equal(scores, tf.linalg.diag_part(scores)),
            _large_compatible_negative(scores.dtype),
            tf.cast(0.0, scores.dtype),
        )
        # Reset only diagonal entries back to 1.0
        scores_mask = tf.linalg.set_diag(
            scores_mask, tf.zeros(shape=(tf.shape(scores_mask)[0])), name='make_diagonal_one'
        )

        outputs = {}
        outputs['logits'] = logits + scores_mask
        outputs['logits_mean'] = logits_mean + scores_mask
        outputs['centre_sentence_embedding_normalized'] = centre_sentence_embedding_normalized
        outputs['neighbour_sentence_embedding_normalized'] = neighbour_sentence_embedding_normalized
        outputs['centre_sentence_embedding_mean_normalized'] = centre_sentence_embedding_mean_normalized
        outputs['neighbour_sentence_embedding_mean_normalized'] = neighbour_sentence_embedding_mean_normalized

        return outputs

    def get_model(self):
        if self.is_training:
            inputs = {}
            # Assume encoder and decoder have same input types
            for k, v in self.encoder.input.items():
                inputs["centre_" + k] = v
            for k, v in self.encoder.input.items():
                inputs["neighbour_" + k] = tf.keras.layers.Input(
                    shape=v.shape[1:], batch_size=v.shape[0], dtype=v.dtype, name=v.name.split(":")[0] + "_2"
                )
        else:
            inputs = self.model.input

        layer_output = self(inputs)
        model = LegacyModel(inputs=inputs, outputs=layer_output, name="similarity_model")
        try:
            model.model_config = self.encoder._config_dict
        except:
            model.model_config = self.encoder.model_config
        return model
