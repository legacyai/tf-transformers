import tensorflow as tf

from tf_transformers.core import LegacyLayer, LegacyModel
from tf_transformers.utils import tf_utils


class Similarity_Model_Pretraining(LegacyLayer):
    def __init__(
        self,
        encoder,
        projection_dimension,
        decoder=None,
        is_training=True,
        use_dropout=False,
        initializer="glorot_uniform",
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

        # As per CLIP paper
        self.logits_scale = tf.Variable(tf.math.log(1 / 0.07), name='logits_scale')

    def call(self, inputs):
        """Call"""
        if self.is_training:
            original_inputs = {k.replace("original_", ""): v for k, v in inputs.items() if k.startswith("original_")}
            corrupted_inputs = {k.replace("corrupted_", ""): v for k, v in inputs.items() if k.startswith("corrupted_")}

            original_outputs = self.encoder(original_inputs)
            corrupted_outputs = self.decoder(corrupted_inputs)

            if 'cls_output' not in original_outputs:
                original_outputs['cls_output'] = tf.keras.layers.Lambda(lambda x: tf.squeeze(x[:, 0:1, :], axis=1))(
                    original_outputs['token_embeddings']
                )
            if 'cls_output' not in corrupted_outputs:
                corrupted_outputs['cls_output'] = tf.keras.layers.Lambda(lambda x: tf.squeeze(x[:, 0:1, :], axis=1))(
                    corrupted_outputs['token_embeddings']
                )

            original_sentence_embedding = self.linear_projection(original_outputs['cls_output'])
            corrupted_sentence_embedding = self.linear_projection(corrupted_outputs['cls_output'])

            original_sentence_embedding_normalized = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(
                original_sentence_embedding
            )
            corrupted_sentence_embedding_normalized = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(
                corrupted_sentence_embedding
            )

            logits = tf.matmul(
                original_sentence_embedding_normalized, corrupted_sentence_embedding_normalized, transpose_b=True
            )

            # Clamp logits to a max of tf.math.log(100) = 4.6051702 as per CLIP model
            self.logits_scale = tf.math.exp(self.logits_scale)
            self.logits_scale = tf.clip_by_value(
                self.logits_scale, clip_value_min=tf.math.log(1 / 0.07), clip_value_max=4.6051752
            )
            logits = self.logits_scale * logits

            corrupted_outputs['logits'] = logits
        else:
            first_inputs = {k.replace("first_", ""): v for k, v in inputs.items() if k.startswith("first_")}
            second_inputs = {k.replace("second_", ""): v for k, v in inputs.items() if k.startswith("second_")}

            first_outputs = self.encoder(first_inputs)
            second_outputs = self.decoder(second_inputs)

            if 'cls_output' not in first_outputs:
                first_outputs['cls_output'] = tf.keras.layers.Lambda(lambda x: tf.squeeze(x[:, 0:1, :], axis=1))(
                    first_outputs['token_embeddings']
                )
            if 'cls_output' not in second_outputs:
                second_outputs['cls_output'] = tf.keras.layers.Lambda(lambda x: tf.squeeze(x[:, 0:1, :], axis=1))(
                    second_outputs['token_embeddings']
                )

            first_sentence_embedding = self.linear_projection(first_outputs['cls_output'])
            second_sentence_embedding = self.linear_projection(second_outputs['cls_output'])

            first_sentence_embedding_normalized = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(
                first_sentence_embedding
            )
            second_sentence_embedding_normalized = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(
                second_sentence_embedding
            )

            logits = tf.matmul(
                first_sentence_embedding_normalized, second_sentence_embedding_normalized, transpose_b=True
            )

            # Clamp logits to a max of tf.math.log(100) = 4.6051702 as per CLIP model
            self.logits_scale = tf.math.exp(self.logits_scale)
            # no need to clamp at testing
            # self.logits_scale = tf.clip_by_value(self.logits_scale,
            # clip_value_min=tf.math.log(1/0.07), clip_value_max=4.6051752)
            logits = tf.cast(self.logits_scale, dtype=tf_utils.get_dtype()) * logits

            outputs = {}
            outputs['first_sentence_embedding_normalized'] = first_sentence_embedding_normalized
            outputs['second_entence_embedding_normalized'] = second_sentence_embedding_normalized
            outputs['logits'] = logits

            return outputs

    def get_model(self):
        if self.is_training:
            inputs = {}
            # Assume encoder and decoder have same input types
            for k, v in self.encoder.input.items():
                inputs["original_" + k] = v
            for k, v in self.encoder.input.items():
                inputs["corrupted_" + k] = tf.keras.layers.Input(
                    shape=v.shape[1:], batch_size=v.shape[0], dtype=v.dtype, name=v.name.split(":")[0] + "_2"
                )
        else:
            inputs = {}
            # Assume encoder and decoder have same input types
            for k, v in self.encoder.input.items():
                inputs["first_" + k] = v
            for k, v in self.encoder.input.items():
                inputs["second_" + k] = tf.keras.layers.Input(
                    shape=v.shape[1:], batch_size=v.shape[0], dtype=v.dtype, name=v.name.split(":")[0] + "_3"
                )
        layer_output = self(inputs)
        model = LegacyModel(inputs=inputs, outputs=layer_output, name="similarity_model")
        try:
            model.model_config = self.encoder._config_dict
        except:
            model.model_config = self.encoder.model_config
        return model
