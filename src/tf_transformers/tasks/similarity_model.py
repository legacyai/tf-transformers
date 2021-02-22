import tensorflow as tf
from tf_transformers.core import LegacyModel, LegacyLayer


class Similarity_Model(LegacyLayer):
    def __init__(self, encoder, decoder=None, is_training=True, initializer="glorot_uniform", siamese=False, **kwargs):
        kwargs["is_training"] = is_training
        super(Similarity_Model, self).__init__(**kwargs)
        self.is_training = is_training
        if siamese:
            self.encoder = encoder
            self.decoder = encoder
        else:
            if decoder is None:
                raise ValueError("When siamese = False, decoder has to be provided. Provided decoder = None.")
            self.encoder = encoder
            self.decoder = decoder

    def get_mean_embeddings(self, token_embeddings, input_mask):
        """"""
        cls_embeddings = token_embeddings[:, 0, :]  # 0 is CLS (<s>)
        # mask PAD tokens
        token_emb_masked = token_embeddings * tf.cast(tf.expand_dims(input_mask, 2), tf.float32)
        total_non_padded_tokens_per_batch = tf.cast(tf.reduce_sum(input_mask, axis=1), tf.float32)
        # Convert to 2D
        total_non_padded_tokens_per_batch = tf.expand_dims(total_non_padded_tokens_per_batch, 1)
        mean_embeddings = tf.reduce_sum(token_emb_masked, axis=1) / total_non_padded_tokens_per_batch
        return mean_embeddings

    def call(self, inputs):
        """
        inputs:
        """
        # Positive negative pairs
        if self.is_training:
            question_inputs = {k.replace("question_", ""): v for k, v in inputs.items() if k.startswith("question_")}
            positive_inputs = {k.replace("positive_", ""): v for k, v in inputs.items() if k.startswith("positive_")}
            negative_inputs = {k.replace("negative_", ""): v for k, v in inputs.items() if k.startswith("negative_")}
            question_results = self.encoder(question_inputs)
            positive_results = self.decoder(positive_inputs)
            negative_results = self.decoder(negative_inputs)

            question_cls = question_results["cls_output"]
            positive_cls = positive_results["cls_output"]
            negative_cls = negative_results["cls_output"]

            question_mean_embeddings = self.get_mean_embeddings(
                question_results["token_embeddings"], question_inputs["input_mask"]
            )
            positive_mean_embeddings = self.get_mean_embeddings(
                positive_results["token_embeddings"], positive_inputs["input_mask"]
            )
            negative_mean_embeddings = self.get_mean_embeddings(
                negative_results["token_embeddings"], negative_inputs["input_mask"]
            )

            results = {
                "question_cls": question_cls,
                "positive_cls": positive_cls,
                "negative_cls": negative_cls,
                "question_mean_embeddings": question_mean_embeddings,
                "positive_mean_embeddings": positive_mean_embeddings,
                "negative_mean_embeddings": negative_mean_embeddings,
            }

            positive_cls_batch = tf.matmul(results["question_cls"], results["positive_cls"], transpose_b=True)
            negative_cls_batch = tf.matmul(results["question_cls"], results["negative_cls"], transpose_b=True)

            positive_mean_batch = tf.matmul(
                results["question_mean_embeddings"], results["positive_mean_embeddings"], transpose_b=True
            )
            negative_mean_batch = tf.matmul(
                results["question_mean_embeddings"], results["negative_mean_embeddings"], transpose_b=True
            )

            results = {
                "positive_cls_batch": positive_cls_batch,
                "negative_cls_batch": negative_cls_batch,
                "positive_mean_batch": positive_mean_batch,
                "negative_mean_batch": negative_mean_batch,
            }

        else:
            # Normal model inputs
            results = self.encoder(inputs)
            results_cls = results["cls_output"]
            results_cls_normalized = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(results_cls)

            results_mean_embeddings = self.get_mean_embeddings(results["token_embeddings"], inputs["input_mask"])

            results_mean_embeddings_normalized = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(
                results_mean_embeddings
            )

            results = {
                "results_cls": results_cls,
                "results_cls_normalized": results_cls_normalized,
                "results_mean_embeddings": results_mean_embeddings,
                "results_mean_embeddings_normalized": results_mean_embeddings_normalized,
            }
        return results

    def get_model(self):
        if self.is_training:
            inputs = {}
            # Assume encoder and decoder have same input types
            for k, v in self.encoder.input.items():
                inputs["question_" + k] = v
            for k, v in self.encoder.input.items():
                inputs["positive_" + k] = tf.keras.layers.Input(
                    shape=v.shape[1:], batch_size=v.shape[0], dtype=v.dtype, name=v.name.split(":")[0] + "_2"
                )
            for k, v in self.encoder.input.items():
                inputs["negative_" + k] = tf.keras.layers.Input(
                    shape=v.shape[1:], batch_size=v.shape[0], dtype=v.dtype, name=v.name.split(":")[0] + "_3"
                )

        else:
            inputs = self.encoder.input

        layer_output = self(inputs)
        model = LegacyModel(inputs=inputs, outputs=layer_output, name="similarity_model")
        model.model_config = self.encoder.model_config
        return model
