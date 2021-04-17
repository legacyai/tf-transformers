import pytest
import tensorflow as tf

from tf_transformers.models import BertModel

from absl import flags
from absl import app
from absl import logging

logging.set_verbosity("INFO")

flags.DEFINE_string("model_name", None, "Name of the model")
FLAGS = flags.FLAGS


def test_main(model_name):
    def test_hf_to_tf_conversion():
        """Test Model Conversion"""
        model, config = BertModel.get_model(model_name=model_name)
        logging.info("Test Successful: BERT `{}` conversion.".format(model_name))

    def test_auto_regressive_encoder():
        """Test Encoder Auto Regressive"""
        model, config = BertModel.get_model(model_name=model_name)
        input_ids = tf.constant([[1, 2, 3], [800, 900, 127]])
        input_mask = tf.ones_like(input_ids)
        input_type_ids = tf.zeros_like(input_ids)

        inputs = {}
        inputs["input_ids"] = input_ids
        inputs["input_mask"] = input_mask
        inputs["input_type_ids"] = input_type_ids

        predictions_non_auto_regressive = []
        predictions_prob_non_auto_regressive = []

        for i in range(10):
            outputs = model(inputs)
            predicted_ids = tf.cast(tf.expand_dims(tf.argmax(outputs["last_token_logits"], axis=1), 1), tf.int32)
            inputs["input_ids"] = tf.concat([inputs["input_ids"], predicted_ids], axis=1)
            inputs["input_type_ids"] = tf.zeros_like(inputs["input_ids"])
            inputs["input_mask"] = tf.ones_like(inputs["input_ids"])
            predictions_non_auto_regressive.append(predicted_ids)
            predictions_prob_non_auto_regressive.append(
                tf.expand_dims(tf.reduce_max(outputs["last_token_logits"], axis=1), 1)
            )
        predictions_non_auto_regressive = tf.concat(predictions_non_auto_regressive, axis=1)
        predictions_prob_non_auto_regressive = tf.concat(predictions_prob_non_auto_regressive, axis=1)

        model, config = BertModel.get_model(model_name="bert-base-uncased", use_auto_regressive=True)
        input_ids = tf.constant([[1, 2, 3], [800, 900, 127]])
        input_mask = tf.ones_like(input_ids)
        input_type_ids = tf.zeros_like(input_ids)

        inputs = {}
        inputs["input_ids"] = input_ids
        inputs["input_mask"] = input_mask
        inputs["input_type_ids"] = input_type_ids

        seq_length = tf.shape(input_ids)[1]
        batch_size = tf.shape(input_ids)[0]

        inputs["all_cache_key"] = tf.zeros((12, batch_size, 12, seq_length, 64))
        inputs["all_cache_value"] = tf.zeros((12, batch_size, 12, seq_length, 64))
        inputs["past_length"] = tf.zeros(shape=(1, batch_size), dtype=tf.int32)
        predictions_auto_regressive = []
        predictions_prob_auto_regressive = []

        past_lengths = []
        for i in range(10):
            outputs = model(inputs)
            predicted_ids = tf.cast(tf.expand_dims(tf.argmax(outputs["last_token_logits"], axis=1), 1), tf.int32)
            inputs["input_ids"] = predicted_ids
            inputs["input_type_ids"] = tf.zeros_like(inputs["input_ids"])
            inputs["input_mask"] = tf.ones_like(inputs["input_ids"])
            inputs["all_cache_key"] = outputs["all_cache_key"]
            inputs["all_cache_value"] = outputs["all_cache_value"]
            inputs["past_length"] = outputs["past_length"]
            past_lengths.append(inputs["past_length"])
            predictions_auto_regressive.append(predicted_ids)
            predictions_prob_auto_regressive.append(
                tf.expand_dims(tf.reduce_max(outputs["last_token_logits"], axis=1), 1)
            )
        predictions_auto_regressive = tf.concat(predictions_auto_regressive, axis=1)
        predictions_prob_auto_regressive = tf.concat(predictions_prob_auto_regressive, axis=1)

        # Assert predictions
        tf.assert_equal(predictions_auto_regressive, predictions_non_auto_regressive)
        logging.info("Test: Successful Auto Regressive Encoder.")

    def test_auto_regressive_decoder_non_shared():
        pass

    def test_auto_regressive_shared():
        pass

    test_hf_to_tf_conversion()


def main(_):
    test_main(FLAGS.model_name)


if __name__ == "__main__":
    flags.mark_flag_as_required("model_name")
    app.run(main)
