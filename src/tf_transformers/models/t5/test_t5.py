import pytest
import tensorflow as tf

from tf_transformers.models import GPT2Model

from absl import flags
from absl import app
from absl import logging

logging.set_verbosity("INFO")

flags.DEFINE_string("model_name", None, "Name of the model")
FLAGS = flags.FLAGS


def test_main(model_name):
    def test_hf_to_tf_conversion():
        """Test Model Conversion"""
        model, config = GPT2Model.get_model(model_name=model_name)
        logging.info("Test Successful: BERT `{}` conversion.".format(model_name))

    def test_auto_regressive_encoder():
        """Test Encoder Auto Regressive"""

        from transformers import GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        model, conifg = GPT2Model.get_model(model_name=model_name)
        text = "Sachin Tendulkar is one of the finest"
        inputs_tf = tokenizer(text, return_tensors="tf")
        inputs = {}
        inputs["input_ids"] = inputs_tf["input_ids"]

        predictions_non_auto_regressive = []
        predictions_prob_non_auto_regressive = []

        for i in range(10):
            outputs = model(inputs)
            predicted_ids = tf.cast(tf.expand_dims(tf.argmax(outputs["last_token_logits"], axis=1), 1), tf.int32)
            inputs["input_ids"] = tf.concat([inputs["input_ids"], predicted_ids], axis=1)
            predictions_non_auto_regressive.append(predicted_ids)
            predictions_prob_non_auto_regressive.append(
                tf.expand_dims(tf.reduce_max(outputs["last_token_logits"], axis=1), 1)
            )
        predictions_non_auto_regressive = tf.concat(predictions_non_auto_regressive, axis=1)
        predictions_prob_non_auto_regressive = tf.concat(predictions_prob_non_auto_regressive, axis=1)

        # Cached
        model, conifg = GPT2Model.get_model(model_name=model_name, use_auto_regressive=True)
        text = "Sachin Tendulkar is one of the finest"
        inputs_tf = tokenizer(text, return_tensors="tf")
        inputs = {}
        inputs["input_ids"] = inputs_tf["input_ids"]

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

    def test_auto_regressive_encoder_variable_batch():
        # Batch predictions (var len)
        from transformers import GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model, config = GPT2Model.get_model(model_name=model_name, use_auto_regressive=True)
        input_ids = tf.ragged.constant(tokenizer(text)["input_ids"]).to_tensor(-1)
        inputs = {}
        inputs["input_ids"] = input_ids

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

            if i == 0:
                masks = tf.cast(tf.not_equal(input_ids, -1), tf.float32)
                masks = tf.reshape(
                    masks,
                    (1, batch_size, 1, seq_length, 1),
                )
                outputs["all_cache_key"] = outputs["all_cache_key"] * masks
                outputs["all_cache_value"] = outputs["all_cache_value"] * masks

            inputs["input_ids"] = predicted_ids
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
        expected_prediction = [
            [1938, 287, 262, 995, 13, 679, 318, 257, 845, 922],
            [484, 821, 523, 881, 517, 621, 655, 257, 3491, 13],
        ]
        expected_probs = [
            [
                -110.00343322753906,
                -84.10372161865234,
                -60.758541107177734,
                -94.87692260742188,
                -72.66572570800781,
                -124.67924499511719,
                -100.1087417602539,
                -103.07884216308594,
                -108.038330078125,
                -108.75567626953125,
            ],
            [
                -92.4664535522461,
                -122.232177734375,
                -114.12687683105469,
                -110.21340942382812,
                -106.74520111083984,
                -108.79459381103516,
                -89.76094055175781,
                -84.4063720703125,
                -102.25302124023438,
                -78.72990417480469,
            ],
        ]
        np.allclose(predictions_auto_regressive.numpy().tolist(), expected_prediction)
        logging.info("Test: Successful Auto Regressive Variable Batch.")

    def test_auto_regressive_generalizable():
        """Test Auto Regressive using Decoder"""
        import tempfile
        import shutil
        from transformers import GPT2Tokenizer
        from tf_transformers.text import TextDecoder

        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        dirpath = tempfile.mkdtemp()
        saved_model_dir = "{}/model_pb".format(dirpath)
        model, config = GPT2Model.get_model(model_name=model_name, use_auto_regressive=True)
        model.save_as_serialize_module(saved_model_dir, overwrite=True)

        loaded = tf.saved_model.load(saved_model_dir)
        decoder = TextDecoder(model=loaded)

        input_ids = tf.ragged.constant(tokenizer(text)["input_ids"]).to_tensor(-1)
        inputs = {}
        inputs["input_ids"] = input_ids
        decoder_results = decoder.decode(inputs, mode="greedy", max_iterations=10, eos_id=-100)
        predcited_ids = decoder_results["predicted_ids"].numpy().tolist()
        expected_ids = [
            [[1938, 287, 262, 995, 13, 679, 318, 257, 845, 922]],
            [[484, 821, 523, 881, 517, 621, 655, 257, 3491, 13]],
        ]

        np.allclose(predicted_ids, expected_ids)
        logging.info("Test: Successful Auto Regressive TextDecoder.")

    def test_auto_regressive_serializable():
        """Test Auto Regressive using Decoder"""
        import tempfile
        import shutil
        from transformers import GPT2Tokenizer
        from tf_transformers.text import TextDecoderSerializable

        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        dirpath = tempfile.mkdtemp()
        saved_model_dir = "{}/model_pb".format(dirpath)
        model, config = GPT2Model.get_model(model_name=model_name, use_auto_regressive=True)
        decoder_layer = TextDecoderSerializable(
            model=model, max_iterations=10, mode="greedy", do_sample=False, eos_id=-100
        )
        decoder_model = decoder_layer.get_model()
        decoder_model.save_serialized(saved_model_dir, overwrite=True)

        loaded_decoder = tf.saved_model.load(saved_model_dir)
        model_pb_decoder = loaded_decoder.signatures["serving_default"]

        text = ["Sachin Tendulkar is one of the finest", "I love stars because"]

        input_ids = tf.ragged.constant(tokenizer(text)["input_ids"]).to_tensor(-1)
        inputs = {}
        inputs["input_ids"] = input_ids
        decoder_results_serialized = model_pb_decoder(**inputs)
        predcited_ids = decoder_results_serialized["predicted_ids"].numpy().tolist()
        expected_ids = [
            [[1938, 287, 262, 995, 13, 679, 318, 257, 845, 922]],
            [[484, 821, 523, 881, 517, 621, 655, 257, 3491, 13]],
        ]

        np.allclose(predicted_ids, expected_ids)
        logging.info("Test: Successful Auto Regressive TextDecoderSerializable.")

    def test_auto_regressive_encoder_decoder():
        from transformers import GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        from tf_transformers.models import EncoderDecoder

        # Without Cache
        encoder_layer, encoder_config = GPT2Model.get_model(
            model_name=model_name, mask_mode="user_defined", return_layer=True
        )
        decoder_layer, decoder_config = GPT2Model.get_model(model_name=model_name, return_layer=True, use_decoder=True)

        # Decoder layer wont load from checkpoint
        # As the graph is different

        # Get decoder variables index and name as dict
        # Assign encoder weights to decoder wherever it matches variable name
        num_assigned = 0
        decoder_var = {var.name: index for index, var in enumerate(decoder_layer.variables)}
        for encoder_var in encoder_layer.variables:
            if encoder_var.name in decoder_var:
                index = decoder_var[encoder_var.name]
                decoder_layer.variables[index].assign(encoder_var)
                num_assigned += 1

        model = EncoderDecoder(encoder_layer, decoder_layer, share_embeddings=True, share_encoder=True)

        # Check encoder decoder generation without caching

        text = "Sachin Tendulkar is one of the finest"
        encoder_input_ids = tf.expand_dims(tf.ragged.constant(tokenizer(text)["input_ids"]), 0)
        encoder_input_mask = tf.ones_like(encoder_input_ids)
        decoder_input_ids = tf.constant([[1]])

        inputs = {}
        inputs["encoder_input_ids"] = encoder_input_ids
        inputs["encoder_input_mask"] = encoder_input_mask
        inputs["decoder_input_ids"] = decoder_input_ids

        predictions_non_auto_regressive = []
        predictions_prob_non_auto_regressive = []

        for i in range(10):
            outputs = model(inputs)
            predicted_ids = tf.cast(tf.expand_dims(tf.argmax(outputs["last_token_logits"], axis=1), 1), tf.int32)
            inputs["encoder_input_ids"] = tf.concat([inputs["encoder_input_ids"], predicted_ids], axis=1)
            inputs["encoder_input_mask"] = tf.ones_like(inputs["encoder_input_ids"])
            predictions_non_auto_regressive.append(predicted_ids)
            predictions_prob_non_auto_regressive.append(
                tf.expand_dims(tf.reduce_max(outputs["last_token_logits"], axis=1), 1)
            )
        predictions_non_auto_regressive = tf.concat(predictions_non_auto_regressive, axis=1)
        predictions_prob_non_auto_regressive = tf.concat(predictions_prob_non_auto_regressive, axis=1)

        # Cache

        encoder_layer, encoder_config = GPT2Model.get_model(
            model_name=model_name, mask_mode="user_defined", return_layer=True
        )
        decoder_layer, decoder_config = GPT2Model.get_model(
            model_name=model_name, return_layer=True, use_decoder=True, use_auto_regressive=True
        )

        # Decoder layer wont load from checkpoint
        # As the graph is different

        # Get decoder variables index and name as dict
        # Assign encoder weights to decoder wherever it matches variable name
        num_assigned = 0
        decoder_var = {var.name: index for index, var in enumerate(decoder_layer.variables)}
        for encoder_var in encoder_layer.variables:
            if encoder_var.name in decoder_var:
                index = decoder_var[encoder_var.name]
                decoder_layer.variables[index].assign(encoder_var)
                num_assigned += 1

        model = EncoderDecoder(encoder_layer, decoder_layer, share_embeddings=True, share_encoder=True)
        # Check encoder decoder generation  caching

        encoder_hidden_dim = encoder_config["embedding_size"]
        num_hidden_layers = decoder_config["num_hidden_layers"]
        num_attention_heads = decoder_config["num_attention_heads"]
        attention_head_size = decoder_config["attention_head_size"]

        text = "Sachin Tendulkar is one of the finest"
        encoder_input_ids = tf.expand_dims(tf.ragged.constant(tokenizer(text)["input_ids"]), 0)
        encoder_input_mask = tf.ones_like(encoder_input_ids)
        decoder_input_ids = tf.constant([[1]])

        batch_size = tf.shape(encoder_input_ids)[0]
        seq_length = tf.shape(encoder_input_ids)[1]

        encoder_hidden_states = tf.zeros((batch_size, seq_length, 768))
        decoder_all_cache_key = tf.zeros(
            (num_hidden_layers, batch_size, num_attention_heads, seq_length, attention_head_size)
        )
        decoder_all_cahce_value = tf.zeros(
            (num_hidden_layers, batch_size, num_attention_heads, seq_length, attention_head_size)
        )

        inputs = {}
        inputs["encoder_input_ids"] = encoder_input_ids
        inputs["encoder_input_mask"] = encoder_input_mask
        inputs["decoder_input_ids"] = decoder_input_ids
        inputs["encoder_hidden_states"] = encoder_hidden_states
        inputs["decoder_all_cache_key"] = decoder_all_cache_key
        inputs["decoder_all_cache_value"] = decoder_all_cahce_value

        predictions_auto_regressive = []
        predictions_prob_auto_regressive = []

        for i in range(10):
            outputs = model(inputs)
            predicted_ids = tf.cast(tf.expand_dims(tf.argmax(outputs["last_token_logits"], axis=1), 1), tf.int32)
            inputs["input_ids"] = predicted_ids
            inputs["decoder_all_cache_key"] = outputs["decoder_all_cache_key"]
            inputs["decoder_all_cache_value"] = outputs["decoder_all_cache_value"]
            inputs["encoder_hidden_states"] = outputs["encoder_hidden_states"]
            predictions_auto_regressive.append(predicted_ids)
            predictions_prob_auto_regressive.append(
                tf.expand_dims(tf.reduce_max(outputs["last_token_logits"], axis=1), 1)
            )
        predictions_auto_regressive = tf.concat(predictions_auto_regressive, axis=1)
        predictions_prob_auto_regressive = tf.concat(predictions_prob_auto_regressive, axis=1)

        tf.assert_equal(predictions_auto_regressive, predictions_non_auto_regressive)
        logging.info("Test: Successful Auto Regressive Encoder Decoder.")
        pass

    def test_auto_regressive_shared():
        pass

    test_hf_to_tf_conversion()
    # test_auto_regressive_encoder


def main(_):
    test_main(FLAGS.model_name)


if __name__ == "__main__":
    flags.mark_flag_as_required("model_name")
    app.run(main)
