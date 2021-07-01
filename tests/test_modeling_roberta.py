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
"""Test ROBERTA Models"""

import unittest

import tensorflow as tf
from absl import logging
from transformers import RobertaTokenizer as Tokenizer

from tf_transformers.models import RobertaModel as Model

logging.get_absl_logger().name = "roberta_testing"

MODEL_NAME = 'roberta-base'


class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print("--------------------setUP--------------------------------------")
        self.model = Model.from_pretrained(MODEL_NAME)
        # self.model_ar  = Model.from_pretrained(MODEL_NAME, use_auto_regressive=True)
        self.tokenizer = Tokenizer.from_pretrained(MODEL_NAME)

    # @unittest.skip
    def test_tf_conversion(self):
        import shutil

        try:
            shutil.rmtree("/tmp/tf_transformers_cache/{}".format(MODEL_NAME))
        except:
            pass
        model = Model.from_pretrained(MODEL_NAME, convert_fn_type='tf')
        logging.info("Test: TF Conversion. ✅")

    #@unittest.skip
    def test_pt_conversion(self):
        import shutil

        try:
            shutil.rmtree("/tmp/tf_transformers_cache/{}".format(MODEL_NAME))
        except:
            pass
        model = Model.from_pretrained(MODEL_NAME, convert_fn_type='pt')
        logging.info("Test: PT Conversion. ✅")

    @unittest.skip("Bert has to be fine tuned for this")
    def test_auto_regressive(self):
        """Test Text Generation using Non Cache and Cached"""

        text = "Sachin Tendulkar is one of the finest"
        inputs_tf = self.tokenizer(text, return_tensors="tf")
        inputs = {}
        inputs["input_ids"] = inputs_tf["input_ids"]

        predictions_non_auto_regressive = []
        predictions_prob_non_auto_regressive = []

        for i in range(10):
            outputs = self.model(inputs)
            predicted_ids = tf.cast(tf.expand_dims(tf.argmax(outputs["last_token_logits"], axis=1), 1), tf.int32)
            inputs["input_ids"] = tf.concat([inputs["input_ids"], predicted_ids], axis=1)
            predictions_non_auto_regressive.append(predicted_ids)
            predictions_prob_non_auto_regressive.append(
                tf.expand_dims(tf.reduce_max(outputs["last_token_logits"], axis=1), 1)
            )
        predictions_non_auto_regressive = tf.concat(predictions_non_auto_regressive, axis=1)
        predictions_prob_non_auto_regressive = tf.concat(predictions_prob_non_auto_regressive, axis=1)

        # -------------------------------------------------------------------------------------------- # noqa
        # Cached
        inputs_tf = self.tokenizer(text, return_tensors="tf")
        inputs = {}
        inputs["input_ids"] = inputs_tf["input_ids"]

        seq_length = tf.shape(inputs["input_ids"])[1]
        batch_size = tf.shape(inputs["input_ids"])[0]

        inputs["all_cache_key"] = tf.zeros((12, batch_size, 12, seq_length, 64))
        inputs["all_cache_value"] = tf.zeros((12, batch_size, 12, seq_length, 64))
        inputs["past_length"] = tf.zeros(shape=(1, batch_size), dtype=tf.int32)
        predictions_auto_regressive = []
        predictions_prob_auto_regressive = []

        past_lengths = []
        for i in range(10):
            outputs = self.model_ar(inputs)
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
        tf.debugging.assert_near(predictions_prob_auto_regressive, predictions_prob_non_auto_regressive, rtol=1.0)
        tf.debugging.assert_equal(predictions_auto_regressive, predictions_non_auto_regressive)
        logging.info("Test: Successful Auto Regressive Encoder. ✅")

    @unittest.skip("Bert has to be fine tuned for this")
    def test_auto_regressive_batch(self):
        """Test Batch Text Generation Auto Regressive"""
        text = ['Sachin Tendulkar is one of the finest', 'I love stars because']
        # -1 is important
        input_ids = tf.ragged.constant(self.tokenizer(text)["input_ids"]).to_tensor(-1)
        inputs = {}
        inputs["input_ids"] = input_ids

        seq_length = tf.shape(inputs["input_ids"])[1]
        batch_size = tf.shape(inputs["input_ids"])[0]

        inputs["all_cache_key"] = tf.zeros((12, batch_size, 12, seq_length, 64))
        inputs["all_cache_value"] = tf.zeros((12, batch_size, 12, seq_length, 64))
        inputs["past_length"] = tf.zeros(shape=(1, batch_size), dtype=tf.int32)
        predictions_auto_regressive = []
        predictions_prob_auto_regressive = []

        past_lengths = []
        for i in range(10):
            outputs = self.model_ar(inputs)
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
        tf.debugging.assert_equal(predictions_auto_regressive.numpy().tolist(), expected_prediction)
        tf.debugging.assert_near(predictions_prob_auto_regressive.numpy().tolist(), expected_probs)
        logging.info("Test: Successful Batch Auto Regressive Encoder. ✅")

    @unittest.skip("Bert has to be fine tuned for this")
    def test_auto_regressive_saved_model(self):
        """Test Auto Regressive using Decoder Saved Model"""
        import shutil
        import tempfile

        from tf_transformers.text import TextDecoder

        text = ['Sachin Tendulkar is one of the finest', 'I love stars because']

        dirpath = tempfile.mkdtemp()
        saved_model_dir = "{}/model_pb".format(dirpath)
        self.model_ar.save_as_serialize_module(saved_model_dir, overwrite=True)

        # Load saved model .
        loaded = tf.saved_model.load(saved_model_dir)
        decoder = TextDecoder(model=loaded)

        # Pad -1
        input_ids = tf.ragged.constant(self.tokenizer(text)["input_ids"]).to_tensor(-1)
        inputs = {}
        inputs["input_ids"] = input_ids
        decoder_results = decoder.decode(inputs, mode="greedy", max_iterations=10, eos_id=-100)
        predicted_ids = decoder_results["predicted_ids"].numpy().tolist()
        expected_ids = [
            [[1938, 287, 262, 995, 13, 679, 318, 257, 845, 922]],
            [[484, 821, 523, 881, 517, 621, 655, 257, 3491, 13]],
        ]
        tf.debugging.assert_equal(predicted_ids, expected_ids)
        shutil.rmtree(dirpath)
        logging.info("Test: Successful Batch Auto Regressive Encoder Saved Model. ✅")

    @unittest.skip("Bert has to be fine tuned for this")
    def test_auto_regressive_keras_model(self):
        """Test Auto Regressive using Decoder Keras Model"""
        from tf_transformers.text import TextDecoder

        text = ['Sachin Tendulkar is one of the finest', 'I love stars because']

        decoder = TextDecoder(model=self.model_ar)

        # Pad -1
        input_ids = tf.ragged.constant(self.tokenizer(text)["input_ids"]).to_tensor(-1)
        inputs = {}
        inputs["input_ids"] = input_ids
        decoder_results = decoder.decode(inputs, mode="greedy", max_iterations=10, eos_id=-100)
        predicted_ids = decoder_results["predicted_ids"].numpy().tolist()
        expected_ids = [
            [[1938, 287, 262, 995, 13, 679, 318, 257, 845, 922]],
            [[484, 821, 523, 881, 517, 621, 655, 257, 3491, 13]],
        ]
        tf.debugging.assert_equal(predicted_ids, expected_ids)
        logging.info("Test: Successful Batch Auto Regressive Encoder Keras Model. ✅")

    # def test_auto_regressive_encoder_decoder():
    #     from transformers import GPT2Tokenizer

    #     tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    #     from tf_transformers.models import EncoderDecoder

    #     # Without Cache
    #     encoder_layer, encoder_config = GPT2Model.get_model(
    #         model_name=model_name, mask_mode="user_defined", return_layer=True
    #     )
    #     decoder_layer, decoder_config = GPT2Model.get_model(model_name=model_name, return_layer=True, use_decoder=True)

    #     # Decoder layer wont load from checkpoint
    #     # As the graph is different

    #     # Get decoder variables index and name as dict
    #     # Assign encoder weights to decoder wherever it matches variable name
    #     num_assigned = 0
    #     decoder_var = {var.name: index for index, var in enumerate(decoder_layer.variables)}
    #     for encoder_var in encoder_layer.variables:
    #         if encoder_var.name in decoder_var:
    #             index = decoder_var[encoder_var.name]
    #             decoder_layer.variables[index].assign(encoder_var)
    #             num_assigned += 1

    #     model = EncoderDecoder(encoder_layer, decoder_layer, share_embeddings=True, share_encoder=True)

    #     # Check encoder decoder generation without caching

    #     text = "Sachin Tendulkar is one of the finest"
    #     encoder_input_ids = tf.expand_dims(tf.ragged.constant(tokenizer(text)["input_ids"]), 0)
    #     encoder_input_mask = tf.ones_like(encoder_input_ids)
    #     decoder_input_ids = tf.constant([[1]])

    #     inputs = {}
    #     inputs["encoder_input_ids"] = encoder_input_ids
    #     inputs["encoder_input_mask"] = encoder_input_mask
    #     inputs["decoder_input_ids"] = decoder_input_ids

    #     predictions_non_auto_regressive = []
    #     predictions_prob_non_auto_regressive = []

    #     for i in range(10):
    #         outputs = model(inputs)
    #         predicted_ids = tf.cast(tf.expand_dims(tf.argmax(outputs["last_token_logits"], axis=1), 1), tf.int32)
    #         inputs["encoder_input_ids"] = tf.concat([inputs["encoder_input_ids"], predicted_ids], axis=1)
    #         inputs["encoder_input_mask"] = tf.ones_like(inputs["encoder_input_ids"])
    #         predictions_non_auto_regressive.append(predicted_ids)
    #         predictions_prob_non_auto_regressive.append(
    #             tf.expand_dims(tf.reduce_max(outputs["last_token_logits"], axis=1), 1)
    #         )
    #     predictions_non_auto_regressive = tf.concat(predictions_non_auto_regressive, axis=1)
    #     predictions_prob_non_auto_regressive = tf.concat(predictions_prob_non_auto_regressive, axis=1)

    #     # Cache

    #     encoder_layer, encoder_config = GPT2Model.get_model(
    #         model_name=model_name, mask_mode="user_defined", return_layer=True
    #     )
    #     decoder_layer, decoder_config = GPT2Model.get_model(
    #         model_name=model_name, return_layer=True, use_decoder=True, use_auto_regressive=True
    #     )

    #     # Decoder layer wont load from checkpoint
    #     # As the graph is different

    #     # Get decoder variables index and name as dict
    #     # Assign encoder weights to decoder wherever it matches variable name
    #     num_assigned = 0
    #     decoder_var = {var.name: index for index, var in enumerate(decoder_layer.variables)}
    #     for encoder_var in encoder_layer.variables:
    #         if encoder_var.name in decoder_var:
    #             index = decoder_var[encoder_var.name]
    #             decoder_layer.variables[index].assign(encoder_var)
    #             num_assigned += 1

    #     model = EncoderDecoder(encoder_layer, decoder_layer, share_embeddings=True, share_encoder=True)
    #     # Check encoder decoder generation  caching

    #     encoder_hidden_dim = encoder_config["embedding_size"]
    #     num_hidden_layers = decoder_config["num_hidden_layers"]
    #     num_attention_heads = decoder_config["num_attention_heads"]
    #     attention_head_size = decoder_config["attention_head_size"]

    #     text = "Sachin Tendulkar is one of the finest"
    #     encoder_input_ids = tf.expand_dims(tf.ragged.constant(tokenizer(text)["input_ids"]), 0)
    #     encoder_input_mask = tf.ones_like(encoder_input_ids)
    #     decoder_input_ids = tf.constant([[1]])

    #     batch_size = tf.shape(encoder_input_ids)[0]
    #     seq_length = tf.shape(encoder_input_ids)[1]

    #     encoder_hidden_states = tf.zeros((batch_size, seq_length, 768))
    #     decoder_all_cache_key = tf.zeros(
    #         (num_hidden_layers, batch_size, num_attention_heads, seq_length, attention_head_size)
    #     )
    #     decoder_all_cahce_value = tf.zeros(
    #         (num_hidden_layers, batch_size, num_attention_heads, seq_length, attention_head_size)
    #     )

    #     inputs = {}
    #     inputs["encoder_input_ids"] = encoder_input_ids
    #     inputs["encoder_input_mask"] = encoder_input_mask
    #     inputs["decoder_input_ids"] = decoder_input_ids
    #     inputs["encoder_hidden_states"] = encoder_hidden_states
    #     inputs["decoder_all_cache_key"] = decoder_all_cache_key
    #     inputs["decoder_all_cache_value"] = decoder_all_cahce_value

    #     predictions_auto_regressive = []
    #     predictions_prob_auto_regressive = []

    #     for i in range(10):
    #         outputs = model(inputs)
    #         predicted_ids = tf.cast(tf.expand_dims(tf.argmax(outputs["last_token_logits"], axis=1), 1), tf.int32)
    #         inputs["input_ids"] = predicted_ids
    #         inputs["decoder_all_cache_key"] = outputs["decoder_all_cache_key"]
    #         inputs["decoder_all_cache_value"] = outputs["decoder_all_cache_value"]
    #         inputs["encoder_hidden_states"] = outputs["encoder_hidden_states"]
    #         predictions_auto_regressive.append(predicted_ids)
    #         predictions_prob_auto_regressive.append(
    #             tf.expand_dims(tf.reduce_max(outputs["last_token_logits"], axis=1), 1)
    #         )
    #     predictions_auto_regressive = tf.concat(predictions_auto_regressive, axis=1)
    #     predictions_prob_auto_regressive = tf.concat(predictions_prob_auto_regressive, axis=1)

    #     tf.assert_equal(predictions_auto_regressive, predictions_non_auto_regressive)
    #     logging.info("Test: Successful Auto Regressive Encoder Decoder.")


if __name__ == '__main__':
    unittest.main()
