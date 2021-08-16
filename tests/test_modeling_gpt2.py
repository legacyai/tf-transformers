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
"""Test GPT2 Models"""

import unittest

import tensorflow as tf
import tempfile
import shutil
from absl import logging
from transformers import GPT2TokenizerFast as Tokenizer
from tf_transformers.text import TextDecoder, TextDecoderSerializable
from tf_transformers.models import GPT2Model as Model

logging.get_absl_logger().name = "gpt2_testing"

MODEL_NAME = 'gpt2'


class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print("--------------------setUP--------------------------------------")
        self.model = Model.from_pretrained(MODEL_NAME)
        self.model_ar = Model.from_pretrained(MODEL_NAME, use_auto_regressive=True)
        self.tokenizer = Tokenizer.from_pretrained(MODEL_NAME)

    # @unittest.skip
    def test_tf_conversion(self):
        import shutil

        try:
            shutil.rmtree("/tmp/tf_transformers_cache/{}".format(MODEL_NAME))
        except:
            pass
        _ = Model.from_pretrained(MODEL_NAME, convert_fn_type='tf')
        logging.info("Test: TF Conversion. ✅")

    # @unittest.skip
    def test_pt_conversion(self):
        import shutil

        try:
            shutil.rmtree("/tmp/tf_transformers_cache/{}".format(MODEL_NAME))
        except:
            pass
        _ = Model.from_pretrained(MODEL_NAME, convert_fn_type='pt')
        logging.info("Test: PT Conversion. ✅")

    # @unittest.skip
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

    # @unittest.skip
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
                masks = tf.reshape(masks, (1, batch_size, 1, seq_length, 1),)
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

    # @unittest.skip
    def test_auto_regressive_saved_model_greedy(self):
        """Test Auto Regressive using Decoder Saved Model"""
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
            [1938, 287, 262, 995, 13, 679, 318, 257, 845, 922],
            [484, 821, 523, 881, 517, 621, 655, 257, 3491, 13],
        ]
        tf.debugging.assert_equal(predicted_ids, expected_ids)
        shutil.rmtree(dirpath)
        logging.info("Test: Successful Auto Regressive Saved Model Greedy. ✅")

    # @unittest.skip
    def test_auto_regressive_keras_model_greedy(self):
        """Test Auto Regressive using Decoder Keras Model"""
        text = ['Sachin Tendulkar is one of the finest', 'I love stars because']

        decoder = TextDecoder(model=self.model_ar)

        # Pad -1
        input_ids = tf.ragged.constant(self.tokenizer(text)["input_ids"]).to_tensor(-1)
        inputs = {}
        inputs["input_ids"] = input_ids
        decoder_results = decoder.decode(inputs, mode="greedy", max_iterations=10, eos_id=-100)
        predicted_ids = decoder_results["predicted_ids"].numpy().tolist()
        expected_ids = [
            [1938, 287, 262, 995, 13, 679, 318, 257, 845, 922],
            [484, 821, 523, 881, 517, 621, 655, 257, 3491, 13],
        ]

        tf.debugging.assert_equal(predicted_ids, expected_ids)
        logging.info("Test: Successful Auto Regressive Keras Model Greedy. ✅")

    # @unittest.skip
    def test_auto_regressive_saved_model_beam(self):
        """Test Auto Regressive using Decoder Saved Model"""
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
        decoder_results = decoder.decode(inputs, mode="beam", num_beams=3, max_iterations=10, eos_id=-100)
        predicted_ids = decoder_results["predicted_ids"].numpy().tolist()
        print("Predicted ids beam", predicted_ids)
        expected_ids = [
            [[19553, 3653, 287, 262, 995, 13, 679, 318, 530, 286]],
            [[484, 821, 262, 691, 1517, 326, 6067, 284, 502, 13]],
        ]
        # tf.debugging.assert_equal(predicted_ids, expected_ids)
        shutil.rmtree(dirpath)
        logging.info("Test: Successful Auto Regressive Saved Model Beam. ✅")

    # @unittest.skip
    def test_auto_regressive_keras_model_beam(self):
        """Test Auto Regressive using Decoder Keras Model"""
        text = ['Sachin Tendulkar is one of the finest', 'I love stars because']

        decoder = TextDecoder(model=self.model_ar)

        # Pad -1
        input_ids = tf.ragged.constant(self.tokenizer(text)["input_ids"]).to_tensor(-1)
        inputs = {}
        inputs["input_ids"] = input_ids
        decoder_results = decoder.decode(inputs, mode="beam", num_beams=3, max_iterations=10, eos_id=-100)
        predicted_ids = decoder_results["predicted_ids"].numpy().tolist()
        expected_ids = [
            [[19553, 3653, 287, 262, 995, 13, 679, 318, 530, 286]],
            [[484, 821, 262, 691, 1517, 326, 6067, 284, 502, 13]],
        ]
        # tf.debugging.assert_equal(predicted_ids, expected_ids)
        logging.info("Test: Successful Auto Regressive Keras Model Greedy. ✅")

    @unittest.skip
    def test_auto_regressive_saved_model_top_k_top_p(self):
        """Test Auto Regressive using Decoder Saved Model"""
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
        decoder_results = decoder.decode(
            inputs, mode="top_k_top_p", top_k=100, top_p=0.7, max_iterations=10, eos_id=-100
        )
        _ = decoder_results["predicted_ids"].numpy().tolist()
        shutil.rmtree(dirpath)
        logging.info("Test: Successful Auto Regressive Saved Model top K top P. ✅")

    @unittest.skip
    def test_auto_regressive_keras_model_top_k_top_p(self):
        """Test Auto Regressive using Decoder Keras Model"""
        text = ['Sachin Tendulkar is one of the finest', 'I love stars because']

        decoder = TextDecoder(model=self.model_ar)

        # Pad -1
        input_ids = tf.ragged.constant(self.tokenizer(text)["input_ids"]).to_tensor(-1)
        inputs = {}
        inputs["input_ids"] = input_ids
        decoder_results = decoder.decode(
            inputs, mode="top_k_top_p", top_k=100, top_p=0.7, max_iterations=10, eos_id=-100
        )
        _ = decoder_results["predicted_ids"].numpy().tolist()
        logging.info("Test: Successful Auto Regressive Keras Model top k top P. ✅")

    @unittest.skip
    def test_tflite(self):
        """Test GPT2 Tflite"""
        model = Model.from_pretrained(model_name=MODEL_NAME, batch_size=1, sequence_length=32,)

        tempdir = tempfile.mkdtemp()
        model.save_serialized(tempdir, overwrite=True)

        converter = tf.lite.TFLiteConverter.from_saved_model("{}".format(tempdir))  # path to the SavedModel directory
        converter.experimental_new_converter = True

        tflite_model = converter.convert()
        open("{}/converted_model.tflite".format(tempdir), "wb").write(tflite_model)

        # Load the TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path="{}/converted_model.tflite".format(tempdir))
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Get result
        # encoder input_ids
        interpreter.set_tensor(
            input_details[0]['index'],
            tf.random.uniform(input_details[0]['shape'], minval=0, maxval=100, dtype=tf.int32),
        )
        interpreter.invoke()
        tflite_output = interpreter.get_tensor(output_details[-1]['index'])
        print("Tflite output shape", tflite_output.shape)
        # tf.debugging.assert_equal(tflite_output.shape, (1, 32, 32128))
        logging.info("Test: TFlite Conversion. ✅")
        shutil.rmtree(tempdir)


if __name__ == '__main__':
    unittest.main()
