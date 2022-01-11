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
"""Test t5 Models"""

import os
import unittest
import tensorflow as tf


import shutil
import tempfile
from absl import logging
from transformers import MT5Tokenizer as Tokenizer

from tf_transformers.models import mT5Model as Model
from tf_transformers.text import TextDecoder, TextDecoderSerializable

logging.get_absl_logger().name = "mt5_testing"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["PYTHONWARNINGS"] = "ignore"

MODEL_NAME = 'google/mt5-small'
DECODER_START_ID = 0
DECODER_EOS_ID = 1


class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print("--------------------setUP--------------------------------------")
        self.model = Model.from_pretrained(MODEL_NAME)
        self.model_ar, self.config = Model.from_pretrained(
            MODEL_NAME, decoder_kwargs={'use_auto_regressive': True}, use_auto_regressive=True, return_config=True
        )
        self.tokenizer = Tokenizer.from_pretrained(MODEL_NAME)

    @unittest.skip
    def test_tf_conversion(self):

        try:
            shutil.rmtree("/tmp/tf_transformers_cache/{}".format(MODEL_NAME))
        except:
            pass
        _ = Model.from_pretrained(MODEL_NAME, convert_fn_type='tf')
        logging.info("Test: TF Conversion. ✅")

    @unittest.skip
    def test_pt_conversion(self):
        try:
            shutil.rmtree("/tmp/tf_transformers_cache/{}".format(MODEL_NAME))
        except:
            pass
        _ = Model.from_pretrained(MODEL_NAME, convert_fn_type='pt')
        logging.info("Test: PT Conversion. ✅")

    @unittest.skip
    def test_auto_regressive(self):
        """Test Text Generation using Non Cache and Cached"""

        # T5 text generation without caching
        text = "translate English to German: The house is wonderful and we wish to be here :)"

        # Create inputs
        inputs_hf = self.tokenizer(text, return_tensors='tf')
        inputs = {}
        inputs['encoder_input_ids'] = inputs_hf['input_ids']
        inputs['encoder_input_mask'] = inputs_hf['attention_mask']
        inputs['decoder_input_ids'] = tf.constant([[DECODER_START_ID]])

        # Iterate
        predictions_non_auto_regressive = []
        predictions_prob_non_auto_regressive = []

        for i in range(13):
            outputs = self.model(inputs)
            predicted_ids = tf.cast(tf.expand_dims(tf.argmax(outputs["last_token_logits"], axis=1), 1), tf.int32)
            inputs["decoder_input_ids"] = tf.concat([inputs["decoder_input_ids"], predicted_ids], axis=1)
            predictions_non_auto_regressive.append(predicted_ids)
            predictions_prob_non_auto_regressive.append(
                tf.expand_dims(tf.reduce_max(outputs["last_token_logits"], axis=1), 1)
            )
        predictions_non_auto_regressive = tf.concat(predictions_non_auto_regressive, axis=1)
        predictions_prob_non_auto_regressive = tf.concat(predictions_prob_non_auto_regressive, axis=1)

        # -------------------------------------------------------------------------------------------------------------#
        # Text generation with cache
        encoder_hidden_dim = self.config['embedding_size']
        num_hidden_layers = self.config['num_hidden_layers']
        num_attention_heads = self.config['num_attention_heads']
        attention_head_size = self.config['attention_head_size']

        # Inputs
        inputs_hf = self.tokenizer(text, return_tensors='tf')
        encoder_input_ids = inputs_hf['input_ids']
        encoder_input_mask = inputs_hf['attention_mask']

        batch_size = tf.shape(encoder_input_ids)[0]
        seq_length = tf.shape(encoder_input_ids)[1]

        decoder_input_ids = tf.reshape([0] * batch_size.numpy(), (batch_size, 1))

        encoder_hidden_states = tf.zeros((batch_size, seq_length, encoder_hidden_dim))
        decoder_all_cache_key = tf.zeros(
            (num_hidden_layers, batch_size, num_attention_heads, seq_length, attention_head_size)
        )
        decoder_all_cahce_value = tf.zeros(
            (num_hidden_layers, batch_size, num_attention_heads, seq_length, attention_head_size)
        )

        inputs = {}
        inputs['encoder_input_ids'] = encoder_input_ids
        inputs['encoder_input_mask'] = encoder_input_mask
        inputs['decoder_input_ids'] = decoder_input_ids
        inputs['encoder_hidden_states'] = encoder_hidden_states
        inputs['decoder_all_cache_key'] = decoder_all_cache_key
        inputs['decoder_all_cache_value'] = decoder_all_cahce_value

        # Iterate
        predictions_auto_regressive = []
        predictions_prob_auto_regressive = []

        for i in range(13):
            outputs = self.model_ar(inputs)
            predicted_ids = tf.cast(tf.expand_dims(tf.argmax(outputs["last_token_logits"], axis=1), 1), tf.int32)
            inputs["decoder_input_ids"] = predicted_ids
            inputs["decoder_all_cache_key"] = outputs["decoder_all_cache_key"]
            inputs["decoder_all_cache_value"] = outputs["decoder_all_cache_value"]
            inputs["encoder_hidden_states"] = outputs["encoder_hidden_states"]
            predictions_auto_regressive.append(predicted_ids)
            predictions_prob_auto_regressive.append(
                tf.expand_dims(tf.reduce_max(outputs["last_token_logits"], axis=1), 1)
            )
        predictions_auto_regressive = tf.concat(predictions_auto_regressive, axis=1)
        predictions_prob_auto_regressive = tf.concat(predictions_prob_auto_regressive, axis=1)

        # ----------------------------------------------------------------------------------------#
        expected_outputs = [[644, 4598, 229, 19250, 64, 558, 7805, 1382, 1110, 3, 10, 61, 1]]
        tf.assert_equal(predictions_non_auto_regressive, predictions_auto_regressive, expected_outputs)
        tf.debugging.assert_near(predictions_non_auto_regressive, predictions_auto_regressive, rtol=1.0)
        logging.info("Test: Successful Auto Regressive Encoder Decoder. ✅")

    @unittest.skip
    def test_auto_regressive_saved_model_greedy(self):
        """Test Auto Regressive using Decoder Saved Model - Greedy"""
        # Text generation using saved_model with TextDecoder

        text = "translate English to German: The house is wonderful and we wish to be here :)"

        # Save as saved model
        saved_model_dir = tempfile.mkdtemp()
        self.model_ar.save_as_serialize_module(saved_model_dir, overwrite=True)

        # Load saved model
        loaded = tf.saved_model.load(saved_model_dir)
        decoder = TextDecoder(model=loaded, decoder_start_token_id=DECODER_START_ID)  # for t5

        # Inputs
        inputs_hf = self.tokenizer(text, return_tensors='tf')
        inputs = {}
        inputs['encoder_input_ids'] = inputs_hf['input_ids']
        inputs['encoder_input_mask'] = inputs_hf['attention_mask']

        decoder_results = decoder.decode(inputs, mode='greedy', max_iterations=13, eos_id=1)

        expected_outputs = [[644, 4598, 229, 19250, 64, 558, 7805, 1382, 1110, 3, 10, 61, 1]]
        print("Decoder ", decoder_results['predicted_ids'].numpy().tolist()[0])
        assert decoder_results['predicted_ids'].numpy().tolist()[0] == expected_outputs
        logging.info("Test: Successful Saved Model Greedy. ✅")
        shutil.rmtree(saved_model_dir)

    @unittest.skip
    def test_auto_regressive_keras_model_greedy(self):
        """Test Auto Regressive using Decoder Keras Model - Greedy"""
        # Text generation using saved_model with TextDecoder

        from tf_transformers.text import TextDecoder

        text = "translate English to German: The house is wonderful and we wish to be here :)"

        decoder = TextDecoder(model=self.model_ar, decoder_start_token_id=DECODER_START_ID)  # for t5

        # Inputs
        inputs_hf = self.tokenizer(text, return_tensors='tf')
        inputs = {}
        inputs['encoder_input_ids'] = inputs_hf['input_ids']
        inputs['encoder_input_mask'] = inputs_hf['attention_mask']

        decoder_results = decoder.decode(inputs, mode='greedy', max_iterations=13, eos_id=1)

        expected_outputs = [[644, 4598, 229, 19250, 64, 558, 7805, 1382, 1110, 3, 10, 61, 1]]

        print("Decoder ", decoder_results['predicted_ids'].numpy().tolist()[0])

        assert decoder_results['predicted_ids'].numpy().tolist()[0] == expected_outputs
        logging.info("Test: Successful Keras Model Greedy. ✅")

    # @unittest.skip
    def test_auto_regressive_saved_model_beam(self):
        """Test Auto Regressive using Decoder Saved Model - Beam"""
        # Text generation using saved_model with TextDecoder

        text = "translate English to German: The house is wonderful and we wish to be here :)"

        # Save as saved model
        saved_model_dir = tempfile.mkdtemp()
        self.model_ar.save_as_serialize_module(saved_model_dir, overwrite=True)

        # Load saved model
        loaded = tf.saved_model.load(saved_model_dir)
        decoder = TextDecoder(model=loaded, decoder_start_token_id=DECODER_START_ID)  # for t5

        # Inputs
        inputs_hf = self.tokenizer(text, return_tensors='tf')
        inputs = {}
        inputs['encoder_input_ids'] = inputs_hf['input_ids']
        inputs['encoder_input_mask'] = inputs_hf['attention_mask']

        decoder_results = decoder.decode(inputs, mode='beam', num_beams=3, max_iterations=13, eos_id=DECODER_EOS_ID)
        top_prediction = decoder_results['predicted_ids'].numpy().tolist()[0][0]
        expected_outputs = [[644, 4598, 229, 19250, 64, 558, 7805, 1382, 1110, 3, 10, 61, 1]]

        assert [top_prediction] == expected_outputs
        logging.info("Test: Successful Saved Model Beam. ✅")
        shutil.rmtree(saved_model_dir)

    # @unittest.skip
    def test_auto_regressive_keras_model_beam(self):
        """Test Auto Regressive using Decoder Keras Model - Beam"""
        # Text generation using saved_model with TextDecoder

        text = "translate English to German: The house is wonderful and we wish to be here :)"

        decoder = TextDecoder(model=self.model_ar, decoder_start_token_id=DECODER_START_ID)  # for t5

        # Inputs
        inputs_hf = self.tokenizer(text, return_tensors='tf')
        inputs = {}
        inputs['encoder_input_ids'] = inputs_hf['input_ids']
        inputs['encoder_input_mask'] = inputs_hf['attention_mask']

        decoder_results = decoder.decode(inputs, mode='beam', num_beams=3, max_iterations=13, eos_id=DECODER_EOS_ID)
        top_prediction = decoder_results['predicted_ids'].numpy().tolist()[0][0]
        expected_outputs = [[644, 4598, 229, 19250, 64, 558, 7805, 1382, 1110, 3, 10, 61, 1]]

        assert [top_prediction] == expected_outputs
        logging.info("Test: Successful Keras Model Beam. ✅")

    @unittest.skip
    def test_auto_regressive_saved_model_topktopP(self):
        """Test Auto Regressive using Decoder Saved Model - topktopP"""
        # Text generation using saved_model with TextDecoder

        text = "translate English to German: The house is wonderful and we wish to be here :)"

        # Save as saved model
        saved_model_dir = tempfile.mkdtemp()
        self.model_ar.save_as_serialize_module(saved_model_dir, overwrite=True)

        # Load saved model
        loaded = tf.saved_model.load(saved_model_dir)
        decoder = TextDecoder(model=loaded, decoder_start_token_id=DECODER_START_ID)  # for t5

        # Inputs
        inputs_hf = self.tokenizer(text, return_tensors='tf')
        inputs = {}
        inputs['encoder_input_ids'] = inputs_hf['input_ids']
        inputs['encoder_input_mask'] = inputs_hf['attention_mask']

        _ = decoder.decode(inputs, mode='top_k_top_p', top_k=100, top_p=0.6, max_iterations=13, eos_id=DECODER_EOS_ID)

        logging.info("Test: Successful Saved Model top K top P. ✅")
        shutil.rmtree(saved_model_dir)

    @unittest.skip
    def test_auto_regressive_keras_model_topktopP(self):
        """Test Auto Regressive using Decoder Keras Model - topktopP"""

        text = "translate English to German: The house is wonderful and we wish to be here :)"

        decoder = TextDecoder(model=self.model_ar, decoder_start_token_id=DECODER_START_ID)  # for t5

        # Inputs
        inputs_hf = self.tokenizer(text, return_tensors='tf')
        inputs = {}
        inputs['encoder_input_ids'] = inputs_hf['input_ids']
        inputs['encoder_input_mask'] = inputs_hf['attention_mask']

        _ = decoder.decode(inputs, mode='top_k_top_p', top_k=100, top_p=0.6, max_iterations=13, eos_id=DECODER_EOS_ID)

        logging.info("Test: Successful Keras Model top k top P. ✅")

    @unittest.skip
    def test_auto_regressive_serializable_greedy(self):
        # Text generation using saved_model with TextDecoderSerializable
        decoder = TextDecoderSerializable(
            model=self.model_ar,
            decoder_start_token_id=DECODER_START_ID,
            max_iterations=15,
            mode="greedy",
            do_sample=False,
            eos_id=DECODER_EOS_ID,
        )

        # Save
        saved_model_dir = tempfile.mkdtemp()
        decoder_model = decoder.get_model()
        decoder_model.save_serialized(saved_model_dir, overwrite=True)

        # Load
        loaded_decoder = tf.saved_model.load(saved_model_dir)
        model_pb_decoder = loaded_decoder.signatures['serving_default']

        text = "translate English to German: The house is wonderful and we wish to be here :)"
        inputs_hf = self.tokenizer(text, return_tensors='tf')
        inputs = {}
        inputs['encoder_input_ids'] = inputs_hf['input_ids']
        inputs['encoder_input_mask'] = inputs_hf['attention_mask']

        decoder_results_serialized = model_pb_decoder(**inputs)
        expected_outputs = [[644, 4598, 229, 19250, 64, 558, 7805, 1382, 1110, 3, 10, 61, 1]]
        assert decoder_results_serialized['predicted_ids'].numpy().tolist()[0] == expected_outputs
        shutil.rmtree(saved_model_dir)
        logging.info("Test: Successful Serializable Model Greedy. ✅")

    @unittest.skip
    def test_auto_regressive_serializable_beam(self):
        # loaded   = tf.saved_model.load(saved_model_dir)
        decoder = TextDecoderSerializable(
            model=self.model_ar,
            decoder_start_token_id=DECODER_START_ID,
            max_iterations=15,
            num_beams=3,
            mode="beam",
            do_sample=False,
            eos_id=DECODER_EOS_ID,
        )

        # Save
        saved_model_dir = tempfile.mkdtemp()
        decoder_model = decoder.get_model()
        decoder_model.save_serialized(saved_model_dir, overwrite=True)

        # Load
        loaded_decoder = tf.saved_model.load(saved_model_dir)
        model_pb_decoder = loaded_decoder.signatures['serving_default']

        text = "translate English to German: The house is wonderful and we wish to be here :)"
        inputs_hf = self.tokenizer(text, return_tensors='tf')
        inputs = {}
        inputs['encoder_input_ids'] = inputs_hf['input_ids']
        inputs['encoder_input_mask'] = inputs_hf['attention_mask']

        decoder_results_serialized = model_pb_decoder(**inputs)
        top_prediction = decoder_results_serialized['predicted_ids'].numpy().tolist()[0][0]
        expected_outputs = [[644, 4598, 229, 19250, 64, 558, 7805, 1382, 1110, 3, 10, 61, 1]]
        assert [top_prediction] == expected_outputs
        shutil.rmtree(saved_model_dir)
        shutil.rmtree(saved_model_dir)
        logging.info("Test: Successful Serializable Model Beam. ✅")

    @unittest.skip
    def test_auto_regressive_serializable_top_k_top_p(self):
        # loaded   = tf.saved_model.load(saved_model_dir)
        decoder = TextDecoderSerializable(
            model=self.model_ar,
            decoder_start_token_id=DECODER_START_ID,
            max_iterations=15,
            top_k=100,
            top_p=0.7,
            mode="top_k_top_p",
            do_sample=False,
            eos_id=DECODER_EOS_ID,
        )

        # Save
        saved_model_dir = tempfile.mkdtemp()
        decoder_model = decoder.get_model()
        decoder_model.save_serialized(saved_model_dir, overwrite=True)

        # Load
        loaded_decoder = tf.saved_model.load(saved_model_dir)
        model_pb_decoder = loaded_decoder.signatures['serving_default']

        text = "translate English to German: The house is wonderful and we wish to be here :)"
        inputs_hf = self.tokenizer(text, return_tensors='tf')
        inputs = {}
        inputs['encoder_input_ids'] = inputs_hf['input_ids']
        inputs['encoder_input_mask'] = inputs_hf['attention_mask']

        _ = model_pb_decoder(**inputs)
        shutil.rmtree(saved_model_dir)
        logging.info("Test: Successful Serializable Model Beam. ✅")

    @unittest.skip
    def test_tflite(self):
        """Test T5 Tflite"""
        model = Model.from_pretrained(
            model_name=MODEL_NAME,
            convert_fn_type='tf',
            encoder_kwargs={'batch_size': 1, 'sequence_length': 32},
            decoder_kwargs={'batch_size': 1, 'sequence_length': 32},
        )

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
        # input_mask
        interpreter.set_tensor(input_details[1]['index'], tf.ones(input_details[1]['shape'], dtype=tf.int32))

        # decoder input ids
        interpreter.set_tensor(
            input_details[2]['index'],
            tf.random.uniform(input_details[2]['shape'], minval=0, maxval=100, dtype=tf.int32),
        )
        interpreter.invoke()
        tflite_output = interpreter.get_tensor(output_details[-1]['index'])

        tf.debugging.assert_equal(tflite_output.shape, (1, 32, 32128))
        logging.info("Test: TFlite Conversion. ✅")
        shutil.rmtree(tempdir)


if __name__ == '__main__':
    unittest.main()
