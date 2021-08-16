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
"""Test ViT Models"""

import unittest

import tensorflow as tf
from absl import logging
from transformers import GPT2TokenizerFast as Tokenizer

from tf_transformers.models import ViTModel as Model

logging.get_absl_logger().name = "vit_testing"

MODEL_NAME = 'google/vit-base-patch16-224'


class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print("--------------------setUP--------------------------------------")
        self.model = Model.from_pretrained(MODEL_NAME)
        # self.tokenizer = Tokenizer.from_pretrained(MODEL_NAME)

    @unittest.skip
    def test_tf_conversion(self):
        raise NotImplementedError()

    #@unittest.skip
    def test_pt_conversion(self):
        
        import shutil

        try:
            shutil.rmtree("/tmp/tf_transformers_cache/{}".format(MODEL_NAME))
        except:
            pass
        model = Model.from_pretrained(MODEL_NAME, convert_fn_type='pt')
        logging.info("Test: PT Conversion. ✅")

    #@unittest.skip
    def test_tflite(self):
        import tempfile
        import shutil
        
        tempdir = tempfile.mkdtemp()
        model = Model.from_pretrained(MODEL_NAME, batch_size=1)
        model.save_serialized(tempdir, overwrite=True)
        
        converter = tf.lite.TFLiteConverter.from_saved_model("{}".format(tempdir)) # path to the SavedModel directory
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
        interpreter.set_tensor(input_details[0]['index'], tf.random.uniform(input_details[0]['shape']))
        interpreter.invoke()
        tflite_output = interpreter.get_tensor(output_details[-1]['index'])
        tf.debugging.assert_equal(tflite_output.shape, (1, 1000))
        logging.info("Test: TFlite Conversion. ✅")
        
        shutil.rmtree(tempdir)


if __name__ == '__main__':
    unittest.main()
