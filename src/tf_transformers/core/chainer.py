# -*- coding: utf-8 -*-
# mypy: ignore-errors

# coding=utf-8
# Copyright 2021 TF-Transformers Authors and The TensorFlow Authors.
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

import tensorflow as tf

from tf_transformers.core.legacy_layer import LegacyLayer
from tf_transformers.core.legacy_model import LegacyModel


class TextGenerationChainer(LegacyLayer):
    def __init__(self, tokenizer_model, model, **kwargs):
        super(TextGenerationChainer, self).__init__(is_training=False, use_dropout=False, name=model.name, **kwargs)

        if not isinstance(tokenizer_model, LegacyModel):
            raise ValueError("Tokenizer model should be inherited from tf_transformers.core.LegacyModel")
        self.model = model
        self.tokenizer_model = tokenizer_model
        self.tokenizer_layer = self.tokenizer_model.layers[-1]
        # Initialize model
        self.model_inputs, self.model_outputs = self.get_model(initialize_only=True)

    def call(self, inputs):
        """Call"""
        features = self.tokenizer_model(inputs)
        result = self.model(features)
        decoded_text = self.tokenizer_layer._tokenizer.detokenize(result['predicted_ids'][:, 0, :])
        result['decoded_text'] = decoded_text
        return result

    def get_model(self, initialize_only=False):
        """Get Model"""
        inputs = self.tokenizer_model.input
        if "iterations" in self.model.input:
            inputs['iterations'] = tf.keras.layers.Input(
                shape=(1,), batch_size=1, ragged=False, dtype=tf.int32, name="iterations"
            )
        layer_outputs = self(inputs)
        if initialize_only:
            return inputs, layer_outputs
        model = LegacyModel(inputs=inputs, outputs=layer_outputs, name=self.model.name)
        return model


class ClassificationChainer(LegacyLayer):
    def __init__(self, tokenizer_model, model, **kwargs):
        super(ClassificationChainer, self).__init__(is_training=False, use_dropout=False, name=model.name, **kwargs)
        self.model = model
        self.tokenizer_model = tokenizer_model
        # Initialize model
        self.model_inputs, self.model_outputs = self.get_model(initialize_only=True)

    def call(self, inputs):
        """Call"""
        features = self.tokenizer_model(inputs)
        result = self.model(features)
        return result

    def get_model(self, initialize_only=False):
        """Get Model"""
        inputs = self.tokenizer_model.input
        layer_outputs = self(inputs)
        if initialize_only:
            return inputs, layer_outputs
        model = LegacyModel(inputs=inputs, outputs=layer_outputs, name=self.model.name)
        return model
