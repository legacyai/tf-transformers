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


class BiasLayer(tf.keras.layers.Layer):
    def __init__(self, name="bias", trainable=True, initializer="zeros", *args, **kwargs):
        self._trainable = trainable
        self._initializer = initializer
        self._name = name
        super(BiasLayer, self).__init__(name=name, trainable=trainable, **kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight(
            name="bias", shape=(input_shape[-1],), initializer=self._initializer, trainable=self._trainable
        )

    def call(self, x):
        return x + self.bias
