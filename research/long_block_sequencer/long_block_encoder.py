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
"""A Long Block Encoder in Tensorflow 2.0"""
import tensorflow as tf
from tf_transformers.core import LegacyModel, LegacyLayer
from tf_transformers.activations import get_activation

class Long_Block_Encoder(LegacyLayer):
    def __init__(
        self, model_layer, num_splits,
        dense_dimension=None,
        gru_units=None,
        activation='gelu', 
        is_training=False, 
        use_dropout=False,
        use_gru_layer=True,
        **kwargs
    ):
        super(Long_Block_Encoder, self).__init__(
            is_training=is_training, use_dropout=use_dropout, name=model_layer.name, **kwargs
        )
        self.model_layer = model_layer
        self.num_splits = num_splits
        self.use_gru_layer = use_gru_layer
        if self.use_gru_layer:
            if gru_units is None:
                raise ValueError("When using GRU layer, set `gru_units`")
            self.projection_layer = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_units, return_sequences=True,
                                                                           name='gru_for_long_block'))
        else:
            if dense_dimension is None:
                raise ValueError("When using dense projection, set `dense_dimension`")
            activation = get_activation("gelu")
            self.projection_layer = tf.keras.layers.Dense(
            dense_dimension, activation=activation, kernel_initializer='glorot_uniform', name="gelu_for_long_block"
        )
        
        self._config_dict = model_layer._config_dict
        self._mask_mode   = model_layer._mask_mode
        self._sequence_length = model_layer._sequence_length
        self.model_inputs, self.model_outputs = self.get_model(initialize_only=True)
    
    def call(self, inputs):
        
        all_outputs_token_embeddings = []
        inputs_splitted = {}
        input_names = []
        for k, v in inputs.items():
            inputs_splitted[k] = tf.split(v, self.num_splits, axis=1)
            input_names.append(k)
            
        for i in range(self.num_splits):
            inputs_main = {}
            for name in input_names:
                inputs_main[name] = inputs_splitted[name][i]
            model_outputs = self.model_layer(inputs_main)
            all_outputs_token_embeddings.append(model_outputs['token_embeddings'])
            
        token_embeddings_concatanted = tf.concat(all_outputs_token_embeddings, axis=1) # over sequence length
        token_embeddings_concatanted = self.gru_layer(token_embeddings_concatanted)
        return {'token_embeddings': token_embeddings_concatanted}
    
 
    def get_model(self, initialize_only=False):
        inputs = {}
        for k, v in self.model_layer.model_inputs.items():
            shape = v.shape
            inputs[k] = tf.keras.layers.Input(
                shape[1:], batch_size=shape[0], name= k, dtype=v.dtype
            )
        layer_output = self(inputs)
        if initialize_only:
            return inputs, layer_output
        model = LegacyModel(inputs=inputs, outputs=layer_output, name="long_span_selection")
        model.model_config = self.model_layer._config_dict
        return model