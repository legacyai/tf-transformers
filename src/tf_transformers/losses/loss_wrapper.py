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
"""Simple wrapper for Tensorflow loss over last and joint layers"""
import tensorflow as tf

from tf_transformers.losses import (
    cross_entropy_loss,
    cross_entropy_loss_for_classification,
)


def get_1d_classification_loss(label_column='labels', prediction_column='class_logits', loss_type: str = None):
    """Get loss based on joint or normal

    Args:
        label_column: the key from data dict.
        y_column: the key of model predictions.
        loss_type: None or "joint"
    """

    if loss_type and loss_type == 'joint':

        def loss_fn(y_true_dict, y_pred_dict):
            """Joint loss over all layers"""
            loss_dict = {}
            loss_holder = []
            for layer_count, per_layer_output in enumerate(y_pred_dict[prediction_column]):

                loss = cross_entropy_loss_for_classification(
                    labels=tf.squeeze(y_true_dict[label_column], axis=1), logits=per_layer_output
                )
                loss_dict['loss_{}'.format(layer_count + 1)] = loss
                loss_holder.append(loss)

            # Mean over batch, across all layers
            loss_dict['loss'] = tf.reduce_mean(loss_holder, axis=0)
            return loss_dict

    else:

        def loss_fn(y_true_dict, y_pred_dict):
            """last layer loss"""
            loss_dict = {}
            loss = cross_entropy_loss_for_classification(
                labels=tf.squeeze(y_true_dict[label_column], axis=1), logits=y_pred_dict[prediction_column]
            )
            loss_dict['loss'] = loss
            return loss_dict

    return loss_fn


def get_lm_loss(
    label_column='masked_lm_labels',
    label_weights_column='masked_lm_weights',
    prediction_column='token_logits',
    loss_type=None,
):
    """Language Model loss

    Args:
        label_column: the key from data dict.
        y_column: the key of model predictions.
        loss_type: None or "joint"
    """
    if loss_type and loss_type == 'joint':

        def lm_loss(y_true_dict, y_pred_dict):
            """Joint loss over all layers"""
            loss_dict = {}
            loss_holder = []
            for layer_count, per_layer_output in enumerate(y_pred_dict['all_layer_{}'.format(prediction_column)]):

                loss = cross_entropy_loss(
                    labels=y_true_dict[label_column],
                    logits=per_layer_output,
                    label_weights=y_true_dict[label_weights_column],
                )
                loss_dict['loss_{}'.format(layer_count + 1)] = loss
                loss_holder.append(loss)
            loss_dict['loss'] = tf.reduce_mean(loss_holder, axis=0)
            return loss_dict

    else:

        def lm_loss(y_true_dict, y_pred_dict):
            """Joint loss over all layers"""
            loss_dict = {}
            loss = cross_entropy_loss(
                labels=y_true_dict[label_column],
                logits=y_pred_dict[prediction_column],
                label_weights=y_true_dict[label_weights_column],
            )
            loss_dict['loss'] = loss
            return loss_dict

    return lm_loss
