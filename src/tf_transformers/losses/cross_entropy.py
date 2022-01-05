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
"""Cross entropy loss in Tensorflow 2.0"""
import tensorflow as tf


def cross_entropy_loss(labels, logits, label_weights=None):
    """
    logits: (.. , vocab_size)
    labels: (.. ) rank should be less than logits
    label_weights: labels shape

    Faster than above implementation
    """

    per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    if label_weights is None:
        label_weights = tf.ones_like(labels)
    label_weights = tf.cast(label_weights, tf.float32)
    per_example_loss = tf.cast(per_example_loss, tf.float32)
    per_example_loss = per_example_loss * label_weights
    # Take mean along axis=1 and then divide that first
    # Otherwise float16 will overflow
    numerator = tf.reduce_sum(per_example_loss, axis=1)
    denominator = tf.reduce_sum(label_weights, axis=1)
    loss = tf.math.divide_no_nan(numerator, denominator)
    return loss  # A tensor of loss of n examples equal to batch_size


def cross_entropy_loss_for_classification(labels, logits, label_weights=None):
    """
    logits: (.. , vocab_size)
    labels: (.. ) rank should be less than logits
    label_weights: labels shape

    Faster than above implementation
    """

    per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    if label_weights is None:
        label_weights = tf.ones_like(labels)
    label_weights = tf.cast(label_weights, tf.float32)
    per_example_loss = tf.cast(per_example_loss, tf.float32)
    per_example_loss = per_example_loss * label_weights
    # Take mean along axis=1 and then divide that first
    # Otherwise float16 will overflow
    # numerator = tf.reduce_sum(per_example_loss, axis=1)
    # denominator = tf.reduce_sum(label_weights, axis=1)
    # loss = tf.math.divide_no_nan(numerator, denominator)
    return per_example_loss  # A tensor of loss of n examples equal to batch_size


def cross_entropy_loss_label_smoothing(labels, logits, smoothing=0.1, label_weights=None):
    """
    logits: (.. , vocab_size)
    labels: (.. ) rank should be less than logits
    label_weights: labels shape

    Faster than above implementation
    """
    confidence = 1.0 - smoothing
    vocab_size = tf.shape(logits)[-1]
    vocab_float = tf.cast(vocab_size - 1, tf.float32)
    low_confidence = (1.0 - confidence) / vocab_float
    soft_targets = tf.one_hot(labels, depth=vocab_size, on_value=confidence, off_value=low_confidence)
    per_example_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=soft_targets)
    # Calculate the best (lowest) possible value of cross entropy, and
    # subtract from the cross entropy loss.
    normalizing_constant = -(
        confidence * tf.math.log(confidence) + vocab_float * low_confidence * tf.math.log(low_confidence + 1e-20)
    )
    per_example_loss = per_example_loss - tf.cast(normalizing_constant, per_example_loss.dtype)
    per_example_loss = tf.cast(per_example_loss, tf.float32)

    if label_weights is None:
        label_weights = tf.ones_like(labels)
    label_weights = tf.cast(label_weights, tf.float32)
    per_example_loss = per_example_loss * label_weights
    # Take mean along axis=1 and then divide that first
    # Otherwise float16 will overflow
    numerator = tf.reduce_sum(per_example_loss, axis=1)
    denominator = tf.reduce_sum(label_weights, axis=1)
    loss = tf.math.divide_no_nan(numerator, denominator)
    return loss
