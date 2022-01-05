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
"""Text Auto Regressive Decoder for Tensorflow Model and saved_model. This uses while loop
This is Serializable as saved_model"""
import numpy as np
import tensorflow as tf

from tf_transformers.core import LegacyModel
from tf_transformers.text import (
    _gather_beams,
    _log_prob_from_logits,
    top_k_logits,
    top_p_logits,
)
from tf_transformers.utils import tf_utils


class TextDecoderModel(tf.keras.layers.Layer):
    """TextDecoderSerializable - This class is responsible for
    saving the model along with decoding
    operation as a saved_model, which makes deployment in production easier.
    """

    def __init__(
        self,
        model,
        mode,
        max_iterations,
        batch_size=None,
        sequence_length=None,
        max_sequence_length=None,
        temperature=1.0,
        alpha=0.0,
        num_beams=1,
        eos_id=-100,
        do_sample=False,
        top_k=0,
        top_p=0,
        num_return_sequences=1,
        input_type_ids=-1,
        input_mask_ids=1,
    ):
        """

        Args:
            model ([tf.keras.Model / tf.keras.Layer]): [The model with which decoding
            has to be performed]
            max_iterations ([int]): [Maximum iterations for decoding]
            num_attention_heads ([int]): [Attention heads of model]
            num_layers ([int]): [Number of model layers]
            attention_state ([int]): [embedding_size//num_attention_heads]
            mode ([str]): ['greedy' , 'beam', 'top_k_top_p']
            batch_size:
            sequence_length:
            input_name_list ([List of int]): [Names of model inputs like input_ids,
            input_mask, etc]
            beam_size (int, optional): [Number of beam size]. Defaults to 1.
            eos_id (int, optional): [end of sentence token id]. Defaults to -100.
            do_sample (bool, optional): [Multinomial sampling]. Defaults to False.
            top_k (int, optional): [top k]. Defaults to 0.
            top_p (int, optional): [top p Nucleus]. Defaults to 0.
            input_mask_ids (int, optional): [if your model has this, provide it].
            Defaults to None.
            input_type_ids (int, optional): [if your model has this, provide it].
            Defaults to None.
            num_return_sequences: (int): [No of return sequences for topk top beam].
            Defaults to 1.
        """

        super(TextDecoderModel, self).__init__()

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.max_iterations = max_iterations

        self.input_type_ids = input_type_ids

        self.model = model

        model_config = model.model_config
        (
            self.embedding_size,
            self.num_attention_heads,
            self.num_hidden_layers,
            self.attention_state,
        ) = self.auto_infer_config(model_config)

        self.input_type_ids = input_type_ids
        self.input_mask_ids = input_mask_ids
        # Validate decoder type ids are there
        self.validate_decoder_type_ids(model.input)

        self.eos_id = eos_id
        self.mode = mode

        self.beam_size = num_beams
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample
        self.num_return_sequences = num_return_sequences

        self.temperature = temperature
        self.alpha = alpha

    def validate_decoder_type_ids(self, inputs):
        if "decoder_input_type_ids" in inputs:
            if self.decoder_input_type_ids < 0:
                raise ValueError(
                    "Seems like you model has `decoder_input_type_ids`,\
                         but it hasn't set yet. Please provide a valid positive index for `decoder_input_type_ids`"
                )

    def auto_infer_config(self, config):
        """[summary]

        Args:
            config ([type]): [description]

        Returns:
            [type]: [description]
        """
        embedding_size = config["embedding_size"]
        decoder_num_attention_heads = config["num_attention_heads"]
        decoder_num_hidden_layers = config["num_hidden_layers"]
        attention_head_size = config["attention_head_size"]
        return (
            embedding_size,
            decoder_num_attention_heads,
            decoder_num_hidden_layers,
            attention_head_size,
        )

    def get_model(self):
        # Call the model in init itself
        inputs = {}
        for input_name, input_tensor in self.model.input.items():
            if input_name in ["input_ids", "input_type_ids", "input_mask"]:
                inputs[input_name] = input_tensor
        layer_outputs = self(inputs)
        decoder_model = LegacyModel(inputs=inputs, outputs=layer_outputs, name="decoder_model")
        return decoder_model

    def reorder_past_batches(self, all_cache_key, all_cache_value, coordinates, beam_size):
        """Reorder the input batch based on beam predictions
        Future beams changes the best path order

        Args:
            all_cache_key ([tf.tensor]): [K from Transformers]
            all_cache_value ([tf.tensor]): [V from Transformers]
            coordinates ([tf.tensor (bach_size x beam_size)]): [The order ]
            beam_size ([int/tf.tensor]): [Number of beams]

        Returns:
            all_cache_key and all_cache_value (Updated)

        """
        # Old Approach
        # coordinates_reshaped = tf.reshape(coordinates_reshaped, -1)
        # all_cache_key   = tf.gather(all_cache_key, coordinates_reshaped , axis=1)
        # all_cache_value = tf.gather(all_cache_value, coordinates_reshaped, axis=1)

        # coordinates_reshaped = coordinates[:, :beam_size, -1] + tf.expand_dims(
        #     tf.range(tf.shape(coordinates)[0]) * beam_size, 1
        # )
        # # TODO: This is somewhat required in this specifc TextDecoderModel
        # coordinates_reshaped = tf.reshape(coordinates_reshaped, -1)
        # all_cache_key = tf.gather(all_cache_key, coordinates_reshaped, axis=1)
        # all_cache_value = tf.gather(all_cache_value, coordinates_reshaped, axis=1)
        return all_cache_key, all_cache_value

    @tf.function
    def call(self, inputs):
        if self.mode == "greedy":
            return self.greedy_decode(inputs)
        if self.mode == "beam":
            return self.beam_decode(inputs)
        if self.mode == 'top_k_top_p':
            return self.top_k_top_p(inputs)

    # Greedy Decoding
    def greedy_decode(self, inputs_original):
        """
        Greedy Decoding

        inputs_original (dict): inputs
        max_iterations (int): Max iterations
        do_sample (bool): Uses multinomial sampling to select most lilely word
        temperature (float): control text generation
        eos_id (int): Default -100
        """
        inputs = inputs_original.copy()
        # We need this to return
        input_ids_original = inputs["input_ids"]
        batch_size = tf.shape(input_ids_original)[0]
        max_sequence_length = tf.shape(input_ids_original)[1]

        # We need this in assign_K_V_zeros function
        input_ids_copy = tf.identity(inputs["input_ids"])

        # Initialize with zeros
        zero_entry = tf.zeros(
            (
                self.num_hidden_layers,
                batch_size,
                self.num_attention_heads,
                max_sequence_length,
                self.attention_state,
            )
        )
        all_cache_key = zero_entry
        all_cache_value = zero_entry
        past_length = tf.expand_dims(tf.zeros(batch_size, dtype=tf.int32), 0)

        # Inputs ready
        inputs["all_cache_key"] = all_cache_key
        inputs["all_cache_value"] = all_cache_value
        inputs["past_length"] = past_length

        all_predictions = []
        all_prediction_probs = []
        # matched_positions = tf.constant([-1] * batch_size)
        # Iterate
        i = 0
        eos_found = False
        while (i < self.max_iterations) and (eos_found is False):

            result = self.model(inputs)
            model_logits = result["last_token_logits"] / self.temperature
            all_cache_key = result["all_cache_key"]
            all_cache_value = result["all_cache_value"]
            past_length = result["past_length"]

            if self.do_sample:
                prediction_ids = tf.random.categorical(model_logits, num_samples=1)
                prediction_probs = tf_utils.gather_values_from_2d_tensor(model_logits, prediction_ids)
                input_ids = tf.cast(prediction_ids, tf.int32)
                all_prediction_probs.append(prediction_probs)
            else:
                prediction_ids = tf.argmax(model_logits, axis=1)
                prediction_probs = tf.reduce_max(model_logits, axis=1)
                input_ids = tf.expand_dims(prediction_ids, axis=1)
                all_prediction_probs.append(prediction_probs)

            all_predictions.append(input_ids)

            if self.eos_id > -1:
                temp_m = tf.concat(all_predictions, axis=1)
                eos_check = tf.greater(
                    tf.reduce_prod(tf.reduce_sum(tf.cast(tf.equal(temp_m, self.eos_id), tf.int32), axis=[1])),
                    0,
                )
                if eos_check:
                    matched_positions = tf.argmax(tf.cast(tf.equal(self.eos_id, temp_m), tf.int32), axis=1)
                    # matched_positions += 1
                    eos_found = True

            if i == 0:

                # This was the old way
                # all_cache_key   = assign_zeros_to_K_V(all_cache_key, \
                # input_ids_copy, batch_size, max_sequence_length)
                # all_cache_value = assign_zeros_to_K_V(all_cache_value, \
                # input_ids_copy, batch_size, max_sequence_length)

                masks = tf.cast(tf.not_equal(input_ids_copy, -1), tf.float32)
                masks = tf.reshape(
                    masks,
                    (1, tf.shape(input_ids_copy)[0], 1, tf.shape(input_ids_copy)[1], 1),
                )
                all_cache_key = all_cache_key * masks
                all_cache_value = all_cache_value * masks

            inputs["input_ids"] = tf.cast(input_ids, tf.int32)
            inputs["all_cache_key"] = all_cache_key
            inputs["all_cache_value"] = all_cache_value
            inputs["past_length"] = past_length

            if self.input_type_ids > -1:
                inputs["input_type_ids"] = tf.ones_like(inputs["input_ids"]) * self.input_type_ids
            if self.input_mask_ids > -1:
                inputs["input_mask"] = tf.ones_like(inputs["input_ids"]) * self.input_mask_ids

            i += 1

        all_predictions = tf.cast(tf.concat(all_predictions, axis=1), tf.int32)
        #         matched_positions = tf.reshape(
        #             tf.argmax(
        #                 tf.cast(tf.equal(self.eos_id, all_predictions), tf.int32), axis=1
        #             ),
        #             -1,
        #         )
        matched_positions = tf.argmax(tf.cast(tf.equal(self.eos_id, all_predictions), tf.int32), axis=1)
        # no eos matched positions will be 0, replace with -1
        eos_pos_mask = tf.cast(tf.equal(matched_positions, 0), tf.int32) * -1
        matched_positions = tf.cast(matched_positions, tf.int32) + eos_pos_mask

        all_predictions = tf.cast(tf.expand_dims(all_predictions, axis=1), tf.int32)
        all_prediction_probs = tf.transpose(all_prediction_probs)
        return {
            "iterations": i + 1,  # scalar
            "input_ids": input_ids_original,  # 2D batch_size x seq_length
            "predicted_ids": all_predictions,  # 3D batch_size x 1 x decoded_length
            "matched_eos_pos": matched_positions,  # 1D (batch_size,)
            "prediction_probs": all_prediction_probs,  # 2D (batch_size x decoded_length)
        }

    def beam_decode(self, inputs_original):

        """
        Beam Decoding

        """

        beam_size = self.beam_size
        # We need this to return
        inputs = inputs_original["input_ids"]
        batch_size = tf.shape(inputs)[0]
        max_sequence_length = tf.shape(inputs)[1]

        # We need this in assign_K_V_zeros function, we should place ot before repeat
        # input_ids_copy = tf.identity(tokenized_input_dict_ragged['input_ids'])

        # Repeat for beam search
        # We need to replicate inpust to beam_size
        inputs_repeated = {}
        for input_key, input_value in inputs_original.items():
            inputs_repeated[input_key] = tf.repeat(input_value, [beam_size], axis=0)

        # We take 2x beams
        beams_to_keep = 2 * beam_size
        batch_size_updated = tf.shape(inputs_repeated["input_ids"])[0]

        # Initialize with zeros
        zero_entry = tf.zeros(
            (
                self.num_hidden_layers,
                batch_size_updated,
                self.num_attention_heads,
                max_sequence_length,
                self.attention_state,
            )
        )
        all_cache_key = zero_entry
        all_cache_value = zero_entry
        past_length = tf.expand_dims(tf.zeros(batch_size_updated, dtype=tf.int32), 0)

        # Inputs ready
        inputs_repeated["all_cache_key"] = all_cache_key
        inputs_repeated["all_cache_value"] = all_cache_value
        inputs_repeated["past_length"] = past_length

        if self.input_type_ids > -1:
            inputs_repeated["input_type_ids"] = tf.ones_like(inputs_repeated["input_ids"]) * self.input_type_ids
        if self.input_mask_ids > -1:
            inputs_repeated["input_mask"] = tf.ones_like(inputs_repeated["input_ids"]) * self.input_mask_ids

        # To keep tract of ids which are alive (exist) after beam search
        alive_log_probs = -np.inf * tf.ones((batch_size, beam_size - 1))

        alive_log_probs = tf.concat([tf.zeros([batch_size, 1]), alive_log_probs], axis=1)
        alive_seq = tf.zeros((batch_size, beam_size, 1))

        # Iterate
        i = 0
        eos_found = False
        while (i < self.max_iterations) and (eos_found is False):
            result = self.model(inputs_repeated)

            model_logits = result["last_token_logits"] / self.temperature
            all_cache_key = result["all_cache_key"]
            all_cache_value = result["all_cache_value"]
            past_length = result["past_length"]

            if self.top_k > 0:
                model_logits = self.top_k_logits(model_logits, k=self.top_k)
            if self.top_p > 0:
                model_logits = self.top_p_logits(model_logits, p=self.top_p)

            vocab_size = tf.shape(model_logits)[1]
            logits = tf.reshape(model_logits, (batch_size, beam_size, -1))
            # # Convert logits to normalized log probs
            candidate_log_probs = _log_prob_from_logits(logits)

            # Calculate new log probabilities if each of the alive sequences were
            # extended # by the the candidate IDs.
            # Shape [batch_size, beam_size, vocab_size]
            log_probs = candidate_log_probs + tf.expand_dims(alive_log_probs, 2)

            # Calculate new log probabilities if each of the alive sequences were
            # extended # by the the candidate IDs.
            # Shape [batch_size, beam_size, vocab_size]
            log_probs = candidate_log_probs + tf.expand_dims(alive_log_probs, axis=2)

            # Add length penalty
            length_penalty = tf.pow(((5.0 + (tf.cast(i, tf.float32) + 1.0)) / 6.0), self.alpha)
            log_probs = log_probs / length_penalty

            # Each batch item has beam_size * vocab_size candidate sequences. For each
            # batch item, get the k candidates with the highest log probabilities.
            flat_log_probs = tf.reshape(log_probs, [-1, beam_size * vocab_size])

            if self.do_sample:
                next_tokens = tf.random.categorical(
                    flat_log_probs, dtype=tf.int32, num_samples=beams_to_keep
                )  # (batch_size, 2 * self.num_beams)

                # # Compute next scores
                next_scores = tf.gather(flat_log_probs, next_tokens, batch_dims=1)  # (batch_size, 2 * self.num_beams)

                # # sort the sampled vector to make sure that the first \
                # self.num_beams samples are the best
                next_scores_indices = tf.argsort(next_scores, direction="DESCENDING", axis=1)
                next_scores = tf.gather(
                    next_scores, next_scores_indices, batch_dims=1
                )  # (batch_size, self.num_beams * 2)
                next_tokens = tf.gather(
                    next_tokens, next_scores_indices, batch_dims=1
                )  # (batch_size, self.num_beams * 2)

                topk_log_probs = next_scores
                topk_indices = next_tokens
            else:
                topk_log_probs, topk_indices = tf.nn.top_k(flat_log_probs, k=beams_to_keep)  # (batch_size x k)

            topk_beam_indices = topk_indices // vocab_size
            topk_seq, coordinates = _gather_beams(alive_seq, topk_beam_indices, batch_size, beams_to_keep)
            topk_seq = tf.cast(topk_seq, tf.int32)
            topk_ids = topk_indices % vocab_size
            topk_seq = tf.concat([topk_seq, tf.expand_dims(topk_ids, axis=2)], axis=2)

            topk_alive_seq = topk_seq[:, :beam_size, :]
            alive_log_probs = topk_log_probs[:, :beam_size]
            input_ids = tf.reshape(topk_ids[:, :beam_size], [-1, 1])
            alive_seq = topk_alive_seq

            if i == 0:
                # This was the old way
                # all_cache_key = assign_zeros_to_K_V(all_cache_key, \
                # input_ids_copy, batch_size, max_sequence_length)
                # all_cache_value = assign_zeros_to_K_V(all_cache_value, \
                # input_ids_copy, batch_size, max_sequence_length)

                # We need batch x beam_size , so we use batch_size_updated
                masks = tf.cast(
                    tf.not_equal(inputs_repeated["input_ids"], -1),
                    tf.float32,
                )
                masks = tf.reshape(masks, (1, batch_size_updated, 1, max_sequence_length, 1))
                all_cache_key = all_cache_key * masks
                all_cache_value = all_cache_value * masks

            all_cache_key, all_cache_value = self.reorder_past_batches(
                all_cache_key, all_cache_value, coordinates, beam_size
            )
            # current_ids = topk_alive_seq[:,:,-1]
            inputs_repeated["input_ids"] = tf.cast(input_ids, tf.int32)
            inputs_repeated["all_cache_key"] = all_cache_key
            inputs_repeated["all_cache_value"] = all_cache_value
            inputs_repeated["past_length"] = past_length

            if self.input_type_ids > -1:
                inputs_repeated["input_type_ids"] = tf.ones_like(inputs_repeated["input_ids"]) * self.input_type_ids
            if self.input_mask_ids > -1:
                inputs_repeated["input_mask"] = tf.ones_like(inputs_repeated["input_ids"]) * self.input_mask_ids

            if self.eos_id > -1:
                eos_check = tf.greater(
                    tf.reduce_prod(
                        tf.reduce_sum(
                            tf.cast(tf.equal(topk_alive_seq, self.eos_id), tf.int32),
                            axis=[2],
                        )
                    ),
                    0,
                )
                # We check for eos_id for all batches and all beams
                if eos_check:
                    matched_positions = (
                        tf.reshape(
                            tf.argmax(
                                tf.cast(tf.equal(self.eos_id, topk_alive_seq), tf.int32),
                                axis=2,
                            ),
                            -1,
                        )
                        - 1
                    )
                    eos_found = True

            i += 1

        matched_positions = tf.argmax(tf.cast(tf.equal(self.eos_id, topk_alive_seq), tf.int32), axis=2)
        # no eos matched positions will be 0, replace with -1
        eos_pos_mask = tf.cast(tf.equal(matched_positions, 0), tf.int32) * -1
        matched_positions = tf.cast(matched_positions, tf.int32) + eos_pos_mask

        # TODO : Add prediction_probs
        return {
            "iterations": i + 1,  # scalar
            "input_ids": inputs,  # 2D batch_size x seq_length
            "predicted_ids": topk_alive_seq[:, :, 1:],  # to avoid initial 0,  # 3D batch_size x 1 x decoded_length
            "matched_eos_pos": matched_positions,  # 1D (batch_size,)
        }

    def top_k_top_p(self, inputs_original):
        """
        Top TopK Decoding

        """

        # We need this to return
        input_ids_original = inputs_original["input_ids"]
        batch_size = tf.shape(input_ids_original)[0]
        max_sequence_length = tf.shape(input_ids_original)[1]  # noqa

        # Repeat for beam search
        inputs_repeated = {}
        for input_key, input_value in inputs_original.items():
            inputs_repeated[input_key] = tf.repeat(input_value, [self.num_return_sequences], axis=0)

        # We take 2x beams
        batch_size_updated = tf.shape(inputs_repeated["input_ids"])[0]

        # Initialize with zeros
        zero_entry = tf.zeros(
            (
                self.num_hidden_layers,
                batch_size_updated,
                self.num_attention_heads,
                max_sequence_length,
                self.attention_state,
            )
        )
        all_cache_key = zero_entry
        all_cache_value = zero_entry
        past_length = tf.expand_dims(tf.zeros(batch_size_updated, dtype=tf.int32), 0)

        # Inputs ready
        inputs_repeated["all_cache_key"] = all_cache_key
        inputs_repeated["all_cache_value"] = all_cache_value
        inputs_repeated["past_length"] = past_length

        if self.input_type_ids > -1:
            inputs_repeated["input_type_ids"] = tf.ones_like(inputs_repeated["input_ids"]) * self.input_type_ids
        if self.input_mask_ids > -1:
            inputs_repeated["input_mask"] = tf.ones_like(inputs_repeated["input_ids"]) * self.input_mask_ids

        all_predictions = []
        all_prediction_probs = []
        # matched_positions = tf.constant([-1] * batch_size_updated)

        # Iterate Over
        i = 0
        eos_found = False
        while (i < self.max_iterations) and (eos_found is False):
            result = self.model(inputs_repeated)
            model_logits = result["last_token_logits"]
            all_cache_key = result["all_cache_key"]
            all_cache_value = result["all_cache_value"]
            past_length = result["past_length"]

            if self.top_k > 0:
                model_logits = top_k_logits(model_logits, k=self.top_k)
            if self.top_p > 0:
                model_logits = top_p_logits(model_logits, p=self.top_p)

            if self.do_sample:
                prediction_ids = tf.random.categorical(model_logits, num_samples=1)
                prediction_probs = tf_utils.gather_values_from_2d_tensor(model_logits, prediction_ids)
                input_ids = tf.cast(prediction_ids, tf.int32)
                all_prediction_probs.append(prediction_probs)
            else:
                prediction_ids = tf.argmax(model_logits, axis=1)
                prediction_probs = tf.reduce_max(model_logits, axis=1)
                input_ids = tf.cast(tf.expand_dims(prediction_ids, axis=1), tf.int32)
                all_prediction_probs.append(prediction_probs)

            all_predictions.append(input_ids)
            if i == 0:
                # all_cache_key = assign_zeros_to_K_V(all_cache_key, \
                # input_ids_copy, batch_size, max_sequence_length)
                # all_cache_value = assign_zeros_to_K_V(all_cache_value, \
                # input_ids_copy, batch_size, max_sequence_length)

                masks = tf.cast(
                    tf.not_equal(inputs_repeated["input_ids"], -1),
                    tf.float32,
                )
                masks = tf.reshape(masks, (1, batch_size_updated, 1, max_sequence_length, 1))

                all_cache_key = all_cache_key * masks
                all_cache_value = all_cache_value * masks

            inputs_repeated["input_ids"] = tf.cast(input_ids, tf.int32)
            inputs_repeated["all_cache_key"] = all_cache_key
            inputs_repeated["all_cache_value"] = all_cache_value
            inputs_repeated["past_length"] = past_length

            if self.input_type_ids > -1:
                inputs_repeated["input_type_ids"] = tf.ones_like(inputs_repeated["input_ids"]) * self.input_type_ids
            if self.input_mask_ids > -1:
                inputs_repeated["input_mask"] = tf.ones_like(inputs_repeated["input_ids"]) * self.input_mask_ids

            if self.eos_id:
                temp_m = tf.concat(all_predictions, axis=1)
                eos_check = tf.greater(
                    tf.reduce_prod(tf.reduce_sum(tf.cast(tf.equal(temp_m, self.eos_id), tf.int32), axis=[1])),
                    0,
                )
                if eos_check:
                    matched_positions = tf.argmax(tf.cast(tf.equal(self.eos_id, temp_m), tf.int32), axis=1)
                    eos_found = True

            i += 1

        matched_positions = (
            tf.reshape(
                tf.argmax(
                    tf.cast(tf.equal(self.eos_id, tf.concat(all_predictions, axis=1)), tf.int32),
                    axis=1,
                ),
                -1,
            )
            - 1
        )
        # no eos matched positions will be 0, replace with -1
        eos_pos_mask = tf.cast(tf.equal(matched_positions, 0), tf.int32) * -1
        matched_positions = tf.cast(matched_positions, tf.int32) + eos_pos_mask

        all_predictions = tf.reshape(tf.concat(all_predictions, axis=1), (batch_size, self.num_return_sequences, -1))
        all_prediction_probs = tf.transpose(all_prediction_probs)
        return {
            "iterations": i + 1,  # scalar
            "input_ids": input_ids_original,  # 2D batch_size x seq_length
            "predicted_ids": all_predictions,  # 3D batch_size x 1 x decoded_length
            "matched_eos_pos": matched_positions,  # 1D (batch_size,)
            "prediction_probs": all_prediction_probs,  # 2D (batch_size x decoded_length)
        }
