import numpy as np
import tensorflow as tf

from tf_transformers.text import (
    _gather_beams,
    _log_prob_from_logits,
    top_k_logits,
    top_p_logits,
)
from tf_transformers.utils import tf_utils


class TextDecoderSeq2Seq(object):
    """Decoder class"""

    def __init__(
        self,
        model,
        decoder_start_token_id,
        input_mask_ids=-1,
        input_type_ids=-1,
    ):
        """[summary]

        Args:
            tokenizer_fn ([type]): [description]
            model ([type]): [description]
            decode_start_token_id ([int]): [Encoder Decoder Model has a START ID]
            encoder_hidden_size (int, optional): \
                [The hidden dimension of encoder after attention
            concactantion]. Defaults to 512.
            decoder_attention_state (int, optional): [description]. Defaults to 512.
            decoder_hidden_size (int, optional): [description]. Defaults to 512.
            decoder_num_layers (int, optional): [description]. Defaults to 8.
            decoder_num_attention_heads (int, optional): [description]. Defaults to 8.
            decoder_input_mask_ids (int, optional): [description]. Defaults to 1.
            decoder_input_type_ids (int, optional): [description]. Defaults to 0.

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """
        self.decode_start_token_id = decoder_start_token_id
        self.decoder_input_type_ids = input_type_ids
        self.decoder_input_mask_ids = input_mask_ids

        self.model = model
        self.model_fn = None
        # keras Model
        if isinstance(model, tf.keras.Model):
            self.model_fn = self.model
            decoder_config = model.model_config["decoder"]
            (
                self.embedding_size,
                self.decoder_num_attention_heads,
                self.decoder_num_hidden_layers,
                self.decoder_attention_state,
            ) = self.auto_infer_config(decoder_config, saved_model=False)

            # Validate decoder type ids are there
            self.validate_decoder_type_ids(model.input)

            if self.decoder_start_token_id is None:
                self.decoder_start_token_id = decoder_config['decoder_start_token_id']
                print("Using default `decoder_start_token_id` {} from the model".format(self.decoder_start_token_id))

        # hubLayer (Not supported)
        # elif isinstance(model, hub.keras_layer.KerasLayer):
        #     self.model_fn = self.model
        else:
            # saved model
            if "saved_model" in str(type(self.model)):
                # Extract signature
                self.model_pb = self.model.signatures["serving_default"]

                def model_fn(x):
                    return self.model_pb(**x)

                self.model_fn = model_fn

            decoder_config = self.model.config
            (
                self.embedding_size,
                self.decoder_num_attention_heads,
                self.decoder_num_hidden_layers,
                self.decoder_attention_state,
            ) = self.auto_infer_config(decoder_config, saved_model=True)

            if self.decoder_start_token_id is None:
                self.decoder_start_token_id = decoder_config['decoder_start_token_id']
                print("Using default `decoder_start_token_id` {} from the model".format(self.decoder_start_token_id))

            # Validate decoder type ids are there
            self.validate_decoder_type_ids(self.model_pb.structured_input_signature[1])

        if self.model_fn is None:
            raise ValueError("Please check the type of your model")
        self.reserved_input_keys = ["input_ids", "input_mask", "input_type_ids"]

    def validate_decoder_type_ids(self, inputs):
        """Validate all extra IDS for models.

        Args:
            inputs (dict): Dict of Keras Inputs
        Raises:
            ValueError: If present and not provided
            ValueError: If present and not provided
        """
        if "decoder_input_type_ids" in inputs:
            if self.decoder_input_type_ids < 0:
                raise ValueError(
                    "Seems like you model has `decoder_input_type_ids`, \
                        but it hasn't set yet. Please provide a valid positive index for `decoder_input_type_ids`"
                )

    def auto_infer_config(self, config, saved_model=False):
        """Automatically Infer size and shape from config

        Args:
            config (dict): Necessary Model config

        Returns:
            [embedding_size, num_attention_heads, num_hidden_layers, attention_head_size]: Numbers
        """
        if saved_model:
            embedding_size = config["embedding_size"].numpy()
            decoder_num_attention_heads = config["num_attention_heads"].numpy()
            decoder_num_hidden_layers = config["num_hidden_layers"].numpy()
            attention_head_size = config["attention_head_size"].numpy()
            return (embedding_size, decoder_num_attention_heads, decoder_num_hidden_layers, attention_head_size)
        else:
            embedding_size = config["embedding_size"]
            decoder_num_attention_heads = config["num_attention_heads"]
            decoder_num_hidden_layers = config["num_hidden_layers"]
            attention_head_size = config["attention_head_size"]
            return (embedding_size, decoder_num_attention_heads, decoder_num_hidden_layers, attention_head_size)

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
        coordinates_reshaped = coordinates[:, :beam_size, -1] + tf.expand_dims(
            tf.range(tf.shape(coordinates)[0]) * beam_size, 1
        )
        # Old Approach
        # coordinates_reshaped = tf.reshape(coordinates_reshaped, -1)
        # all_cache_key   = tf.gather(all_cache_key, coordinates_reshaped , axis=1)
        # all_cache_value = tf.gather(all_cache_value, coordinates_reshaped, axis=1)

        coordinates_reshaped = tf.reshape(coordinates_reshaped, (1, -1))
        all_cache_key = tf.squeeze(tf.gather(all_cache_key, coordinates_reshaped, axis=1), axis=1)
        all_cache_value = tf.squeeze(tf.gather(all_cache_value, coordinates_reshaped, axis=1), axis=1)
        return all_cache_key, all_cache_value

    # Greedy Decoding
    def greedy_decode(self, inputs_original, max_iterations, do_sample=False, temperature=1.0, eos_id=-100):
        """
        Greedy Decoding

        text_list: a list of text
        length: 50 (decoding length)
        """
        inputs = inputs_original.copy()
        # We need this to return
        input_ids_original = inputs["encoder_input_ids"]
        batch_size = tf.shape(inputs["encoder_input_ids"])[0]
        # Initialize with zeros
        encoder_sequence_length = tf.shape(inputs["encoder_input_ids"])[1]
        decoder_start_sequence_length = 1

        encoder_hidden_states = tf.zeros((batch_size, encoder_sequence_length, self.embedding_size))
        all_cache_key = tf.zeros(
            (
                self.decoder_num_hidden_layers,
                batch_size,
                self.decoder_num_attention_heads,
                decoder_start_sequence_length,
                self.decoder_attention_state,
            )
        )
        all_cache_value = tf.zeros(
            (
                self.decoder_num_hidden_layers,
                batch_size,
                self.decoder_num_attention_heads,
                decoder_start_sequence_length,
                self.decoder_attention_state,
            )
        )

        # Inputs ready
        inputs["decoder_input_ids"] = tf.cast(tf.ones(shape=(batch_size, 1)) * self.decode_start_token_id, tf.int32)

        if self.decoder_input_type_ids > -1:
            inputs["decoder_input_type_ids"] = tf.ones_like(inputs["decoder_input_ids"]) * self.decoder_input_type_ids

        inputs["decoder_all_cache_key"] = all_cache_key
        inputs["decoder_all_cache_value"] = all_cache_value
        inputs["encoder_hidden_states"] = encoder_hidden_states

        all_predictions = []
        all_prediction_probs = []
        matched_positions = tf.constant([-1] * batch_size)
        i = 0
        eos_found = False
        while (i < max_iterations) and (eos_found is False):

            result = self.model_fn(inputs)

            model_logits = result["last_token_logits"] / temperature
            all_cache_key = result["decoder_all_cache_key"]
            all_cache_value = result["decoder_all_cache_value"]
            encoder_hidden_states = result["encoder_hidden_states"]
            if do_sample:
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
            if eos_id > -1:
                temp_m = tf.concat(all_predictions, axis=1)
                eos_check = tf.greater(
                    tf.reduce_prod(
                        tf.reduce_sum(
                            tf.cast(
                                tf.equal(tf.concat(all_predictions, axis=1), eos_id),
                                tf.int32,
                            ),
                            axis=[1],
                        )
                    ),
                    0,
                )
                if eos_check:
                    matched_positions = tf.argmax(tf.cast(tf.equal(eos_id, temp_m), tf.int32), axis=1)
                    eos_found = True

            inputs["decoder_input_ids"] = tf.cast(input_ids, tf.int32)
            inputs["decoder_all_cache_key"] = all_cache_key
            inputs["decoder_all_cache_value"] = all_cache_value
            inputs["encoder_hidden_states"] = encoder_hidden_states

            if self.decoder_input_type_ids > -1:
                inputs["decoder_input_type_ids"] = (
                    tf.ones_like(inputs["decoder_input_ids"]) * self.decoder_input_type_ids
                )
            if self.decoder_input_mask_ids > -1:
                inputs["decoder_input_mask"] = tf.ones_like(inputs["decoder_input_ids"]) * self.decoder_input_mask_ids

            # We assume decoder_input_type_ids remains same throughout decoding process
            i += 1
        matched_positions = tf.reshape(
            tf.argmax(
                tf.cast(tf.equal(eos_id, tf.concat(all_predictions, axis=1)), tf.int32),
                axis=1,
            ),
            -1,
        )
        # no eos matched positions will be 0, replace with -1
        eos_pos_mask = tf.cast(tf.equal(matched_positions, 0), tf.int32) * -1
        matched_positions = tf.cast(matched_positions, tf.int32) + eos_pos_mask

        all_predictions = tf.expand_dims(tf.concat(all_predictions, axis=1), 1)
        all_prediction_probs = tf.transpose(all_prediction_probs)
        return {
            "iterations": i + 1,  # scalar
            "input_ids": input_ids_original,  # 2D batch_size x seq_length
            "predicted_ids": all_predictions,  # 3D batch_size x 1 x decoded_length
            "matched_eos_pos": matched_positions,  # 1D (batch_size,)
            "prediction_probs": all_prediction_probs,  # 2D (batch_size x decoded_length)
        }

    def beam_decode(
        self,
        inputs_original,
        num_beams,
        max_iterations,
        temperature=1.0,
        alpha=0.0,
        top_k=0,
        top_p=0,
        do_sample=False,
        eos_id=-100,
    ):

        """
        Beam Decoding

        inputs_original (dict): inputs
        max_iterations (int): iterations
        beam_size (int): beam size
        do_sample (bool): Using multinomial distribution to sample the most likely \
         word, still uses beam
        temperature (float): control text generation
        eos_id (int): Default -100
        alpha (float): length penalty
        top_k (int): Top K sampling
        top_p (float): Between 0 and 1 . Nucleus sampling
        """

        beam_size = num_beams
        # We need this to return
        inputs = inputs_original["encoder_input_ids"]
        batch_size = tf.shape(inputs)[0]
        max_sequence_length = tf.shape(inputs)[1]  # noqa

        inputs_repeated = {}
        # Repeat for beam search
        for input_key, input_value in inputs_original.items():
            inputs_repeated[input_key] = tf.repeat(input_value, [beam_size], axis=0)

        # We take 2x beams
        beams_to_keep = 2 * beam_size
        batch_size_updated = inputs_repeated["encoder_input_ids"].shape[0]
        decoder_start_sequence_length = 1

        # Initialize with zeros
        encoder_sequence_length = tf.shape(inputs_repeated["encoder_input_ids"])[1]

        encoder_hidden_states = tf.zeros((batch_size_updated, encoder_sequence_length, self.embedding_size))
        zero_entry = tf.zeros(
            (
                self.decoder_num_hidden_layers,
                batch_size_updated,
                self.decoder_num_attention_heads,
                decoder_start_sequence_length,
                self.decoder_attention_state,
            )
        )
        all_cache_key = zero_entry
        all_cache_value = zero_entry

        # Inputs ready
        inputs_repeated["decoder_input_ids"] = tf.cast(
            tf.ones(shape=(batch_size_updated, 1)) * self.decode_start_token_id,
            tf.int32,
        )
        inputs_repeated["decoder_all_cache_key"] = all_cache_key
        inputs_repeated["decoder_all_cache_value"] = all_cache_value
        inputs_repeated["encoder_hidden_states"] = encoder_hidden_states

        # Type IDS
        if self.decoder_input_type_ids > -1:
            inputs_repeated["decoder_input_type_ids"] = (
                tf.ones_like(inputs_repeated["decoder_input_ids"]) * self.decoder_input_type_ids
            )

        # Input MASK
        if self.decoder_input_mask_ids > -1:
            inputs_repeated["decoder_input_mask"] = (
                tf.ones_like(inputs_repeated["decoder_input_ids"]) * self.decoder_input_type_ids
            )

        alive_log_probs = -np.inf * tf.ones((batch_size, beam_size - 1))

        alive_log_probs = tf.concat([tf.zeros([batch_size, 1]), alive_log_probs], axis=1)
        alive_seq = tf.zeros((batch_size, beam_size, 1))

        # Iterate
        i = 0
        eos_found = False
        while (i < max_iterations) and (eos_found is False):

            result = self.model_fn(inputs_repeated)

            model_logits = result["last_token_logits"] / temperature
            all_cache_key = result["decoder_all_cache_key"]
            all_cache_value = result["decoder_all_cache_value"]
            encoder_hidden_states = result["encoder_hidden_states"]

            if top_k > 0:
                model_logits = top_k_logits(model_logits, k=top_k)
            if top_p > 0:
                model_logits = top_p_logits(model_logits, p=top_p)

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
            length_penalty = tf.pow(((5.0 + (tf.cast(i, tf.float32) + 1.0)) / 6.0), alpha)
            log_probs = log_probs / length_penalty
            # Each batch item has beam_size * vocab_size candidate sequences. For each
            # batch item, get the k candidates with the highest log probabilities.
            flat_log_probs = tf.reshape(log_probs, [-1, beam_size * vocab_size])

            if do_sample:
                next_tokens = tf.random.categorical(
                    flat_log_probs, dtype=tf.int32, num_samples=beams_to_keep
                )  # (batch_size, 2 * num_beams)

                # # Compute next scores
                next_scores = tf.gather(flat_log_probs, next_tokens, batch_dims=1)  # (batch_size, 2 * num_beams)

                # # sort the sampled vector to make sure that \
                # the first num_beams samples are the best
                next_scores_indices = tf.argsort(next_scores, direction="DESCENDING", axis=1)
                next_scores = tf.gather(next_scores, next_scores_indices, batch_dims=1)  # (batch_size, num_beams * 2)
                next_tokens = tf.gather(next_tokens, next_scores_indices, batch_dims=1)  # (batch_size, num_beams * 2)

                topk_log_probs = next_scores
                topk_indices = next_tokens
            else:
                topk_log_probs, topk_indices = tf.nn.top_k(
                    flat_log_probs, k=beams_to_keep
                )  # (batch_size x k (beams_to_keep))

            topk_beam_indices = topk_indices // vocab_size
            topk_seq, coordinates = _gather_beams(alive_seq, topk_beam_indices, batch_size, beams_to_keep)
            topk_seq = tf.cast(topk_seq, tf.int32)
            topk_ids = topk_indices % vocab_size
            topk_seq = tf.concat([topk_seq, tf.expand_dims(topk_ids, axis=2)], axis=2)

            topk_alive_seq = topk_seq[:, :beam_size, :]
            alive_log_probs = topk_log_probs[:, :beam_size]
            input_ids = tf.reshape(topk_ids[:, :beam_size], [-1, 1])
            alive_seq = topk_alive_seq

            all_cache_key, all_cache_value = self.reorder_past_batches(
                all_cache_key, all_cache_value, coordinates, beam_size
            )

            inputs_repeated["decoder_input_ids"] = tf.cast(input_ids, tf.int32)
            inputs_repeated["decoder_all_cache_key"] = all_cache_key
            inputs_repeated["decoder_all_cache_value"] = all_cache_value
            inputs_repeated["encoder_hidden_states"] = encoder_hidden_states

            if self.decoder_input_type_ids > -1:
                inputs_repeated["decoder_input_type_ids"] = (
                    tf.ones_like(inputs_repeated["decoder_input_ids"]) * self.decoder_input_type_ids
                )
            if self.decoder_input_mask_ids > -1:
                inputs_repeated["decoder_input_mask"] = (
                    tf.ones_like(inputs_repeated["decoder_input_ids"]) * self.decoder_input_mask_ids
                )

            if eos_id > -1:
                eos_check = tf.greater(
                    tf.reduce_prod(
                        tf.reduce_sum(
                            tf.cast(tf.equal(topk_alive_seq, eos_id), tf.int32),
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
                                tf.cast(tf.equal(eos_id, topk_alive_seq), tf.int32),
                                axis=2,
                            ),
                            -1,
                        )
                        - 1
                    )
                    eos_found = True

            i += 1

        matched_positions = (
            tf.reshape(
                tf.argmax(tf.cast(tf.equal(eos_id, topk_alive_seq), tf.int32), axis=2),
                -1,
            )
            - 1
        )
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

    def top_k_top_p(
        self,
        inputs_original,
        max_iterations,
        top_k=0,
        top_p=0,
        temperature=1.0,
        do_sample=True,
        num_return_sequences=1,
        eos_id=-100,
    ):

        """
        Top TopK Decoding

        inputs_original (dict): inputs
        max_iterations (int): iterations
        do_sample (bool): Using multinomial distribution to sample the most likely \
         word, still uses beam
        temperature (float): control text generation
        eos_id (int): Default -100
        alpha (float): length penalty
        top_k (int): Top K sampling
        top_p (float): Between 0 and 1 . Nucleus sampling
        """

        # We need this to return
        input_ids_original = inputs_original["encoder_input_ids"]
        batch_size = tf.shape(input_ids_original)[0]
        # max_sequence_length = tf.shape(input_ids_original)[1]

        # Repeat for beam search
        inputs_repeated = {}
        for input_key, input_value in inputs_original.items():
            inputs_repeated[input_key] = tf.repeat(input_value, [num_return_sequences], axis=0)

        # We take 2x beams
        batch_size_updated = inputs_repeated["encoder_input_ids"].shape[0]
        decoder_start_sequence_length = 1
        encoder_sequence_length = tf.shape(inputs_repeated["encoder_input_ids"])[1]

        # Initialize with zeros
        encoder_hidden_states = tf.zeros((batch_size_updated, encoder_sequence_length, self.embedding_size))
        zero_entry = tf.zeros(
            (
                self.decoder_num_hidden_layers,
                batch_size_updated,
                self.decoder_num_attention_heads,
                decoder_start_sequence_length,
                self.decoder_attention_state,
            )
        )
        all_cache_key = zero_entry
        all_cache_value = zero_entry

        # Inputs ready
        inputs_repeated["decoder_input_ids"] = tf.cast(
            tf.ones(shape=(batch_size_updated, 1)) * self.decode_start_token_id,
            tf.int32,
        )
        inputs_repeated["decoder_all_cache_key"] = all_cache_key
        inputs_repeated["decoder_all_cache_value"] = all_cache_value
        inputs_repeated["encoder_hidden_states"] = encoder_hidden_states

        # Type IDS
        if self.decoder_input_type_ids > -1:
            inputs_repeated["decoder_input_type_ids"] = (
                tf.ones_like(inputs_repeated["decoder_input_ids"]) * self.decoder_input_type_ids
            )

        # Input MASK
        if self.decoder_input_mask_ids > -1:
            inputs_repeated["decoder_input_mask"] = (
                tf.ones_like(inputs_repeated["decoder_input_ids"]) * self.decoder_input_mask_ids
            )

        all_predictions = []
        all_prediction_probs = []

        # Iterate Over
        i = 0
        eos_found = False
        while (i < max_iterations) and (eos_found is False):
            result = self.model_fn(inputs_repeated)

            model_logits = result["last_token_logits"] / temperature
            all_cache_key = result["decoder_all_cache_key"]
            all_cache_value = result["decoder_all_cache_value"]
            encoder_hidden_states = result["encoder_hidden_states"]

            if top_k > 0:
                model_logits = top_k_logits(model_logits, k=top_k)
            if top_p > 0:
                model_logits = top_p_logits(model_logits, p=top_p)

            if do_sample:
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

            inputs_repeated["decoder_input_ids"] = tf.cast(input_ids, tf.int32)
            inputs_repeated["decoder_all_cache_key"] = all_cache_key
            inputs_repeated["decoder_all_cache_value"] = all_cache_value
            inputs_repeated["encoder_hidden_states"] = encoder_hidden_states

            if self.decoder_input_type_ids > -1:
                inputs_repeated["decoder_input_type_ids"] = (
                    tf.ones_like(inputs_repeated["decoder_input_ids"]) * self.decoder_input_type_ids
                )
            if self.decoder_input_mask_ids > -1:
                inputs_repeated["decoder_input_mask"] = (
                    tf.ones_like(inputs_repeated["decoder_input_ids"]) * self.decoder_input_mask_ids
                )

            if eos_id > -1:
                temp_m = tf.concat(all_predictions, axis=1)
                eos_check = tf.greater(
                    tf.reduce_prod(tf.reduce_sum(tf.cast(tf.equal(temp_m, eos_id), tf.int32), axis=[1])),
                    0,
                )
                if eos_check:
                    matched_positions = tf.argmax(tf.cast(tf.equal(eos_id, temp_m), tf.int32), axis=1)
                    eos_found = True

            i += 1

        matched_positions = (
            tf.reshape(
                tf.argmax(
                    tf.cast(tf.equal(eos_id, tf.concat(all_predictions, axis=1)), tf.int32),
                    axis=1,
                ),
                -1,
            )
            - 1
        )
        # no eos matched positions will be 0, replace with -1
        eos_pos_mask = tf.cast(tf.equal(matched_positions, 0), tf.int32) * -1
        matched_positions = tf.cast(matched_positions, tf.int32) + eos_pos_mask

        all_predictions = tf.reshape(tf.concat(all_predictions, axis=1), (batch_size, num_return_sequences, -1))
        all_prediction_probs = tf.transpose(all_prediction_probs)
        return {
            "iterations": i + 1,  # scalar
            "input_ids": input_ids_original,  # 2D batch_size x seq_length
            "predicted_ids": all_predictions,  # 3D batch_size x 1 x decoded_length
            "matched_eos_pos": matched_positions,  # 1D (batch_size,)
            "prediction_probs": all_prediction_probs,  # 2D (batch_size x decoded_length)
        }

    def decode(
        self,
        inputs,
        max_iterations,
        num_beams=3,
        sampling_temperature=1.0,
        alpha=0.0,
        num_return_sequences=1,
        eos_id=-100,
        top_k=0,
        top_p=0,
        do_sample=False,
        mode="greedy",
    ):

        """
        Decoding

        inputs (dict): inputs
        max_iterations (int): iterations
        beam_size (int): beam size
        do_sample (bool): Using multinomial distribution to sample the most likely \
         word, still uses beam
        temperature (float): control text generation
        eos_id (int): Default -100
        alpha (float): length penalty
        top_k (int): Top K sampling
        top_p (float): Between 0 and 1 . Nucleus sampling
        mode (str): ['greedy' , 'beam', 'top_k_top_p']
        """
        if mode == "greedy":
            result = self.greedy_decode(
                inputs,
                max_iterations=max_iterations,
                temperature=sampling_temperature,
                do_sample=do_sample,
                eos_id=eos_id,
            )
        if mode == "beam":
            result = self.beam_decode(
                inputs,
                max_iterations=max_iterations,
                num_beams=num_beams,
                temperature=sampling_temperature,
                alpha=alpha,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                eos_id=eos_id,
            )
        if mode == "top_k_top_p":
            result = self.top_k_top_p(
                inputs,
                max_iterations=max_iterations,
                temperature=sampling_temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                eos_id=eos_id,
            )
        return result
