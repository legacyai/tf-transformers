import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from tf_transformers.text import (_gather_beams, _log_prob_from_logits,
                                  assign_zeros_to_K_V, top_k_logits,
                                  top_p_logits)


class TextDecoder(object):
    """Decoder class"""

    def __init__(
        self,
        model,
        input_mask_ids=-1,
        input_type_ids=-1,
    ):
        """
        Arguments:
            model_fn: convert (inputs) to (logits and necessary outputs)
            tokenizer_fn: a function which converts text into list of ids (List of List)



        """
        self.model = model
        self.model_fn = None
        self.input_type_ids = input_type_ids
        self.input_mask_ids = input_mask_ids
        # keras Model
        if isinstance(model, tf.keras.Model):
            self.model_fn = self.model
            model_config = model.model_config
            (
                self.embedding_size,
                self.num_attention_heads,
                self.num_hidden_layers,
                self.attention_state,
            ) = self.auto_infer_config(model_config, saved_model=False)

            # Validate decoder type ids are there
            self.validate_decoder_type_ids(model.input)

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

            model_config = self.model.config
            (
                self.embedding_size,
                self.num_attention_heads,
                self.num_hidden_layers,
                self.attention_state,
            ) = self.auto_infer_config(model_config, saved_model=True)

            # Validate decoder type ids are there
            self.validate_decoder_type_ids(self.model_pb.structured_input_signature[1])

        if self.model_fn is None:
            raise ValueError("Please check the type of your model")
        self.reserved_input_keys = ["input_ids", "input_mask", "input_type_ids"]

    def validate_decoder_type_ids(self, inputs):
        if "input_type_ids" in inputs:
            if self.input_type_ids < 0:
                raise ValueError(
                    "Seems like you model has `input_type_ids`, \
                        but it hasn't set yet. Please provide a valid positive index for `decoder_input_type_ids`"
                )
        if "input_mask" in inputs:
            if self.input_mask_ids < 0:
                raise ValueError(
                    "Seems like you model has `input_mask_ids`, \
                        but it hasn't set yet. Please provide a valid positive index for `decoder_input_type_ids`"
                )

    def auto_infer_config(self, config, saved_model=False):
        """[summary]

        Args:
            config ([type]): [description]

        Returns:
            [type]: [description]
        """
        if saved_model:
            embedding_size = config["embedding_size"].numpy()
            num_attention_heads = config["num_attention_heads"].numpy()
            num_hidden_layers = config["num_hidden_layers"].numpy()
            attention_head_size = config["attention_head_size"].numpy()
            return (embedding_size, num_attention_heads, num_hidden_layers, attention_head_size)
        else:
            embedding_size = config["embedding_size"]
            num_attention_heads = config["num_attention_heads"]
            num_hidden_layers = config["num_hidden_layers"]
            attention_head_size = config["attention_head_size"]
            return (embedding_size, num_attention_heads, num_hidden_layers, attention_head_size)

    def reorder_past_batches(self, all_cache_key, all_cache_value, coordinates, beam_size):
        """
        Reorder the input batch based on beam predictions
        Future beams changes the best path order
        """
        coordinates_reshaped = coordinates[:, :beam_size, -1] + tf.expand_dims(
            tf.range(tf.shape(coordinates)[0]) * beam_size, 1
        )
        coordinates_reshaped = tf.reshape(coordinates_reshaped, -1)
        all_cache_key = tf.gather(all_cache_key, coordinates_reshaped, axis=1)
        all_cache_value = tf.gather(all_cache_value, coordinates_reshaped, axis=1)
        return all_cache_key, all_cache_value

    # Greedy Decoding
    def greedy_decode(
        self, tokenized_input_dict_original, max_iterations, do_sample=False, temperature=1.0, eos_id=-100
    ):
        """
        Greedy Decoding

        tokenized_input_dict: inputs
        max_iterations: 50 (decodingmax_iterations)
        """
        tokenized_input_dict = tokenized_input_dict_original.copy()
        # We need this to return
        input_ids_original = tokenized_input_dict["input_ids"]
        batch_size = tf.shape(input_ids_original)[0]
        max_sequence_length = tf.shape(input_ids_original)[1]

        # We need this in assign_K_V_zeros function
        input_ids_copy = tf.identity(tokenized_input_dict["input_ids"])

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
        tokenized_input_dict["all_cache_key"] = all_cache_key
        tokenized_input_dict["all_cache_value"] = all_cache_value
        tokenized_input_dict["past_length"] = past_length

        all_predictions = []
        all_prediction_probs = []
        matched_positions = tf.constant([-1] * batch_size)
        # Iterate
        for i in range(max_iterations):

            result = self.model_fn(tokenized_input_dict)
            model_logits = result["last_token_logits"] / temperature
            all_cache_key = result["all_cache_key"]
            all_cache_value = result["all_cache_value"]
            past_length = result["past_length"]

            if do_sample:
                prediction_ids = tf.random.categorical(model_logits, num_samples=1)
                input_ids = tf.cast(prediction_ids, tf.int32)
            else:
                prediction_ids = tf.argmax(model_logits, axis=1)
                prediction_probs = tf.reduce_max(model_logits, axis=1)
                input_ids = tf.expand_dims(prediction_ids, axis=1)
                all_prediction_probs.append(prediction_probs)

            all_predictions.append(input_ids)

            if eos_id:
                temp_m = tf.concat(all_predictions, axis=1)
                eos_check = tf.greater(
                    tf.reduce_prod(tf.reduce_sum(tf.cast(tf.equal(temp_m, eos_id), tf.int32), axis=[1])),
                    0,
                )
                if eos_check:
                    matched_positions = tf.argmax(tf.cast(tf.equal(eos_id, temp_m), tf.int32), axis=1)
                    # matched_positions += 1
                    break

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

            tokenized_input_dict["input_ids"] = tf.cast(input_ids, tf.int32)
            tokenized_input_dict["all_cache_key"] = all_cache_key
            tokenized_input_dict["all_cache_value"] = all_cache_value
            tokenized_input_dict["past_length"] = past_length

            if self.input_type_ids > -1:
                tokenized_input_dict["input_type_ids"] = (
                    tf.ones_like(tokenized_input_dict["input_ids"]) * self.input_type_ids
                )
            if self.input_mask_ids > -1:
                tokenized_input_dict["input_mask"] = (
                    tf.ones_like(tokenized_input_dict["input_ids"]) * self.input_mask_ids
                )

        all_predictions = tf.concat(all_predictions, axis=1)
        matched_positions = tf.reshape(tf.argmax(tf.cast(tf.equal(eos_id, all_predictions), tf.int32), axis=1), -1)
        # no eos matched positions will be 0, replace with -1
        eos_pos_mask = tf.cast(tf.equal(matched_positions, 0), tf.int32) * -1
        matched_positions = tf.cast(matched_positions, tf.int32) + eos_pos_mask

        all_predictions = tf.cast(tf.expand_dims(all_predictions, axis=1), tf.int32)
        return {
            "iterations": i + 1,
            "input_ids": input_ids_original,
            "predicted_ids": all_predictions,
            "matched_eos_pos": matched_positions,
            "prediction_probs": all_prediction_probs,
        }

    def beam_decode(
        self,
        tokenized_input_dict,
        beam_size,
        max_iterations,
        temperature=1.0,
        alpha=0.0,
        top_k=0,
        top_p=0,
        do_sample=False,
        eos_id=-100,
    ):

        """Supports Variable Batch Decoding for GPT2

         text_list: a list of text
         beam_size: int
        max_iterations: number of steps to decode
         vocab_size: vocabulary size
         do_sample: Using multinomial distribution to sample the most likely \
         word, still uses beam
         eos_ids: list of IDS, to consider as decoder stop
        """

        # We need this to return
        input_ids_original = tokenized_input_dict["input_ids"]
        batch_size = tf.shape(input_ids_original)[0]
        max_sequence_length = tf.shape(input_ids_original)[1]

        # We need this in assign_K_V_zeros function, we should place ot before repeat
        # input_ids_copy = tf.identity(tokenized_input_dict_ragged['input_ids'])

        # Repeat for beam search
        tokenized_input_dict_ragged = {}
        for input_key, input_value in tokenized_input_dict.items():
            tokenized_input_dict_ragged[input_key] = tf.repeat(input_value, [beam_size], axis=0)

        # We take 2x beams
        beams_to_keep = 2 * beam_size
        batch_size_updated = tokenized_input_dict_ragged["input_ids"].shape[0]

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
        tokenized_input_dict_ragged["all_cache_key"] = all_cache_key
        tokenized_input_dict_ragged["all_cache_value"] = all_cache_value
        tokenized_input_dict_ragged["past_length"] = past_length

        if self.input_type_ids > -1:
            tokenized_input_dict_ragged["input_type_ids"] = (
                tf.ones_like(tokenized_input_dict_ragged["input_ids"]) * self.input_type_ids
            )
        if self.input_mask_ids > -1:
            tokenized_input_dict_ragged["input_mask"] = (
                tf.ones_like(tokenized_input_dict_ragged["input_ids"]) * self.input_mask_ids
            )

        alive_log_probs = -np.inf * tf.ones((batch_size, beam_size - 1))

        alive_log_probs = tf.concat([tf.zeros([batch_size, 1]), alive_log_probs], axis=1)
        alive_seq = tf.zeros((batch_size, beam_size, 1))

        matched_positions = tf.constant([-1] * batch_size_updated)

        # Iterate
        for i in range(max_iterations):
            result = self.model_fn(tokenized_input_dict_ragged)

            model_logits = result["last_token_logits"] / temperature
            all_cache_key = result["all_cache_key"]
            all_cache_value = result["all_cache_value"]
            past_length = result["past_length"]

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

                # # sort the sampled vector to make sure that the first \
                # num_beams samples are the best
                next_scores_indices = tf.argsort(next_scores, direction="DESCENDING", axis=1)
                next_scores = tf.gather(next_scores, next_scores_indices, batch_dims=1)  # (batch_size, num_beams * 2)
                next_tokens = tf.gather(next_tokens, next_scores_indices, batch_dims=1)  # (batch_size, num_beams * 2)

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
                    tf.not_equal(tokenized_input_dict_ragged["input_ids"], -1),
                    tf.float32,
                )
                masks = tf.reshape(masks, (1, batch_size_updated, 1, max_sequence_length, 1))
                all_cache_key = all_cache_key * masks
                all_cache_value = all_cache_value * masks

            all_cache_key, all_cache_value = self.reorder_past_batches(
                all_cache_key, all_cache_value, coordinates, beam_size
            )
            # current_ids = topk_alive_seq[:,:,-1]
            tokenized_input_dict_ragged["input_ids"] = tf.cast(input_ids, tf.int32)
            tokenized_input_dict_ragged["all_cache_key"] = all_cache_key
            tokenized_input_dict_ragged["all_cache_value"] = all_cache_value
            tokenized_input_dict_ragged["past_length"] = past_length

            if self.input_type_ids > -1:
                tokenized_input_dict_ragged["input_type_ids"] = (
                    tf.ones_like(tokenized_input_dict_ragged["input_ids"]) * self.input_type_ids
                )
            if self.input_mask_ids > -1:
                tokenized_input_dict_ragged["input_mask"] = (
                    tf.ones_like(tokenized_input_dict_ragged["input_ids"]) * self.input_mask_ids
                )

            if eos_id:
                eos_check = tf.greater(
                    tf.reduce_prod(
                        tf.reduce_sum(
                            tf.cast(tf.equal(topk_alive_seq, eos_id), tf.int32),
                            axis=[2],
                        )
                    ),
                    0,
                )
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
                    # matched_positions += 1
                    break
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

        return {
            "iterations": i + 1,
            "input_ids": input_ids_original,
            "predicted_ids": topk_alive_seq[:, :, 1:],  # to avoid initial 0
            "matched_eos_pos": matched_positions,
        }

    def top_k_top_p(
        self,
        tokenized_input_dict,
        max_iterations,
        top_k=0,
        top_p=0,
        temperature=1.0,
        do_sample=True,
        num_return_sequences=1,
        eos_id=-100,
    ):

        # We need this to return
        input_ids_original = tokenized_input_dict["input_ids"]
        batch_size = tf.shape(input_ids_original)[0]
        max_sequence_length = tf.shape(input_ids_original)[1]

        # Repeat for beam search
        tokenized_input_dict_ragged = {}
        for input_key, input_value in tokenized_input_dict.items():
            tokenized_input_dict_ragged[input_key] = tf.repeat(input_value, [num_return_sequences], axis=0)

        # We take 2x beams
        batch_size_updated = tokenized_input_dict_ragged["input_ids"].shape[0]

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
        tokenized_input_dict_ragged["all_cache_key"] = all_cache_key
        tokenized_input_dict_ragged["all_cache_value"] = all_cache_value
        tokenized_input_dict_ragged["past_length"] = past_length

        if self.input_type_ids > -1:
            tokenized_input_dict_ragged["input_type_ids"] = (
                tf.ones_like(tokenized_input_dict_ragged["input_ids"]) * self.input_type_ids
            )
        if self.input_mask_ids > -1:
            tokenized_input_dict_ragged["input_mask"] = (
                tf.ones_like(tokenized_input_dict_ragged["input_ids"]) * self.input_mask_ids
            )

        all_predictions = []
        matched_positions = tf.constant([-1] * batch_size_updated)

        # Iterate Over
        for i in range(max_iterations):
            result = self.model_fn(tokenized_input_dict_ragged)
            model_logits = result["last_token_logits"]
            all_cache_key = result["all_cache_key"]
            all_cache_value = result["all_cache_value"]
            past_length = result["past_length"]

            if top_k > 0:
                model_logits = top_k_logits(model_logits, k=top_k)
            if top_p > 0:
                model_logits = top_p_logits(model_logits, p=top_p)

            if do_sample:
                prediction_ids = tf.random.categorical(model_logits, num_samples=1)
                input_ids = tf.cast(prediction_ids, tf.int32)
            else:
                prediction_ids = tf.argmax(model_logits, axis=1)
                input_ids = tf.cast(tf.expand_dims(prediction_ids, axis=1), tf.int32)
            all_predictions.append(input_ids)
            if i == 0:
                # all_cache_key = assign_zeros_to_K_V(all_cache_key, \
                # input_ids_copy, batch_size, max_sequence_length)
                # all_cache_value = assign_zeros_to_K_V(all_cache_value, \
                # input_ids_copy, batch_size, max_sequence_length)

                masks = tf.cast(
                    tf.not_equal(tokenized_input_dict_ragged["input_ids"], -1),
                    tf.float32,
                )
                masks = tf.reshape(masks, (1, batch_size_updated, 1, max_sequence_length, 1))
                all_cache_key = all_cache_key * masks
                all_cache_value = all_cache_value * masks

            tokenized_input_dict_ragged["input_ids"] = tf.cast(input_ids, tf.int32)
            tokenized_input_dict_ragged["all_cache_key"] = all_cache_key
            tokenized_input_dict_ragged["all_cache_value"] = all_cache_value
            tokenized_input_dict_ragged["past_length"] = past_length

            if self.input_type_ids > -1:
                tokenized_input_dict_ragged["input_type_ids"] = (
                    tf.ones_like(tokenized_input_dict_ragged["input_ids"]) * self.input_type_ids
                )
            if self.input_mask_ids > -1:
                tokenized_input_dict_ragged["input_mask"] = (
                    tf.ones_like(tokenized_input_dict_ragged["input_ids"]) * self.input_mask_ids
                )

            if eos_id:
                temp_m = tf.concat(all_predictions, axis=1)
                eos_check = tf.greater(
                    tf.reduce_prod(tf.reduce_sum(tf.cast(tf.equal(temp_m, eos_id), tf.int32), axis=[1])),
                    0,
                )
                if eos_check:
                    matched_positions = tf.argmax(tf.cast(tf.equal(eos_id, temp_m), tf.int32), axis=1)
                    # matched_positions += 1
                    break

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
        return {
            "iterations": i + 1,
            "input_ids": input_ids_original,
            "predicted_ids": all_predictions,
            "matched_eos_pos": matched_positions,
        }

    def decode(
        self,
        tokenized_input_dict,
        max_iterations,
        beam_size=3,
        num_return_sequences=1,
        sampling_temperature=1.0,
        alpha=0.0,
        eos_id=-100,
        top_k=0,
        top_p=0,
        do_sample=False,
        mode="greedy",
    ):
        if mode == "greedy":
            result = self.greedy_decode(
                tokenized_input_dict,
                max_iterations=max_iterations,
                temperature=sampling_temperature,
                do_sample=do_sample,
                eos_id=eos_id,
            )
        if mode == "beam":
            result = self.beam_decode(
                tokenized_input_dict,
                max_iterations=max_iterations,
                beam_size=beam_size,
                temperature=sampling_temperature,
                alpha=alpha,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                eos_id=eos_id,
            )
        if mode == "top_k_top_p":
            result = self.top_k_top_p(
                tokenized_input_dict,
                max_iterations=max_iterations,
                temperature=sampling_temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                eos_id=eos_id,
            )
        return result
