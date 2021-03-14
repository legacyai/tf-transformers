import numpy as np
import tensorflow as tf

from tf_transformers.text import (_gather_beams, _log_prob_from_logits,
                                  assign_zeros_to_K_V, top_k_logits,
                                  top_p_logits)

tf.keras.backend.clear_session()


class TextDecoderSerializableSeq2Seq(tf.keras.layers.Layer):
    """TextDecoderSerializable - This class is responsible for saving
    the model along with decoding
    operation as a saved_model, which makes deployment in production easier.
    """

    def __init__(
        self,
        model,
        decoder_start_token_id,
        mode,
        max_iterations=None,
        batch_size=None,
        sequence_length=None,
        max_sequence_length=None,
        beam_size=1,
        eos_id=-100,
        sampling_temperature=1.0,
        alpha=0.0,
        do_sample=False,
        top_k=0,
        top_p=0,
        num_return_sequences=1,
        decoder_input_type_ids=-1,
    ):
        """[Init]

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

        super(TextDecoderSerializableSeq2Seq, self).__init__()

        self.max_iterations = max_iterations
        self.decoder_start_token_id = decoder_start_token_id

        self.batch_size = batch_size
        self.sequence_length = sequence_length

        self.model = model

        decoder_config = model.model_config["decoder"]
        (
            self.embedding_size,
            self.decoder_num_attention_heads,
            self.decoder_num_hidden_layers,
            self.decoder_attention_state,
        ) = self.auto_infer_config(decoder_config)

        self.decoder_input_type_ids = decoder_input_type_ids
        # Validate decoder type ids are there
        self.validate_decoder_type_ids(model.input)

        (
            self.decoder_input_name_list,
            self.input_name_list,
            self.model_inputs,
        ) = self.get_inputs()

        self.input_name_map = {i: k for i, k in enumerate(self.input_name_list)}
        self.decoder_input_name_map = {i: k for i, k in enumerate(self.decoder_input_name_list)}

        self.eos_id = eos_id
        self.mode = mode

        self.beam_size = beam_size
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample
        self.num_return_sequences = num_return_sequences

        self.temperature = sampling_temperature
        self.alpha = alpha

        if self.mode == "greedy":
            self.decoder_fn = self.greedy()
        elif self.mode == "beam":
            self.decoder_fn = self.beam()
        elif self.mode == "top_k_top_p":
            self.decoder_fn = self.top_k_top()

    def get_model(self):
        # Call the model in init itself
        layer_outputs = self(self.model_inputs)
        decoder_model = tf.keras.Model(inputs=self.model_inputs, outputs=layer_outputs, name="decoder_model")
        return decoder_model

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
        return (embedding_size, decoder_num_attention_heads, decoder_num_hidden_layers, attention_head_size)

    def get_inputs(self):

        input_ids = tf.keras.layers.Input(
            shape=(self.sequence_length,),
            batch_size=self.batch_size,
            ragged=False,
            dtype=tf.int32,
            name="input_ids",
        )
        input_mask = tf.keras.layers.Input(
            shape=(self.sequence_length,),
            batch_size=self.batch_size,
            ragged=False,
            dtype=tf.int32,
            name="input_mask",
        )
        input_type_ids = tf.keras.layers.Input(
            shape=(self.sequence_length,),
            batch_size=self.batch_size,
            ragged=False,
            dtype=tf.int32,
            name="input_type_ids",
        )
        self.input_name_list = []
        if "encoder_input_ids" in self.model.input:
            self.input_name_list.append("encoder_input_ids")
        if "encoder_input_mask" in self.model.input:
            self.input_name_list.append("encoder_input_mask")
        if "encoder_input_type_ids" in self.model.input:
            self.input_name_list.append("encoder_input_type_ids")
        inputs = {}
        for name in self.input_name_list:
            if name == "encoder_input_ids":
                inputs["encoder_input_ids"] = input_ids
                continue
            if name == "encoder_input_mask":
                inputs["encoder_input_mask"] = input_mask
            if name == "encoder_input_type_ids":
                inputs["encoder_input_type_ids"] = input_type_ids

        self.decoder_input_name_list = ["decoder_input_ids"]
        if "decoder_input_type_ids" in self.model.input:
            self.decoder_input_name_list.append("decoder_input_type_ids")

        if self.max_iterations is None:
            inputs["iterations"] = tf.keras.layers.Input(
                shape=(1,), batch_size=1, ragged=False, dtype=tf.int32, name="iterator"
            )

        return self.decoder_input_name_list, self.input_name_list, inputs

    def reorder_past_batches(self, all_cache_key, all_cache_value, coordinates, beam_size):
        """[Reorder the input batch based on beam predictions
        Future beams changes the best path order]

        Args:
            all_cache_key ([tf.tensor]): [K from Transformers]
            all_cache_value ([tf.tensor]): [V from Transformers]
            coordinates ([tf.tensor (bach_size x beam_size)]): [The order ]
            beam_size ([int/tf.tensor]): [Number of beams]

        Returns:
            [type]: [description]

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

    def greedy(self):
        """
        This function will perform greedy decoding.
        """

        # EOS check function
        def cond(
            i,
            encoder_inputs_tuple,
            decoder_inputs_tuple,
            encoder_hidden_states,
            all_cache_key,
            all_cache_value,
            decoded_ids,
        ):
            eos_check = tf.greater(
                tf.reduce_prod(tf.reduce_sum(tf.cast(tf.equal(decoded_ids, self.eos_id), tf.int32), axis=[1])),
                0,
            )
            return tf.not_equal(eos_check, True)

        def body(
            i,
            encoder_inputs_tuple,
            decoder_inputs_tuple,
            encoder_hidden_states,
            all_cache_key,
            all_cache_value,
            decoded_ids,
        ):

            """[This is the body of the beam decoder]

            Args:
                i ([tf.tensor]): [iterator (an int)]
                inputs ([List of model inputs]): [description]
                all_cache_key ([K]): [description]
                all_cache_value ([V]): [description]
                past_length ([tf.tensor (1 x batch_size)]): [description]
                This is our main output or decoded ids]
                alive_log_probs ([tf.tensor]): [To keep track of active ids]
                alive_seq ([tf.tensor]): [description]

            Returns:
                [List of tensors]: [Outputs]
            """
            inputs = {}
            for k in range(len(self.input_name_list)):
                inputs[self.input_name_list[k]] = encoder_inputs_tuple[k]

            for k in range(len(self.decoder_input_name_list)):
                inputs[self.decoder_input_name_list[k]] = decoder_inputs_tuple[k]

            inputs["encoder_hidden_states"] = encoder_hidden_states
            inputs["decoder_all_cache_key"] = all_cache_key
            inputs["decoder_all_cache_value"] = all_cache_value

            model_outputs = self.model(inputs)
            model_logits = model_outputs["last_token_logits"]
            model_logits = model_logits / self.temperature

            if self.do_sample:
                prediction_ids = tf.random.categorical(model_logits, num_samples=1)
                input_ids = tf.cast(prediction_ids, tf.int32)
            else:
                prediction_ids = tf.argmax(model_logits, axis=1)
                input_ids = tf.cast(tf.expand_dims(prediction_ids, axis=1), tf.int32)

            decoder_inputs_tuple = [None] * len(self.decoder_input_name_list)

            for index, name in self.decoder_input_name_map.items():
                if name == "decoder_input_ids":
                    decoder_inputs_tuple[index] = input_ids
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue
                if name == "decoder_input_type_ids":
                    decoder_inputs_tuple[index] = tf.ones_like(input_ids) * self.decoder_input_type_ids
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue
            # Convert to tuple
            decoder_inputs_tuple = tuple(decoder_inputs_tuple)

            return [
                i + 1,
                encoder_inputs_tuple,
                decoder_inputs_tuple,
                model_outputs["encoder_hidden_states"],
                model_outputs["decoder_all_cache_key"],
                model_outputs["decoder_all_cache_value"],
                tf.concat([decoded_ids, input_ids], axis=1),
            ]

        # @tf.function(experimental_relax_shapes=True)
        def call_greedy(inputs):
            encoder_inputs_copy = inputs.copy()
            input_ids_orig = inputs["encoder_input_ids"]
            # Original batch size and sequence length
            batch_size = tf.shape(inputs["encoder_input_ids"])[0]
            # Initialize with zeros
            encoder_sequence_length = tf.shape(inputs["encoder_input_ids"])[1]
            decoder_start_sequence_length = 1

            if self.max_iterations is None:
                iterations = tf.squeeze(inputs["iterations"])
            else:
                iterations = self.max_iterations

            model_inputs = {}
            for input_key, input_value in inputs.items():
                # We dont want iterations in model_inputs
                if input_key == "iterations":
                    continue
                model_inputs[input_key] = input_value
            # Prepare inputs
            # Encoder hidden states
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

            # Prepare Decoder inputs
            decoder_input_ids = tf.cast(tf.ones(shape=(batch_size, 1)) * self.decoder_start_token_id, tf.int32)

            # Iterator to keep track of the loop
            i = tf.constant([[0]])

            # Add remaining model inputs
            model_inputs["decoder_all_cache_key"] = all_cache_key
            model_inputs["decoder_all_cache_value"] = all_cache_value
            model_inputs["encoder_hidden_states"] = encoder_hidden_states
            model_inputs["decoder_input_ids"] = decoder_input_ids

            if "decoder_input_type_ids" in self.decoder_input_name_list:
                model_inputs["decoder_input_type_ids"] = (
                    tf.ones_like(model_inputs["decoder_input_ids"]) * self.decoder_input_type_ids
                )

            model_outputs = self.model(model_inputs)
            model_logits = model_outputs["last_token_logits"]
            model_logits = model_logits / self.temperature

            if self.do_sample:
                prediction_ids = tf.random.categorical(model_logits, num_samples=1)
                input_ids = tf.cast(prediction_ids, tf.int32)
            else:
                prediction_ids = tf.argmax(model_logits, axis=1)
                input_ids = tf.cast(tf.expand_dims(prediction_ids, axis=1), tf.int32)

            # Update iter
            i = i + 1
            decoded_ids = tf.concat([decoder_input_ids, input_ids], axis=1)

            # Even though encoder values are not using , we need it as per
            # models serialized versions
            encoder_inputs_tuple = [None] * len(self.input_name_list)
            encoder_input_shapes_tuple = [tf.TensorShape([None, None])] * len(self.input_name_list)
            for index, name in self.input_name_map.items():
                if name == "encoder_input_ids":
                    encoder_inputs_tuple[index] = encoder_inputs_copy["encoder_input_ids"]
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue
                if name == "encoder_input_type_ids":
                    encoder_inputs_tuple[index] = encoder_inputs_copy["encoder_input_type_ids"]
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue
                if name == "encoder_input_mask":
                    encoder_inputs_tuple[index] = encoder_inputs_copy["encoder_input_mask"]
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue

            encoder_inputs_tuple = tuple(encoder_inputs_tuple)
            encoder_input_shapes_tuple = tuple(encoder_input_shapes_tuple)

            # END
            decoder_inputs_tuple = [None] * len(self.decoder_input_name_list)
            decoder_input_shapes_tuple = [tf.TensorShape([None, None])] * len(self.decoder_input_name_list)
            for index, name in self.decoder_input_name_map.items():
                if name == "decoder_input_ids":
                    decoder_inputs_tuple[index] = input_ids
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue
                if name == "decoder_input_type_ids":
                    decoder_inputs_tuple[index] = tf.ones_like(input_ids) * self.decoder_input_type_ids

            decoder_inputs_tuple = tuple(decoder_inputs_tuple)
            decoder_input_shapes_tuple = tuple(decoder_input_shapes_tuple)

            results = tf.while_loop(
                cond,
                body,
                maximum_iterations=iterations - 1,
                loop_vars=[
                    i,
                    encoder_inputs_tuple,
                    decoder_inputs_tuple,
                    model_outputs["encoder_hidden_states"],
                    model_outputs["decoder_all_cache_key"],
                    model_outputs["decoder_all_cache_value"],
                    decoded_ids,
                ],
                shape_invariants=[
                    i.get_shape(),
                    encoder_input_shapes_tuple,
                    decoder_input_shapes_tuple,
                    tf.TensorShape([None, None, None]),
                    tf.TensorShape(
                        [
                            self.decoder_num_hidden_layers,
                            None,
                            self.decoder_num_attention_heads,
                            None,
                            self.decoder_attention_state,
                        ]
                    ),
                    tf.TensorShape(
                        [
                            self.decoder_num_hidden_layers,
                            None,
                            self.decoder_num_attention_heads,
                            None,
                            self.decoder_attention_state,
                        ]
                    ),
                    tf.TensorShape([None, None]),
                ],
            )

            results_dict = {}
            results_dict["iterations"] = results[0]
            results_dict["input_ids"] = input_ids_orig
            # Skip -1 initial ids
            results_dict["predicted_ids"] = results[-1][:, 1:]
            # Add matched positions here
            matched_positions = tf.argmax(
                tf.cast(tf.equal(self.eos_id, results_dict["predicted_ids"]), tf.int64),
                axis=1,
            )
            # no eos matched positions will be 0, replace with -1
            eos_pos_mask = tf.cast(tf.equal(matched_positions, 0), tf.int64) * -1
            matched_positions = tf.cast(matched_positions, tf.int64) + eos_pos_mask
            results_dict["matched_eos_pos"] = tf.cast(matched_positions, tf.int32)
            results_dict["predicted_ids"] = tf.cast(tf.expand_dims(results[-1][:, 1:], 1), tf.int32)

            return results_dict

        return call_greedy

    def beam(self):
        """
        This function will perform beam decoding.
        """

        # EOS check function
        def cond(
            i,
            encoder_inputs_tuple,
            decoder_inputs_tuple,
            encoder_hidden_states,
            all_cache_key,
            all_cache_value,
            alive_log_probs,
            alive_seq,
        ):
            eos_check = tf.greater(
                tf.reduce_prod(tf.reduce_sum(tf.cast(tf.equal(alive_seq, self.eos_id), tf.int32), axis=[2])),
                0,
            )
            return tf.not_equal(eos_check, True)

        def body(
            i,
            encoder_inputs_tuple,
            decoder_inputs_tuple,
            encoder_hidden_states,
            all_cache_key,
            all_cache_value,
            alive_log_probs,
            alive_seq,
        ):
            """[This is the body of the beam decoder]

            Args:
                i ([tf.tensor]): [iterator (an int)]
                inputs ([List of model inputs]): [description]
                all_cache_key ([K]): [description]
                all_cache_value ([V]): [description]
                past_length ([tf.tensor (1 x batch_size)]): [description]
                This is our main output or decoded ids]
                alive_log_probs ([tf.tensor]): [To keep track of active ids]
                alive_seq ([tf.tensor]): [description]

            Returns:
                [List of tensors]: [Outputs]
            """
            inputs = {}
            for k in range(len(self.input_name_list)):
                inputs[self.input_name_list[k]] = encoder_inputs_tuple[k]

            for k in range(len(self.decoder_input_name_list)):
                inputs[self.decoder_input_name_list[k]] = decoder_inputs_tuple[k]

            inputs["encoder_hidden_states"] = encoder_hidden_states
            inputs["decoder_all_cache_key"] = all_cache_key
            inputs["decoder_all_cache_value"] = all_cache_value

            beams_to_keep = 2 * self.beam_size

            model_outputs = self.model(inputs)

            model_logits = model_outputs["last_token_logits"]
            model_logits = model_logits / self.temperature

            all_cache_key = model_outputs["decoder_all_cache_key"]
            all_cache_value = model_outputs["decoder_all_cache_value"]

            if self.top_k > 0:
                model_logits = top_k_logits(model_logits, k=self.top_k)
            if self.top_p > 0:
                model_logits = top_p_logits(model_logits, p=self.top_p)

            vocab_size = tf.shape(model_logits)[1]
            batch_size = tf.shape(inputs["encoder_input_ids"])[0] // self.beam_size
            logits = tf.reshape(model_logits, (batch_size, self.beam_size, -1))
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
            flat_log_probs = tf.reshape(log_probs, [-1, self.beam_size * vocab_size])

            if self.do_sample:
                next_tokens = tf.random.categorical(
                    flat_log_probs, dtype=tf.int32, num_samples=beams_to_keep
                )  # (batch_size, 2 * num_beams)

                # # Compute next scores
                next_scores = tf.gather(flat_log_probs, next_tokens, batch_dims=1)  # (batch_size, 2 * num_beams)

                # # sort the sampled vector to make sure that the first num_beams
                #  samples are the best
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

            topk_alive_seq = topk_seq[:, : self.beam_size, :]
            alive_log_probs = topk_log_probs[:, : self.beam_size]
            input_ids = tf.reshape(topk_ids[:, : self.beam_size], [-1, 1])
            alive_seq = topk_alive_seq

            all_cache_key, all_cache_value = self.reorder_past_batches(
                all_cache_key, all_cache_value, coordinates, self.beam_size
            )
            model_outputs["decoder_all_cache_key"] = all_cache_key
            model_outputs["decoder_all_cache_value"] = all_cache_value

            decoder_inputs_tuple = [None] * len(self.decoder_input_name_list)

            for index, name in self.decoder_input_name_map.items():
                if name == "decoder_input_ids":
                    decoder_inputs_tuple[index] = input_ids
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue
                if name == "decoder_input_type_ids":
                    decoder_inputs_tuple[index] = tf.ones_like(input_ids) * self.decoder_input_type_ids
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue
            # Convert to tuple
            decoder_inputs_tuple = tuple(decoder_inputs_tuple)

            return [
                i + 1,
                encoder_inputs_tuple,
                decoder_inputs_tuple,
                model_outputs["encoder_hidden_states"],
                model_outputs["decoder_all_cache_key"],
                model_outputs["decoder_all_cache_value"],
                alive_log_probs,
                alive_seq,
            ]

        # @tf.function(experimental_relax_shapes=True)
        def call_beam(inputs):
            """The main function to perform beam search
            Args:
                inputs ([dict]): [dict of tf.tensors (model inputs)]
            """

            # We take 2x beams
            beams_to_keep = 2 * self.beam_size

            input_ids_orig = inputs["encoder_input_ids"]
            # Original batch size and sequence length
            batch_size = tf.shape(inputs["encoder_input_ids"])[0]
            # Initialize with zeros
            encoder_sequence_length = tf.shape(inputs["encoder_input_ids"])[1]
            decoder_start_sequence_length = 1

            model_inputs = {}
            for input_key, input_value in inputs.items():
                # We dont want iterations in model_inputs
                if input_key == "iterations":
                    continue
                model_inputs[input_key] = tf.repeat(input_value, [self.beam_size], axis=0)

            encoder_inputs_copy = model_inputs.copy()
            # New batch size
            batch_size_updated = tf.shape(model_inputs["encoder_input_ids"])[0]
            # Prepare inputs
            # Encoder hidden states
            encoder_hidden_states = tf.zeros((batch_size_updated, encoder_sequence_length, self.embedding_size))
            all_cache_key = tf.zeros(
                (
                    self.decoder_num_hidden_layers,
                    batch_size_updated,
                    self.decoder_num_attention_heads,
                    decoder_start_sequence_length,
                    self.decoder_attention_state,
                )
            )
            all_cache_value = tf.zeros(
                (
                    self.decoder_num_hidden_layers,
                    batch_size_updated,
                    self.decoder_num_attention_heads,
                    decoder_start_sequence_length,
                    self.decoder_attention_state,
                )
            )

            # Prepare Decoder inputs
            decoder_input_ids = tf.cast(
                tf.ones(shape=(batch_size_updated, 1)) * self.decoder_start_token_id,
                tf.int32,
            )

            i = tf.constant([[0]])

            if self.max_iterations is None:
                iterations = tf.squeeze(inputs["iterations"])
            else:
                iterations = self.max_iterations

            # Add remaining model inputs
            # Add remaining model inputs
            model_inputs["decoder_all_cache_key"] = all_cache_key
            model_inputs["decoder_all_cache_value"] = all_cache_value
            model_inputs["encoder_hidden_states"] = encoder_hidden_states
            model_inputs["decoder_input_ids"] = decoder_input_ids

            if "decoder_input_type_ids" in self.decoder_input_name_list:
                model_inputs["decoder_input_type_ids"] = (
                    tf.ones_like(model_inputs["decoder_input_ids"]) * self.decoder_input_type_ids
                )

            # We need this to re-ordering and keep track of best -log(prob))
            alive_log_probs = -np.inf * tf.ones((batch_size, self.beam_size - 1))
            alive_log_probs = tf.concat([tf.zeros([batch_size, 1]), alive_log_probs], axis=1)
            alive_seq = tf.zeros((batch_size, self.beam_size, 1))

            # First pass to the model
            model_outputs = self.model(model_inputs)
            model_logits = model_outputs["last_token_logits"]
            model_logits = model_logits / self.temperature

            # Update iter
            i = i + 1

            all_cache_key = model_outputs["decoder_all_cache_key"]
            all_cache_value = model_outputs["decoder_all_cache_value"]

            if self.top_k > 0:
                model_logits = top_k_logits(model_logits, k=self.top_k)
            if self.top_p > 0:
                model_logits = top_p_logits(model_logits, p=self.top_p)

            # vocab size
            vocab_size = tf.shape(model_logits)[1]
            logits = tf.reshape(model_logits, (batch_size, self.beam_size, -1))
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
            flat_log_probs = tf.reshape(log_probs, [-1, self.beam_size * vocab_size])

            if self.do_sample:
                next_tokens = tf.random.categorical(
                    flat_log_probs, dtype=tf.int32, num_samples=beams_to_keep
                )  # (batch_size, 2 * num_beams)

                # # Compute next scores
                next_scores = tf.gather(flat_log_probs, next_tokens, batch_dims=1)  # (batch_size, 2 * num_beams)

                # # sort the sampled vector to make sure that the first num_beams
                # samples are the best
                next_scores_indices = tf.argsort(next_scores, direction="DESCENDING", axis=1)
                next_scores = tf.gather(next_scores, next_scores_indices, batch_dims=1)  # (batch_size, num_beams * 2)
                next_tokens = tf.gather(next_tokens, next_scores_indices, batch_dims=1)  # (batch_size, num_beams * 2)

                topk_log_probs = next_scores
                topk_indices = next_tokens
            else:
                topk_log_probs, topk_indices = tf.nn.top_k(flat_log_probs, k=beams_to_keep)

            topk_beam_indices = topk_indices // vocab_size
            topk_seq, coordinates = _gather_beams(alive_seq, topk_beam_indices, batch_size, beams_to_keep)
            topk_seq = tf.cast(topk_seq, tf.int32)
            topk_ids = topk_indices % vocab_size
            topk_seq = tf.concat([topk_seq, tf.expand_dims(topk_ids, axis=2)], axis=2)

            topk_alive_seq = topk_seq[:, : self.beam_size, :]
            alive_log_probs = topk_log_probs[:, : self.beam_size]
            input_ids = tf.reshape(topk_ids[:, : self.beam_size], [-1, 1])
            alive_seq = topk_alive_seq

            # decoded_ids = tf.concat([decoder_input_ids, input_ids], axis=1)

            # Even though encoder values are not using , we need ot as per
            # models serialized versions
            encoder_inputs_tuple = [None] * len(self.input_name_list)
            encoder_input_shapes_tuple = [tf.TensorShape([None, None])] * len(self.input_name_list)
            for index, name in self.input_name_map.items():
                if name == "encoder_input_ids":
                    encoder_inputs_tuple[index] = encoder_inputs_copy["encoder_input_ids"]
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue
                if name == "encoder_input_type_ids":
                    encoder_inputs_tuple[index] = encoder_inputs_copy["encoder_input_type_ids"]
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue
                if name == "encoder_input_mask":
                    encoder_inputs_tuple[index] = encoder_inputs_copy["encoder_input_mask"]
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue

            encoder_inputs_tuple = tuple(encoder_inputs_tuple)
            encoder_input_shapes_tuple = tuple(encoder_input_shapes_tuple)

            # END
            decoder_inputs_tuple = [None] * len(self.decoder_input_name_list)
            decoder_input_shapes_tuple = [tf.TensorShape([None, None])] * len(self.decoder_input_name_list)
            for index, name in self.decoder_input_name_map.items():
                if name == "decoder_input_ids":
                    decoder_inputs_tuple[index] = input_ids
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue
                if name == "decoder_input_type_ids":
                    decoder_inputs_tuple[index] = tf.ones_like(input_ids) * self.decoder_input_type_ids

            decoder_inputs_tuple = tuple(decoder_inputs_tuple)
            decoder_input_shapes_tuple = tuple(decoder_input_shapes_tuple)

            all_cache_key, all_cache_value = self.reorder_past_batches(
                all_cache_key, all_cache_value, coordinates, self.beam_size
            )
            model_outputs["decoder_all_cache_key"] = all_cache_key
            model_outputs["decoder_all_cache_value"] = all_cache_value

            # END
            results = tf.while_loop(
                cond,
                body,
                maximum_iterations=iterations - 1,
                loop_vars=[
                    i,
                    encoder_inputs_tuple,
                    decoder_inputs_tuple,
                    model_outputs["encoder_hidden_states"],
                    model_outputs["decoder_all_cache_key"],
                    model_outputs["decoder_all_cache_value"],
                    alive_log_probs,
                    alive_seq,
                ],
                shape_invariants=[
                    i.get_shape(),
                    encoder_input_shapes_tuple,
                    decoder_input_shapes_tuple,
                    tf.TensorShape([None, None, None]),
                    tf.TensorShape(
                        [
                            self.decoder_num_hidden_layers,
                            None,
                            self.decoder_num_attention_heads,
                            None,
                            self.decoder_attention_state,
                        ]
                    ),
                    tf.TensorShape(
                        [
                            self.decoder_num_hidden_layers,
                            None,
                            self.decoder_num_attention_heads,
                            None,
                            self.decoder_attention_state,
                        ]
                    ),
                    tf.TensorShape([None, None]),
                    tf.TensorShape([None, None, None]),
                ],
            )

            results_dict = {}
            results_dict["iterations"] = results[0]
            results_dict["input_ids"] = input_ids_orig
            # Skip -1 initial ids
            results_dict["predicted_ids"] = tf.cast(results[-1][:, :, 1:], tf.int32)  # to remove initial 0

            matched_positions = (
                tf.squeeze(
                    tf.reshape(
                        tf.argmax(
                            tf.cast(
                                tf.equal(self.eos_id, results_dict["predicted_ids"]),
                                tf.int32,
                            ),
                            axis=2,
                        ),
                        (-1, batch_size * self.beam_size),
                    ),
                    [0],
                )
                - 1
            )
            # no eos matched positions will be 0, replace with -1
            eos_pos_mask = tf.cast(tf.equal(matched_positions, 0), tf.int32) * -1
            matched_positions = tf.cast(matched_positions, tf.int32) + eos_pos_mask
            results_dict["matched_eos_pos"] = tf.cast(matched_positions, tf.int32)

            return results_dict

        return call_beam

    def top_k_top(self):

        # EOS check function
        def cond(
            i,
            encoder_inputs_tuple,
            decoder_inputs_tuple,
            encoder_hidden_states,
            all_cache_key,
            all_cache_value,
            decoded_ids,
        ):
            eos_check = tf.greater(
                tf.reduce_prod(tf.reduce_sum(tf.cast(tf.equal(decoded_ids, self.eos_id), tf.int32), axis=[1])),
                0,
            )
            return tf.not_equal(eos_check, True)

        def body(
            i,
            encoder_inputs_tuple,
            decoder_inputs_tuple,
            encoder_hidden_states,
            all_cache_key,
            all_cache_value,
            decoded_ids,
        ):

            """[This is the body of the beam decoder]

            Args:
                i ([tf.tensor]): [iterator (an int)]
                inputs ([List of model inputs]): [description]
                all_cache_key ([K]): [description]
                all_cache_value ([V]): [description]
                past_length ([tf.tensor (1 x batch_size)]): [description]
                This is our main output or decoded ids]
                alive_log_probs ([tf.tensor]): [To keep track of active ids]
                alive_seq ([tf.tensor]): [description]

            Returns:
                [List of tensors]: [Outputs]
            """
            inputs = {}
            for k in range(len(self.input_name_list)):
                inputs[self.input_name_list[k]] = encoder_inputs_tuple[k]

            for k in range(len(self.decoder_input_name_list)):
                inputs[self.decoder_input_name_list[k]] = decoder_inputs_tuple[k]

            inputs["encoder_hidden_states"] = encoder_hidden_states
            inputs["decoder_all_cache_key"] = all_cache_key
            inputs["decoder_all_cache_value"] = all_cache_value

            model_outputs = self.model(inputs)
            model_logits = model_outputs["last_token_logits"]
            model_logits = model_logits / self.temperature

            if self.top_k > 0:
                model_logits = top_k_logits(model_logits, k=self.top_k)
            if self.top_p > 0:
                model_logits = top_p_logits(model_logits, p=self.top_p)

            if self.do_sample:
                prediction_ids = tf.random.categorical(model_logits, num_samples=1)
                input_ids = tf.cast(prediction_ids, tf.int32)
            else:
                prediction_ids = tf.argmax(model_logits, axis=1)
                input_ids = tf.cast(tf.expand_dims(prediction_ids, axis=1), tf.int32)

            decoder_inputs_tuple = [None] * len(self.decoder_input_name_list)

            for index, name in self.decoder_input_name_map.items():
                if name == "decoder_input_ids":
                    decoder_inputs_tuple[index] = input_ids
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue
                if name == "decoder_input_type_ids":
                    decoder_inputs_tuple[index] = tf.ones_like(input_ids) * self.decoder_input_type_ids
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue
            # Convert to tuple
            decoder_inputs_tuple = tuple(decoder_inputs_tuple)

            return [
                i + 1,
                encoder_inputs_tuple,
                decoder_inputs_tuple,
                model_outputs["encoder_hidden_states"],
                model_outputs["decoder_all_cache_key"],
                model_outputs["decoder_all_cache_value"],
                tf.concat([decoded_ids, input_ids], axis=1),
            ]

        def call_top_k_top_p(inputs):
            """The main function to perform Top K top P (Nucleus) decoding
            Args:
                inputs ([dict]): [dict of tf.tensors (model inputs)]
            """
            input_ids_orig = inputs["encoder_input_ids"]
            # Original batch size and sequence length
            batch_size = tf.shape(inputs["encoder_input_ids"])[0]
            # Initialize with zeros
            encoder_sequence_length = tf.shape(inputs["encoder_input_ids"])[1]
            decoder_start_sequence_length = 1

            if self.max_iterations is None:
                iterations = tf.squeeze(inputs["iterations"])
            else:
                iterations = self.max_iterations

            model_inputs = {}
            for input_key, input_value in inputs.items():
                if input_key == "iterations":
                    continue
                model_inputs[input_key] = tf.repeat(input_value, [self.num_return_sequences], axis=0)

            encoder_inputs_copy = model_inputs.copy()
            # New batch size
            batch_size_updated = tf.shape(model_inputs["encoder_input_ids"])[0]
            # Prepare inputs
            # Encoder hidden states
            encoder_hidden_states = tf.zeros((batch_size_updated, encoder_sequence_length, self.embedding_size))
            all_cache_key = tf.zeros(
                (
                    self.decoder_num_hidden_layers,
                    batch_size_updated,
                    self.decoder_num_attention_heads,
                    decoder_start_sequence_length,
                    self.decoder_attention_state,
                )
            )
            all_cache_value = tf.zeros(
                (
                    self.decoder_num_hidden_layers,
                    batch_size_updated,
                    self.decoder_num_attention_heads,
                    decoder_start_sequence_length,
                    self.decoder_attention_state,
                )
            )

            # Prepare Decoder inputs
            decoder_input_ids = tf.cast(
                tf.ones(shape=(batch_size_updated, 1)) * self.decoder_start_token_id,
                tf.int32,
            )

            i = tf.constant([[0]])

            # Add remaining model inputs
            # Add remaining model inputs
            model_inputs["decoder_all_cache_key"] = all_cache_key
            model_inputs["decoder_all_cache_value"] = all_cache_value
            model_inputs["encoder_hidden_states"] = encoder_hidden_states
            model_inputs["decoder_input_ids"] = decoder_input_ids

            if "decoder_input_type_ids" in self.decoder_input_name_list:
                model_inputs["decoder_input_type_ids"] = (
                    tf.ones_like(model_inputs["decoder_input_ids"]) * self.decoder_input_type_ids
                )

            # First pass to the model
            model_outputs = self.model(model_inputs)
            model_logits = model_outputs["last_token_logits"]
            model_logits = model_logits / self.temperature

            if self.top_k > 0:
                model_logits = top_k_logits(model_logits, k=self.top_k)
            if self.top_p > 0:
                model_logits = top_p_logits(model_logits, p=self.top_p)

            if self.do_sample:
                prediction_ids = tf.random.categorical(model_logits, num_samples=1)
                input_ids = tf.cast(prediction_ids, tf.int32)
            else:
                prediction_ids = tf.argmax(model_logits, axis=1)
                input_ids = tf.cast(tf.expand_dims(prediction_ids, axis=1), tf.int32)

            # Update iter
            i = i + 1
            decoded_ids = tf.concat([decoder_input_ids, input_ids], axis=1)

            # Even though encoder values are not using , we need ot as per models
            #  serialized versions
            encoder_inputs_tuple = [None] * len(self.input_name_list)
            encoder_input_shapes_tuple = [tf.TensorShape([None, None])] * len(self.input_name_list)
            for index, name in self.input_name_map.items():
                if name == "encoder_input_ids":
                    encoder_inputs_tuple[index] = encoder_inputs_copy["encoder_input_ids"]
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue
                if name == "encoder_input_type_ids":
                    encoder_inputs_tuple[index] = encoder_inputs_copy["encoder_input_type_ids"]
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue
                if name == "encoder_input_mask":
                    encoder_inputs_tuple[index] = encoder_inputs_copy["encoder_input_mask"]
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue

            encoder_inputs_tuple = tuple(encoder_inputs_tuple)
            encoder_input_shapes_tuple = tuple(encoder_input_shapes_tuple)

            # END
            decoder_inputs_tuple = [None] * len(self.decoder_input_name_list)
            decoder_input_shapes_tuple = [tf.TensorShape([None, None])] * len(self.decoder_input_name_list)
            for index, name in self.decoder_input_name_map.items():
                if name == "decoder_input_ids":
                    decoder_inputs_tuple[index] = input_ids
                    continue
                if name == "decoder_input_type_ids":
                    decoder_inputs_tuple[index] = tf.ones_like(input_ids) * self.decoder_input_type_ids

            decoder_inputs_tuple = tuple(decoder_inputs_tuple)
            decoder_input_shapes_tuple = tuple(decoder_input_shapes_tuple)

            results = tf.while_loop(
                cond,
                body,
                maximum_iterations=iterations - 1,
                loop_vars=[
                    i,
                    encoder_inputs_tuple,
                    decoder_inputs_tuple,
                    model_outputs["encoder_hidden_states"],
                    model_outputs["decoder_all_cache_key"],
                    model_outputs["decoder_all_cache_value"],
                    decoded_ids,
                ],
                shape_invariants=[
                    i.get_shape(),
                    encoder_input_shapes_tuple,
                    decoder_input_shapes_tuple,
                    tf.TensorShape([None, None, None]),
                    tf.TensorShape(
                        [
                            self.decoder_num_hidden_layers,
                            None,
                            self.decoder_num_attention_heads,
                            None,
                            self.decoder_attention_state,
                        ]
                    ),
                    tf.TensorShape(
                        [
                            self.decoder_num_hidden_layers,
                            None,
                            self.decoder_num_attention_heads,
                            None,
                            self.decoder_attention_state,
                        ]
                    ),
                    tf.TensorShape([None, None]),
                ],
            )

            results_dict = {}
            results_dict["iterations"] = results[0]
            results_dict["input_ids"] = input_ids_orig
            # Skip -1 initial ids
            results_dict["predicted_ids"] = results[-1][:, 1:]
            results_dict["predicted_ids"] = tf.cast(
                tf.reshape(
                    results_dict["predicted_ids"],
                    (batch_size, self.num_return_sequences, -1),
                ),
                tf.int32,
            )

            matched_positions = (
                tf.squeeze(
                    tf.reshape(
                        tf.argmax(
                            tf.cast(
                                tf.equal(self.eos_id, results_dict["predicted_ids"]),
                                tf.int32,
                            ),
                            axis=2,
                        ),
                        (-1, batch_size * self.num_return_sequences),
                    ),
                    [0],
                )
                - 1
            )
            # no eos matched positions will be 0, replace with -1
            eos_pos_mask = tf.cast(tf.equal(matched_positions, 0), tf.int32) * -1
            matched_positions = tf.cast(matched_positions, tf.int32) + eos_pos_mask
            results_dict["matched_eos_pos"] = tf.cast(matched_positions, tf.int32)

            return results_dict

        return call_top_k_top_p

    def call(self, inputs):
        results_dict = self.decoder_fn(inputs)
        return results_dict
