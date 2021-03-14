import numpy as np
import tensorflow as tf

from tf_transformers.text import (_gather_beams, _log_prob_from_logits,
                                  assign_zeros_to_K_V, top_k_logits,
                                  top_p_logits)

tf.keras.backend.clear_session()


class TextDecoderSerializable(tf.keras.layers.Layer):
    """TextDecoderSerializable - This class is responsible for
    saving the model along with decoding
    operation as a saved_model, which makes deployment in production easier.
    """

    def __init__(
        self,
        model,
        mode,
        max_iterations=None,
        batch_size=None,
        sequence_length=None,
        max_sequence_length=None,
        sampling_temperature=1.0,
        alpha=0.0,
        beam_size=1,
        eos_id=-100,
        do_sample=False,
        top_k=0,
        top_p=0,
        num_return_sequences=1,
        input_type_ids=-1,
        input_mask_ids=1,
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

        super(TextDecoderSerializable, self).__init__()

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

        # self.input_name_list = input_name_list
        self.input_name_list, self.model_inputs = self.get_inputs()
        self.input_name_map = {i: k for i, k in enumerate(self.input_name_list)}

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

    def validate_decoder_type_ids(self, inputs):
        if "input_type_ids" in inputs:
            if self.input_type_ids < 0:
                raise ValueError(
                    "Seems like you model has `decoder_input_type_ids`,\
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

    def get_model(self):
        # Call the model in init itself
        layer_outputs = self(self.model_inputs)
        decoder_model = tf.keras.Model(inputs=self.model_inputs, outputs=layer_outputs, name="decoder_model")
        return decoder_model

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
        if "input_ids" in self.model.input:
            self.input_name_list.append("input_ids")
        if "input_mask" in self.model.input:
            self.input_name_list.append("input_mask")
        if "input_type_ids" in self.model.input:
            self.input_name_list.append("input_type_ids")

        inputs = {}
        for name in self.input_name_list:
            if name == "input_ids":
                inputs["input_ids"] = input_ids
                continue
            if name == "input_mask":
                inputs["input_mask"] = input_mask
            if name == "input_type_ids":
                inputs["input_type_ids"] = input_type_ids

        if self.max_iterations is None:
            inputs["iterations"] = tf.keras.layers.Input(
                shape=(1,), batch_size=1, ragged=False, dtype=tf.int32, name="iterator"
            )

        return self.input_name_list, inputs

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
        def cond(i, input_ids, all_cache_key, all_cache_value, past_length, initial_id):
            eos_check = tf.greater(
                tf.reduce_prod(tf.reduce_sum(tf.cast(tf.equal(initial_id, self.eos_id), tf.int32), axis=[1])),
                0,
            )
            return tf.not_equal(eos_check, True)

        def body(i, inputs_tuple, all_cache_key, all_cache_value, past_length, initial_id):

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
                inputs[self.input_name_list[k]] = inputs_tuple[k]

            inputs["all_cache_key"] = all_cache_key
            inputs["all_cache_value"] = all_cache_value
            inputs["past_length"] = past_length

            model_outputs = self.model(inputs)
            model_logits = model_outputs["last_token_logits"]

            all_cache_key = model_outputs["all_cache_key"]
            all_cache_value = model_outputs["all_cache_value"]
            past_length = model_outputs["past_length"]

            if self.do_sample:
                prediction_ids = tf.random.categorical(model_logits, num_samples=1)
                input_ids = tf.cast(prediction_ids, tf.int32)
            else:
                prediction_ids = tf.argmax(model_logits, axis=1)
                input_ids = tf.cast(tf.expand_dims(prediction_ids, axis=1), tf.int32)

            inputs_tuple = [None] * len(self.input_name_list)

            for index, name in self.input_name_map.items():
                if name == "input_ids":
                    inputs_tuple[index] = input_ids
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue
                if name == "input_type_ids":
                    inputs_tuple[index] = tf.ones_like(input_ids) * self.input_type_ids
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue
                if name == "input_mask":
                    inputs_tuple[index] = tf.ones_like(input_ids) * self.input_mask_ids
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue
            # Convert to tuple
            inputs_tuple = tuple(inputs_tuple)

            return [
                i + 1,
                inputs_tuple,
                model_outputs["all_cache_key"],
                model_outputs["all_cache_value"],
                model_outputs["past_length"],
                tf.concat([initial_id, input_ids], axis=1),
            ]

        # @tf.function(experimental_relax_shapes=True)
        def call_greedy(inputs):
            input_ids_orig = inputs["input_ids"]
            # Original batch size and sequence length
            batch_size = tf.shape(inputs["input_ids"])[0]
            max_sequence_length = tf.shape(inputs["input_ids"])[1]
            # Repeat for beam search (We nedd batch_size x beam_size)
            model_inputs = {}
            for input_key, input_value in inputs.items():
                if input_key == "iterations":
                    continue
                model_inputs[input_key] = input_value

            # Pre-initialize addtional inputs
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
            # past_length for keeping track of positional ids
            past_length = tf.expand_dims(tf.zeros(batch_size, dtype=tf.int32), 0)
            # Iterator to keep track of the loop
            i = tf.constant([[0]])

            if self.max_iterations is None:
                iterations = tf.squeeze(inputs["iterations"])
            else:
                iterations = self.max_iterations

            initial_id = tf.ones(shape=(batch_size, 1), dtype=tf.int32)

            # Add remaining model inputs
            model_inputs["all_cache_key"] = all_cache_key
            model_inputs["all_cache_value"] = all_cache_value
            model_inputs["past_length"] = past_length

            if "input_type_ids" in self.input_name_list:
                model_inputs["input_type_ids"] = tf.ones_like(model_inputs["input_ids"]) * self.input_type_ids

            if "input_mask" in self.input_name_list:
                model_inputs["input_mask"] = tf.ones_like(model_inputs["input_ids"]) * self.input_mask_ids

            model_outputs = self.model(model_inputs)
            model_logits = model_outputs["last_token_logits"] / self.temperature
            if self.do_sample:
                prediction_ids = tf.random.categorical(model_logits, num_samples=1)
                input_ids = tf.cast(prediction_ids, tf.int32)
            else:
                prediction_ids = tf.argmax(model_logits, axis=1)
                input_ids = tf.cast(tf.expand_dims(prediction_ids, axis=1), tf.int32)

            # Update iter
            i = i + 1
            all_cache_key = model_outputs["all_cache_key"]
            all_cache_value = model_outputs["all_cache_value"]
            initial_id = tf.concat([initial_id, input_ids], axis=1)

            masks = tf.cast(tf.not_equal(inputs["input_ids"], -1), tf.float32)
            masks = tf.reshape(masks, (1, batch_size, 1, max_sequence_length, 1))
            all_cache_key = all_cache_key * masks
            all_cache_value = all_cache_value * masks

            # END
            inputs_tuple = [None] * len(self.input_name_list)
            input_shapes_tuple = [tf.TensorShape([None, None])] * len(self.input_name_list)
            for index, name in self.input_name_map.items():
                if name == "input_ids":
                    inputs_tuple[index] = input_ids
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue
                if name == "input_type_ids":
                    inputs_tuple[index] = tf.ones_like(input_ids) * self.input_type_ids
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue
                if name == "input_mask":
                    inputs_tuple[index] = tf.ones_like(input_ids) * self.input_mask_ids
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue

            inputs_tuple = tuple(inputs_tuple)
            input_shapes_tuple = tuple(input_shapes_tuple)

            # maximum_iterations=self.max_iterations - 1
            # maximum_iterations = iterations-1
            results = tf.while_loop(
                cond,
                body,
                maximum_iterations=iterations - 1,
                loop_vars=[
                    i,
                    inputs_tuple,
                    all_cache_key,
                    all_cache_value,
                    model_outputs["past_length"],
                    initial_id,
                ],
                shape_invariants=[
                    i.get_shape(),
                    input_shapes_tuple,
                    tf.TensorShape(
                        [
                            self.num_hidden_layers,
                            None,
                            self.num_attention_heads,
                            None,
                            self.attention_state,
                        ]
                    ),
                    tf.TensorShape(
                        [
                            self.num_hidden_layers,
                            None,
                            self.num_attention_heads,
                            None,
                            self.attention_state,
                        ]
                    ),
                    tf.TensorShape([None, None]),
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
            results_dict["matched_eos_pos"] = matched_positions
            results_dict["predicted_ids"] = tf.expand_dims(results[-1][:, 1:], 1)

            return results_dict

        return call_greedy

    def beam(self):
        """
        This function will perform beam decoding.
        """

        # EOS check function
        def cond(
            i,
            input_ids,
            all_cache_key,
            all_cache_value,
            past_length,
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
            inputs_tuple,
            all_cache_key,
            all_cache_value,
            past_length,
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
                inputs[self.input_name_list[k]] = inputs_tuple[k]
            inputs["all_cache_key"] = all_cache_key
            inputs["all_cache_value"] = all_cache_value
            inputs["past_length"] = past_length

            beams_to_keep = 2 * self.beam_size
            model_outputs = self.model(inputs)

            model_logits = model_outputs["last_token_logits"]

            all_cache_key = model_outputs["all_cache_key"]
            all_cache_value = model_outputs["all_cache_value"]
            past_length = model_outputs["past_length"]

            if self.top_k > 0:
                model_logits = top_k_logits(model_logits, k=self.top_k)
            if self.top_p > 0:
                model_logits = top_p_logits(model_logits, p=self.top_p)

            vocab_size = tf.shape(model_logits)[1]
            batch_size = tf.shape(inputs["input_ids"])[0] // self.beam_size
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

            inputs_tuple = [None] * len(self.input_name_list)

            for index, name in self.input_name_map.items():
                if name == "input_ids":
                    inputs_tuple[index] = input_ids
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue
                if name == "input_type_ids":
                    inputs_tuple[index] = tf.ones_like(input_ids) * self.input_type_ids
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue
                if name == "input_mask":
                    inputs_tuple[index] = tf.ones_like(input_ids) * self.input_mask_ids
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue
            # Convert to tuple
            inputs_tuple = tuple(inputs_tuple)

            all_cache_key, all_cache_value = self.reorder_past_batches(
                all_cache_key, all_cache_value, coordinates, self.beam_size
            )
            model_outputs["all_cache_key"] = all_cache_key
            model_outputs["all_cache_value"] = all_cache_value

            return [
                i + 1,
                inputs_tuple,
                model_outputs["all_cache_key"],
                model_outputs["all_cache_value"],
                model_outputs["past_length"],
                alive_log_probs,
                alive_seq,
            ]

        # @tf.function(experimental_relax_shapes=True)
        def call_beam(inputs):
            """The main function to perform beam search
            Args:
                inputs ([dict]): [dict of tf.tensors (model inputs)]
            """
            input_ids_orig = inputs["input_ids"]
            # We take 2x beams
            beams_to_keep = 2 * self.beam_size
            # Original batch size and sequence length
            batch_size = tf.shape(inputs["input_ids"])[0]
            max_sequence_length = tf.shape(inputs["input_ids"])[1]
            # Repeat for beam search (We nedd batch_size x beam_size)
            model_inputs = {}
            for input_key, input_value in inputs.items():
                if input_key == "iterations":
                    continue
                model_inputs[input_key] = tf.repeat(input_value, [self.beam_size], axis=0)
            # New batch size
            batch_size_updated = tf.shape(model_inputs["input_ids"])[0]

            # Pre-initialize addtional inputs
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
            # past_length for keeping track of positional ids
            past_length = tf.expand_dims(tf.zeros(batch_size_updated, dtype=tf.int32), 0)
            # Iterator to keep track of the loop
            i = tf.constant([[0]])

            if self.max_iterations is None:
                iterations = tf.squeeze(inputs["iterations"])
            else:
                iterations = self.max_iterations

            # Add remaining model inputs
            model_inputs["all_cache_key"] = all_cache_key
            model_inputs["all_cache_value"] = all_cache_value
            model_inputs["past_length"] = past_length

            if "input_type_ids" in self.input_name_list:
                model_inputs["input_type_ids"] = tf.ones_like(model_inputs["input_ids"]) * self.input_type_ids

            if "input_mask" in self.input_name_list:
                model_inputs["input_mask"] = tf.ones_like(model_inputs["input_ids"]) * self.input_mask_ids

            # We need this to re-ordering and keep track of best -log(prob))
            alive_log_probs = -np.inf * tf.ones((batch_size, self.beam_size - 1))
            alive_log_probs = tf.concat([tf.zeros([batch_size, 1]), alive_log_probs], axis=1)
            alive_seq = tf.zeros((batch_size, self.beam_size, 1))

            # First pass to the model
            model_outputs = self.model(model_inputs)
            model_logits = model_outputs["last_token_logits"] / self.temperature
            # Update iter
            i = i + 1
            all_cache_key = model_outputs["all_cache_key"]
            all_cache_value = model_outputs["all_cache_value"]
            past_length = model_outputs["past_length"]

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

            inputs_tuple = [None] * len(self.input_name_list)
            input_shapes_tuple = [tf.TensorShape([None, None])] * len(self.input_name_list)
            for index, name in self.input_name_map.items():
                if name == "input_ids":
                    inputs_tuple[index] = input_ids
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue
                if name == "input_type_ids":
                    inputs_tuple[index] = tf.ones_like(input_ids) * self.input_type_ids
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue
                if name == "input_mask":
                    inputs_tuple[index] = tf.ones_like(input_ids) * self.input_mask_ids
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue

            inputs_tuple = tuple(inputs_tuple)
            input_shapes_tuple = tuple(input_shapes_tuple)

            # on step 0

            masks = tf.cast(tf.not_equal(model_inputs["input_ids"], -1), tf.float32)
            masks = tf.reshape(
                masks,
                (1, batch_size_updated, 1, tf.shape(model_inputs["input_ids"])[1], 1),
            )
            all_cache_key = all_cache_key * masks
            all_cache_value = all_cache_value * masks

            all_cache_key, all_cache_value = self.reorder_past_batches(
                all_cache_key, all_cache_value, coordinates, self.beam_size
            )

            # END
            results = tf.while_loop(
                cond,
                body,
                maximum_iterations=iterations - 1,
                loop_vars=[
                    i,
                    inputs_tuple,
                    all_cache_key,
                    all_cache_value,
                    past_length,
                    alive_log_probs,
                    alive_seq,
                ],
                shape_invariants=[
                    i.get_shape(),
                    input_shapes_tuple,
                    tf.TensorShape(
                        [
                            self.num_hidden_layers,
                            None,
                            self.num_attention_heads,
                            None,
                            self.attention_state,
                        ]
                    ),
                    tf.TensorShape(
                        [
                            self.num_hidden_layers,
                            None,
                            self.num_attention_heads,
                            None,
                            self.attention_state,
                        ]
                    ),
                    tf.TensorShape([None, None]),
                    tf.TensorShape([None, None]),
                    tf.TensorShape([None, None, None]),
                ],
            )

            results_dict = {}
            results_dict["iterations"] = results[0]
            results_dict["input_ids"] = input_ids_orig
            # Skip -1 initial ids
            results_dict["predicted_ids"] = results[-1][:, :, 1:]  # to remove initial 0

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
            results_dict["matched_eos_pos"] = matched_positions

            return results_dict

        return call_beam

    def top_k_top(self):

        # EOS check function
        def cond(i, input_ids, all_cache_key, all_cache_value, past_length, initial_id):
            eos_check = tf.greater(
                tf.reduce_prod(tf.reduce_sum(tf.cast(tf.equal(initial_id, self.eos_id), tf.int32), axis=[1])),
                0,
            )
            return tf.not_equal(eos_check, True)

        def body(i, inputs_tuple, all_cache_key, all_cache_value, past_length, initial_id):
            """[This is the body of the top k top p decoder]

            Args:
                i ([tf.tensor]): [iterator (an int)]
                inputs ([List of model inputs]): [description]
                all_cache_key ([K]): [description]
                all_cache_value ([V]): [description]
                past_length ([tf.tensor (1 x batch_size)]): [description]
                This is our main output or decoded ids]
                initial_id ([tf.tensor]): [To keep track of concatanted ids generated
                in each iteration]

            Returns:
                [List of tensors]: [Outputs]
            """
            inputs = {}
            for k in range(len(self.input_name_list)):
                inputs[self.input_name_list[k]] = inputs_tuple[k]
            inputs["all_cache_key"] = all_cache_key
            inputs["all_cache_value"] = all_cache_value
            inputs["past_length"] = past_length

            model_outputs = self.model(inputs)
            model_logits = model_outputs["last_token_logits"]

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

            inputs_tuple = [None] * len(self.input_name_list)

            for index, name in self.input_name_map.items():
                if name == "input_ids":
                    inputs_tuple[index] = input_ids
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue
                if name == "input_type_ids":
                    inputs_tuple[index] = tf.ones_like(input_ids) * self.input_type_ids
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue
                if name == "input_mask":
                    inputs_tuple[index] = tf.ones_like(input_ids) * self.input_mask_ids
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue
            # Convert to tuple
            inputs_tuple = tuple(inputs_tuple)
            return [
                i + 1,
                inputs_tuple,
                model_outputs["all_cache_key"],
                model_outputs["all_cache_value"],
                model_outputs["past_length"],
                tf.concat([initial_id, input_ids], axis=1),
            ]

        def call_top_k_top_p(inputs):
            """The main function to perform Top K top P (Nucleus) decoding
            Args:
                inputs ([dict]): [dict of tf.tensors (model inputs)]
            """
            input_ids_orig = inputs["input_ids"]
            batch_size = tf.shape(inputs["input_ids"])[0]
            max_sequence_length = tf.shape(inputs["input_ids"])[1]

            if self.max_iterations is None:
                iterations = tf.squeeze(inputs["iterations"])
            else:
                iterations = self.max_iterations

            model_inputs = {}
            for input_key, input_value in inputs.items():
                if input_key == "iterations":
                    continue
                model_inputs[input_key] = tf.repeat(input_value, [self.num_return_sequences], axis=0)
            # Updated batch size
            batch_size_updated = tf.shape(model_inputs["input_ids"])[0]

            # Pre-initialize addtional inputs
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
            # past_length for keeping track of positional ids
            past_length = tf.expand_dims(tf.zeros(batch_size_updated, dtype=tf.int32), 0)
            # Iterator to keep track of the loop
            i = tf.constant([[0]])
            initial_id = tf.ones(shape=(batch_size_updated, 1), dtype=tf.int32)

            # Add remaining model inputs
            model_inputs["all_cache_key"] = all_cache_key
            model_inputs["all_cache_value"] = all_cache_value
            model_inputs["past_length"] = past_length

            if "input_type_ids" in self.input_name_list:
                model_inputs["input_type_ids"] = tf.ones_like(model_inputs["input_ids"]) * self.input_type_ids

            if "input_mask" in self.input_name_list:
                model_inputs["input_mask"] = tf.ones_like(model_inputs["input_ids"]) * self.input_mask_ids

            # First pass to the model
            model_outputs = self.model(model_inputs)
            model_logits = model_outputs["last_token_logits"]

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
            inputs_tuple = [None] * len(self.input_name_list)
            input_shapes_tuple = [tf.TensorShape([None, None])] * len(self.input_name_list)
            for index, name in self.input_name_map.items():
                if name == "input_ids":
                    inputs_tuple[index] = input_ids
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue
                if name == "input_type_ids":
                    inputs_tuple[index] = tf.ones_like(input_ids) * self.input_type_ids
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue
                if name == "input_mask":
                    inputs_tuple[index] = tf.ones_like(input_ids) * self.input_mask_ids
                    # input_shapes_tuple.append(tf.TensorShape([None, None]))
                    continue

            inputs_tuple = tuple(inputs_tuple)
            input_shapes_tuple = tuple(input_shapes_tuple)

            # Concatanate
            initial_id = tf.concat([initial_id, input_ids], axis=1)

            # on step 0

            masks = tf.cast(tf.not_equal(model_inputs["input_ids"], -1), tf.float32)
            masks = tf.reshape(
                masks,
                (1, batch_size_updated, 1, tf.shape(model_inputs["input_ids"])[1], 1),
            )

            all_cache_key = model_outputs["all_cache_key"]
            all_cache_value = model_outputs["all_cache_value"]
            all_cache_key = all_cache_key * masks
            all_cache_value = all_cache_value * masks
            # END

            results = tf.while_loop(
                cond,
                body,
                maximum_iterations=iterations - 1,
                loop_vars=[
                    i,
                    inputs_tuple,
                    all_cache_key,
                    all_cache_value,
                    model_outputs["past_length"],
                    initial_id,
                ],
                shape_invariants=[
                    i.get_shape(),
                    input_shapes_tuple,
                    tf.TensorShape(
                        [
                            self.num_hidden_layers,
                            None,
                            self.num_attention_heads,
                            None,
                            self.attention_state,
                        ]
                    ),
                    tf.TensorShape(
                        [
                            self.num_hidden_layers,
                            None,
                            self.num_attention_heads,
                            None,
                            self.attention_state,
                        ]
                    ),
                    tf.TensorShape([None, None]),
                    tf.TensorShape([None, None]),
                ],
            )

            results_dict = {}
            results_dict["iterations"] = results[0]
            results_dict["input_ids"] = input_ids_orig
            # Skip -1 initial ids
            results_dict["predicted_ids"] = results[-1][:, 1:]
            results_dict["predicted_ids"] = tf.reshape(
                results_dict["predicted_ids"],
                (batch_size, self.num_return_sequences, -1),
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
            results_dict["matched_eos_pos"] = matched_positions

            return results_dict

        return call_top_k_top_p

    def call(self, inputs):
        results_dict = self.decoder_fn(inputs)
        return results_dict
