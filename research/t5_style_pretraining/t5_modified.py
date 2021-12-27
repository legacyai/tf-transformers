import tensorflow as tf
from absl import logging

from tf_transformers.core import LegacyLayer, LegacyModel
from tf_transformers.layers.mask import CrossAttentionMask
from tf_transformers.utils import tf_utils


class EncoderDecoderwithMLM(LegacyLayer):
    def __init__(
        self,
        encoder,
        decoder,
        is_training=False,
        use_dropout=False,
        decoder_start_token_id=None,
        **kwargs,
    ):

        self._encoder = encoder
        self._decoder = decoder
        self._is_training = is_training
        self._use_dropout = use_dropout
        self.decoder_start_token_id = decoder_start_token_id

        self._encoder_config_dict = self._encoder._config_dict
        self._decoder_config_dict = self._decoder._config_dict

        self._model_name = self._encoder.name + "_" + self._decoder.name
        self._model_name = self._model_name.replace("tf_transformers", "").replace("/", "")

        super(EncoderDecoderwithMLM, self).__init__(
            is_training=self._is_training, use_dropout=self._use_dropout, name=self._model_name, **kwargs
        )

        if self._encoder._mask_mode != "user_defined":
            raise ValueError("mask_mode for encoder should be `user-defined`.")
        # Two different hidden dimension has to be changed
        if self._encoder_config_dict["embedding_size"] != self._decoder_config_dict["embedding_size"]:
            self._encoder_decoder_projection = tf.keras.layers.Dense(
                self._encoder_config_dict["embedding_size"], activation="linear"
            )
        else:
            self._encoder_decoder_projection = tf.identity

        self.logits_scale = tf.Variable(tf.math.log(1 / 0.07), name='logits_scale')

        self._pooler_layer = tf.keras.layers.Dense(
            units=self._encoder_config_dict["embedding_size"],
            activation="tanh",
            kernel_initializer=self._encoder_config_dict['initializer'],
            name="pooler_transform",
        )

        self.linear_projection = tf.keras.layers.Dense(
            units=self._encoder_config_dict["embedding_size"],
            activation=None,
            kernel_initializer=self._encoder_config_dict['initializer'],
            name="linear_projection",
        )

        # Initialize model
        self.model_inputs, self.model_ouputs = self.get_model(initialize_only=True)

    def get_model(self, initialize_only=False):
        """Convert tf.keras.Layer to a tf.keras.Model/LegacyModel.
        Args:
            self: model (tf.keras.Layer) instance
            initialize_only: bool

        """
        # encoder_sequence_length = self._encoder._sequence_length
        # decoder_sequence_length = self._decoder._sequence_length

        encoder_inputs = self._encoder.model_inputs
        decoder_inputs = self._decoder.model_inputs

        inputs = {}

        # Add MLM inputs layer
        inputs['masked_lm_positions'] = tf.keras.layers.Input(
            shape=(None,),
            batch_size=None,
            dtype=tf.int32,
            name="masked_lm_positions",
        )
        # Convert 'input_ids' --> 'encoder_input_ids'
        # Add 'encoder' prefix
        for k, v in encoder_inputs.items():
            if k in ["input_ids", "input_mask", "input_type_ids"]:
                shape = encoder_inputs[k].shape
                inputs["encoder_" + k] = tf.keras.layers.Input(
                    shape[1:], batch_size=encoder_inputs[k].shape[0], name="encoder_" + k, dtype=encoder_inputs[k].dtype
                )
                continue
            inputs["encoder_" + k] = v

        # Convert 'input_ids' --> 'decoder_input_ids'
        # Add 'decoder' prefix
        for k, v in decoder_inputs.items():
            # Do not add prefix if 'decoder' or 'encoder' is present
            if k.startswith("decoder") or k.startswith("encoder"):
                inputs[k] = v
                continue
            # We don't want to change name of all_cache_key to decoder_all_cache_key
            # and similarily for all_cache_value . As it will raise issues
            # while serializing
            if k in ["all_cache_key", "all_cache_value"]:
                shape = decoder_inputs[k].shape
                inputs["decoder_" + k] = tf.keras.layers.Input(
                    shape[1:], batch_size=decoder_inputs[k].shape[0], name="decoder_" + k, dtype=decoder_inputs[k].dtype
                )
                continue
            inputs["decoder_" + k] = v

        del inputs["decoder_encoder_mask"]
        if not self._decoder._use_auto_regressive:
            del inputs["encoder_hidden_states"]
        layer_outputs = self(inputs)
        if initialize_only:
            return inputs, layer_outputs

        model = LegacyModel(inputs=inputs, outputs=layer_outputs, name=self._model_name)
        # Add config
        config = {}
        config["encoder"] = self._encoder._config_dict
        # Check for decoder start token id in decoder
        if "decoder_start_token_id" not in self._decoder._config_dict:
            if self.decoder_start_token_id is None:
                raise ValueError(
                    "In EncoderDecoder setting, `decoder_start_token_id` has to set either from config or\
                    constructor. Assuming we are in Auto Regressive setting"
                )
            else:
                logging.info("Setting decoder_start_token_id = {}".format(self.decoder_start_token_id))
                self._decoder._config_dict["decoder_start_token_id"] = self.decoder_start_token_id
        else:
            # If it is None in config
            if self._decoder._config_dict["decoder_start_token_id"] is None:
                if self.decoder_start_token_id is None:
                    raise ValueError(
                        "In EncoderDecoder setting, `decoder_start_token_id` has to set either from config or\
                        constructor. Assuming we are in Auto Regressive setting"
                    )
                else:
                    # If it is set in encoder-decoder constructor use it
                    logging.info("Setting decoder_start_token_id = {}".format(self.decoder_start_token_id))
                    self._decoder._config_dict["decoder_start_token_id"] = self.decoder_start_token_id

        config["decoder"] = self._decoder._config_dict
        model.model_config = config
        return model

    def _gather_indexes(self, sequence_tensor, positions):
        """Gathers the vectors at the specific positions, for performance.
        Args:
            sequence_tensor: Sequence output of shape
                (`batch_size`, `seq_length`, num_hidden) where num_hidden is number of
                hidden units.
            positions: Positions ids of tokens in sequence to mask for pretraining
                of with dimension (batch_size, num_predictions) where
                `num_predictions` is maximum number of tokens to mask out and predict
                per each sequence.
        Returns:
            Masked out sequence tensor of shape (batch_size * num_predictions,
            num_hidden).
        """
        sequence_shape = tf.shape(sequence_tensor)
        batch_size, seq_length = sequence_shape[0], sequence_shape[1]
        width = sequence_shape[2]

        flat_offsets = tf.reshape(tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
        flat_positions = tf.reshape(positions + flat_offsets, [-1])
        flat_sequence_tensor = tf.reshape(sequence_tensor, [batch_size * seq_length, width])
        output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
        # Make it 3D (b x s x h)
        output_tensor = tf.reshape(output_tensor, (batch_size, -1, sequence_tensor.shape[2]))
        return output_tensor

    def call_forward(self, inputs):

        # Replace 'encoder_input_ids'      to 'input_ids'
        # Replace 'encoder_input_mask'     to 'input_mask'
        # Replace 'encoder_input_type_ids' to 'input_type_ids'
        encoder_inputs = {
            k.replace("encoder_", ""): v
            for k, v in inputs.items()
            if k.startswith("encoder_")
            if k not in ["encoder_hidden_states"]
        }
        encoder_inputs['masked_lm_positions'] = inputs['masked_lm_positions']
        # Replace 'decoder_input_ids'      to 'input_ids'
        # Replace 'decoder_input_mask'     to 'input_mask'
        # Replace 'decoder_input_type_ids' to 'input_type_ids'
        decoder_inputs = {
            k.replace("decoder_", ""): v
            for k, v in inputs.items()
            if k.startswith("decoder_")
            if k not in ["decoder_encoder_mask"]
        }

        if 'masked_lm_positions' in decoder_inputs:
            del decoder_inputs['masked_lm_positions']

        # Call Encoder and take the last hidden states (B x S x E)
        encoder_outputs = self._encoder(encoder_inputs)

        encoder_hidden_states = encoder_outputs["token_embeddings"]
        masked_lm_positions = encoder_inputs['masked_lm_positions']

        encoder_embeddings_mlm = self._gather_indexes(encoder_hidden_states, masked_lm_positions)
        encoder_hidden_states = self._encoder_decoder_projection(encoder_hidden_states)

        # Sometimes if we pad a sequence, last poistions might not represent CLS, so this is better
        # encoder_cls_positions = tf.where(tf.equal(inputs['encoder_input_ids'], self.cls_token_id))
        # encoder_cls = tf.gather_nd(encoder_hidden_states, encoder_cls_positions)

        # 0 is for CLS_ENC position
        encoder_cls = encoder_hidden_states[:, 0, :]

        encoder_cls = self.linear_projection(self._pooler_layer(encoder_cls))
        encoder_logits = tf.matmul(
            encoder_embeddings_mlm, tf.cast(self._encoder.get_embedding_table(), tf_utils.get_dtype()), transpose_b=True
        )

        # This is decoder_encoder_mask
        decoder_encoder_mask = CrossAttentionMask()([decoder_inputs["input_ids"], encoder_inputs["input_mask"]])

        # Add the inputs to decoder
        decoder_inputs["encoder_hidden_states"] = encoder_hidden_states
        decoder_inputs["decoder_encoder_mask"] = decoder_encoder_mask

        decoder_outputs = self._decoder(decoder_inputs)
        decoder_outputs["encoder_hidden_states"] = encoder_hidden_states

        # Sometimes if we pad a sequence, last poistions might not represent CLS, so this is better
        # decoder_cls_positions = tf.where(tf.equal(inputs['decoder_input_ids'], self.cls_token_id))
        # decoder_cls = tf.gather_nd(decoder_outputs['token_embeddings'], decoder_cls_positions)
        # decoder_cls = self.linear_projection(self._pooler_layer(decoder_cls))

        # -1 is CLS_DEC last token position
        decoder_cls = decoder_outputs['token_embeddings'][:, -1, :]

        encoder_cls_normalized = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(encoder_cls)
        decoder_cls_normalized = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(decoder_cls)

        logits = tf.matmul(encoder_cls_normalized, decoder_cls_normalized, transpose_b=True)

        # Based on CLIP
        logits_scale = tf.math.exp(self.logits_scale)
        logits_scale = tf.clip_by_value(logits_scale, clip_value_min=tf.math.log(1 / 0.07), clip_value_max=4.6051752)
        logits_scale = tf.cast(logits_scale, dtype=tf_utils.get_dtype())

        logits = logits * logits_scale

        outputs = {}
        outputs['encoder_embeddings'] = encoder_hidden_states
        outputs['encoder_cls_token'] = encoder_cls
        outputs['decoder_embeddings'] = decoder_outputs['token_embeddings']
        outputs['decoder_cls_token'] = decoder_cls
        outputs['encoder_token_logits'] = encoder_logits
        outputs['decoder_token_logits'] = decoder_outputs['token_logits']
        outputs['logits'] = logits

        return outputs

    def call_auto_regressive(self, inputs):
        """Forward pass of an EncoderDecoder Model

        Args:
            inputs ([dict of tf.Tensor]): This is the input to the model.

            'encoder_input_ids'         --> tf.int32 (b x s)
            'encoder_input_mask'        --> tf.int32 (b x s) # optional
            'encoder_input_type_ids'    --> tf.int32 (b x s) # optional

            'decoder_input_ids'         --> tf.int32 (b x s)
            'decoder_input_mask'        --> tf.int32 (b x s) # optional
            'decoder_input_type_ids'    --> tf.int32 (b x s) # optional

            'decoder_all_cache_key'     --> tf.float32 (num_hidden_layers ,
                                    batch_size ,
                                    num_attention_heads ,
                                    sequence_length,
                                    attention_head_size)

            'decoder_all_cache_value'    --> tf.float32 (num_hidden_layers ,
                                    batch_size ,
                                    num_attention_heads ,
                                    sequence_length,
                                    attention_head_size)

        Returns:
            [dict of tf.Tensor]: Output from the model

            'cls_output'        --> tf.float32 (b x s) # optional
            'token_embeddings'  --> tf.float32 (b x s x h)
            'all_layer_token_embeddings' --> tf.float32 (List of (b x s x h)
                                            from all layers)
            'all_layer_cls_output'       --> tf.float32 (List of (b x s)
                                            from all layers)

        """
        encoder_inputs = {
            k.replace("encoder_", ""): v
            for k, v in inputs.items()
            if k.startswith("encoder_")
            if k not in ["encoder_hidden_states"]
        }
        # encoder_inputs["encoder_hidden_states"] = inputs["encoder_hidden_states"]

        decoder_inputs = {
            k.replace("decoder_", ""): v
            for k, v in inputs.items()
            if k.startswith("decoder_")
            if k not in ["decoder_encoder_mask"]
        }
        decoder_encoder_mask = CrossAttentionMask()([decoder_inputs["input_ids"], encoder_inputs["input_mask"]])

        # While decoding we have to calculate it only once
        def use_cache_encoder():
            return tf.identity(inputs["encoder_hidden_states"])

        # First step of decoding process
        def calculate_encoder_hidden_state():
            encoder_outputs = self._encoder(encoder_inputs)
            encoder_hidden_states = self._encoder_decoder_projection(encoder_outputs["token_embeddings"])
            return tf.cast(encoder_hidden_states, tf.float32)

        cache_key = decoder_inputs["all_cache_key"]
        encoder_hidden_states = tf.cond(
            tf.equal(tf.reduce_sum(cache_key), 0.0),
            lambda: calculate_encoder_hidden_state(),
            lambda: use_cache_encoder(),
        )

        decoder_inputs["encoder_hidden_states"] = encoder_hidden_states
        decoder_inputs["decoder_encoder_mask"] = decoder_encoder_mask

        decoder_outputs = self._decoder(decoder_inputs)
        decoder_outputs["encoder_hidden_states"] = encoder_hidden_states
        decoder_outputs["decoder_all_cache_key"] = decoder_outputs["all_cache_key"]
        decoder_outputs["decoder_all_cache_value"] = decoder_outputs["all_cache_value"]

        del decoder_outputs["all_cache_key"]
        del decoder_outputs["all_cache_value"]

        return decoder_outputs

    def call(self, inputs):
        """Call

        Args:
            inputs : dict of tf.tensor

        Returns:
            dict of tf.tensor
        """
        if self._is_training is False and self._decoder._use_auto_regressive:
            outputs = self.call_auto_regressive(inputs)
        else:
            outputs = self.call_forward(inputs)
        return outputs
