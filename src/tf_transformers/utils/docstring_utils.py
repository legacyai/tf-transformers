ENCODER_PRETRAINED_DOCSTRING = r"""
        (:class:`{0}`) is a wrapper around (:class:`{1}`).

        Instantiates a :class:`~{1}` from a :obj:`model_name`

        Args:
            model_name (:obj:`str`): Name of the model as in HuggingFace Transformers. eg: (:obj:`{2}`)
            cache_dir  (:obj:`str`): Where model will be saved after conversion: If None for Linux based machine
                                     the directory will be (:obj:`/tmp/tf_transformers_cache`)
            model_checkpoint_dir (:obj:`str`): Model checkpoint directory. If provided, we won't convert from HF,
                                     rather, we try to load it from it.
            convert_from_hf (:obj:`bool`): Whether we need to convert from HF to TFT(tensorflow transformers).
                                     default (:obj:`True`).
            return_layer (:obj:`bool`): Whether to return tf.keras.layers.Layer/LegacyLayer.
            return_config (:obj:`bool`): Whether to return TransformerConfig.
            convert_fn_type (:obj:`bool`): It will accept either of 3 values :obj:`('tf', 'pt', 'both')`.
                                     default (:obj:`both`).
            save_checkpoint_cache (:obj:`bool`): Do we need to save the checkpoints to cache directory.
            load_from_cache (:obj:`bool`): Do we need to load the checkpoints from :obj:(`cache_dir`), if exists.

            kwargs (:obj:`Dict[str, Any]`):
                Additional parameters required to pass to the :class:`~{1}`
                like: :obj:`mask_mode`, :obj:`use_dropout`, :obj:`is_training`, :obj:`use_auto_regressive`,
                :obj:`use_decoder`, :obj:`batch_size`, :obj:`sequence_length`, :obj:`use_mlm_layer`,
                :obj:`use_masked_lm_positions`, :obj:`return_all_layer_outputs` etc. For more details please refer
                :class:`~{1}`.
        Returns:
            LegacyModel/LegacyLayer with or without config.

        Examples::

            {3}

        """

ENCODER_MODEL_CONFIG_DOCSTRING = r"""
        Instantiates a :class:`~{0}` from a :class:`~{1}`

        Args:
            config_dict (:obj:`{1}`):
                Model configuration Object.
            kwargs (:obj:`Dict[str, Any]`):
                Additional parameters required to pass to the :class:`~{0}`
                like: :obj:`mask_mode`, :obj:`use_dropout`, :obj:`is_training`, :obj:`use_auto_regressive`,
                :obj:`use_decoder`, :obj:`batch_size`, :obj:`sequence_length`, :obj:`use_mlm_layer`,
                :obj:`use_masked_lm_positions`, :obj:`return_all_layer_outputs` etc

        Returns:
            LegacyModel/LegacyLayer .
        """
ENCODER_CLASS_DOCSTRING = r"""

    This model inherits from :class:`~tf_transformers.core.LegacyLayer`. Check the superclass documentation details
    and design inspiration of LegacyLayer.
    This model is also a `tf.keras.layers.Layer <https://www.tensorflow.org/api_docs/python/tf/keras/layers>`__ subclass. Use
    it as a regular TF 2.0 Keras layer and refer to the TF 2.0 documentation for all matter related to general usage
    and behavior.

    .. note::

        TF 2.0 models accepts input formats as Dict:
        - having all inputs as dict .

        This is recommended and useful when using :meth:`tf.keras.Model.fit` method which currently requires having all
        the tensors in the first argument of the model call function: :obj:`model(inputs)`.

        Encoder Only Models like Bert, Albert, Roberta, GPT2 etc requires inputs in following format:
        - a dictionary with one or several input Tensors associated to the input names given in the docstring:
        :obj:`model({{"input_ids": input_ids,\
        "input_mask": input_mask, \
        "input_type_ids": input_type_ids}})`.

    Args:
        config (:class:`~tf_transformers.core.TransformerConfig`): Model configuration.
        mask_mode (:obj:`str`): Mask Mode defines the type of mask used in the architecture.

            (:obj:`causal`: Casual LM masking with masking upper triangular matrix eg: GPT2).

            (:obj:`user_defined`: User defined LM masking eg: Bert, Albert, Roberta etc).

            (:obj:`prefix`: Casual LM masking but bidirectional to model inputs).
        name (:obj:`str`): Model name.
        use_dropout (:obj:`bool`): If True only dropout will be enabled.
        is_training (:obj:`bool`): This is used to mainly switch to auto regressive tasks.
        use_decoder (:obj:`bool`): If true we will use :meth:`call_decoder` while :obj:`is_training=True`.
        batch_size  (:obj:`int`, `optional`, defaults to :obj:`None`): Batch size.
        sequence_length (:obj:`int`, `optional`, defaults to :obj:`None`):: Sequence Length.
                If :obj:`position_embedding_type='absolute'`, sequence length should be with in
                :obj:`{0}.max_position_embeddings` limit.
        use_mlm_layer (:obj:`bool`, `optional`, defaults to :obj:`True`): If True, we will add
                :meth:`tf.keras.layers.Layer` head as in case of Bert, Albert etc.
        use_masked_lm_positions (:obj:`bool`, `optional`, defaults to :obj:`False`): If training
                an `MLM Language Model`, set `use_masked_lm_positions=True`. If it is True,
                inputs should have :obj:`masked_lm_positions` in inputs . `(batch_size x masked_labels_length)`.
        return_all_layer_outputs  (:obj:`bool`, `optional`, defaults to :obj:`False`): If True,
                outputs from all layers will be returned to the model.


"""

CALL_ENCODER_DOCSTRING = r"""
        An Encoder in general accepts :obj:`input_ids` and optional (:obj:`input_mask` and :obj:`input_type_ids`)

        .. note::
            a. **Inputs**.

            TF 2.0 models accepts input formats as :obj:`Dict`:
            - having all inputs as dict .

            :obj:`input_mask`
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            :obj:`input_type_ids`
            - 0 for tokens that are **Sentence A**,
            - 1 for tokens that are **Sentence B**. (Not applicable to all. **Roberta** has only **0**.)

            Encoder Only Models like Bert, Albert, Roberta, GPT2 etc requires inputs in following format:
            - a dictionary with one or several input Tensors.

            b. **Outputs**.

            TF 2.0 models returns output formats as :obj:`Dict`:
            - having all outputs as dict .

        Args:
            inputs: Dict of tf.Tensor (:obj:`"input_ids": tf.int32 (batch_size x sequence_length)`,
                                       :obj:`"input_type_ids": Optional. tf.int32 (batch_size x sequence_length)`,
                                       :obj:`"input_mask": Optional. tf.int32 (batch_size x sequence_length)`)

        Returns:
            result: Dict of tf.Tensor.

            (:obj:`"cls_output": tf.float32 (batch_size x sequence_length)`,
            :obj:`"token_embeddings": tf.float32 (batch_size x sequence_length x embedding_size)`,
            :obj:`"token_logits": tf.float32 (batch_size x sequence_length x vocab_size)`)

            If :obj:`return_all_layers=True`:

            (:obj:`"all_layer_token_embeddings": tf.float32 List[(batch_size x sequence_length x embedding_size)]`,
            :obj:`"all_layer_token_logits": tf.float32 List[(batch_size x sequence_length x vocab_size)]`,
            :obj:`"all_layer_cls_output": tf.float32 List[(batch_size x sequence_length)]`)

        """

CALL_ENCODER_DOCSTRING_IMAGE = r"""
        An Encoder in general accepts :obj:`input_ids` and optional (:obj:`input_mask` and :obj:`input_type_ids`)

        .. note::
            a. **Inputs**.

            TF 2.0 models accepts input formats as :obj:`Dict`:
            - having all inputs as dict .

            :obj:`input_mask`
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            :obj:`input_type_ids`
            - 0 for tokens that are **Sentence A**,
            - 1 for tokens that are **Sentence B**. (Not applicable to all. **Roberta** has only **0**.)

            Encoder Only Models like Bert, Albert, Roberta, GPT2 etc requires inputs in following format:
            - a dictionary with one or several input Tensors.

            b. **Outputs**.

            TF 2.0 models returns output formats as :obj:`Dict`:
            - having all outputs as dict .

        Args:
            inputs: Dict of tf.Tensor (:obj:`"input_pixels": tf.int32 (batch_size x sequence_length)`,
                                       :obj:`"input_type_ids": Optional. tf.int32 (batch_size x sequence_length)`,
                                       :obj:`"input_mask": Optional. tf.int32 (batch_size x sequence_length)`)

        Returns:
            result: Dict of tf.Tensor.

            (:obj:`"cls_output": tf.float32 (batch_size x sequence_length)`,
            :obj:`"token_embeddings": tf.float32 (batch_size x sequence_length x embedding_size)`,
            :obj:`"token_logits": tf.float32 (batch_size x sequence_length x vocab_size)`)

            If :obj:`return_all_layers=True`:

            (:obj:`"all_layer_token_embeddings": tf.float32 List[(batch_size x sequence_length x embedding_size)]`,
            :obj:`"all_layer_token_logits": tf.float32 List[(batch_size x sequence_length x vocab_size)]`,
            :obj:`"all_layer_cls_output": tf.float32 List[(batch_size x sequence_length)]`)

        """
        
CALL_ENCODER_DOCSTRING_CLIP = r"""
        An Encoder in general accepts :obj:`input_ids` and optional (:obj:`input_mask` and :obj:`input_type_ids`)

        .. note::
            a. **Inputs**.

            TF 2.0 models accepts input formats as :obj:`Dict`:
            - having all inputs as dict .

            :obj:`input_mask`
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            :obj:`input_type_ids`
            - 0 for tokens that are **Sentence A**,
            - 1 for tokens that are **Sentence B**. (Not applicable to all. **Roberta** has only **0**.)

            Encoder Only Models like Bert, Albert, Roberta, GPT2 etc requires inputs in following format:
            - a dictionary with one or several input Tensors.

            b. **Outputs**.

            TF 2.0 models returns output formats as :obj:`Dict`:
            - having all outputs as dict .

        Args:
            inputs: Dict of tf.Tensor (:obj:`"input_pixels": tf.int32 (batch_size x sequence_length)`,
                                       :obj:`"input_ids": Optional. tf.int32 (batch_size x sequence_length)`
                                       :obj:`"input_type_ids": Optional. tf.int32 (batch_size x sequence_length)`,
                                       :obj:`"input_mask": Optional. tf.int32 (batch_size x sequence_length)`)

        Returns:
            result: Dict of tf.Tensor.

            (:obj:`"image_cls_output": tf.float32 (batch_size x sequence_length)`,
            :obj:`"image_features": tf.float32 (batch_size x sequence_length)`,
            :obj:`"text_cls_output": tf.float32 (batch_size x sequence_length)`,       
            :obj:`"text_features": tf.float32 (batch_size x sequence_length)`,
            :obj:`"text_token_embeddings": tf.float32 (batch_size x sequence_length x embedding_size)`
            )

            If :obj:`return_all_layers=True`:

            (:obj:`"all_layer_token_embeddings": tf.float32 List[(batch_size x sequence_length x embedding_size)]`,
            :obj:`"all_layer_token_logits": tf.float32 List[(batch_size x sequence_length x vocab_size)]`,
            :obj:`"all_layer_cls_output": tf.float32 List[(batch_size x sequence_length)]`)

        """

CALL_DECODER_DOCSTRING = r"""
        An Encoder in general accepts :obj:`input_ids` and optional (:obj:`input_mask` and :obj:`input_type_ids`)

        .. note::
            a. **Inputs**.

            TF 2.0 models accepts input formats as :obj:`Dict`:
            - having all inputs as dict .

            Encoder Only Models like Bert, Albert, Roberta, GPT2 etc requires inputs in following format:
            - a dictionary with one or several input Tensors.

            Note: :obj:`decoder_encoder_mask`, will be created automatically inside.

            b. **Outputs**.

            TF 2.0 models returns output formats as :obj:`Dict`:
            - having all outputs as dict .

        Args:
            inputs: Dict of tf.Tensor (:obj:`"input_ids": tf.int32 (batch_size x decoder_sequence_length)`,
                                       :obj:`"input_type_ids": Optional. tf.int32 (batch_size x
                                       decoder_sequence_length)`,
                                       :obj:`"input_mask": Optional. tf.int32 (batch_size x decoder_sequence_length)`,
                                       :obj:`"encoder_hidden_states": tf.float32 (batch_size x
                                       encoder_sequence_length x embedding_size),
                                       :obj:`"decoder_encoder_mask": tf.float32 (batch_size x
                                       encoder_sequence_length x decoder_sequence_length))

        Returns:
            result: Dict of tf.Tensor.

            (:obj:`"cls_output": tf.float32 (batch_size x sequence_length)`,
            :obj:`"token_embeddings": tf.float32 (batch_size x sequence_length x embedding_size)`,
            :obj:`"token_logits": tf.float32 (batch_size x sequence_length x vocab_size)`)

            If :obj:`return_all_layers=True`:

            (:obj:`"all_layer_token_embeddings": tf.float32 List[(batch_size x sequence_length x embedding_size)]`,
            :obj:`"all_layer_token_logits": tf.float32 List[(batch_size x sequence_length x vocab_size)]`,
            :obj:`"all_layer_cls_output": tf.float32 List[(batch_size x sequence_length)]`)

        """


CALL_ENCODER_AUTO_REGRESSIVE_DOCSTRING = r"""
        An Encoder Auto Regressive model in general accepts :obj:`input_ids`, optional (:obj:`input_mask`,
        :obj:`input_type_ids`), :obj:`all_cache_key`, :obj:`all_cache_value`, :obj:`past_length`.

        This is required for caching. But a user need not to be worry about this, as this will be
        encapsulated/abstracted inside :class:`TextDecoder`.

        .. note::
            a. **Inputs**.

            TF 2.0 models accepts input formats as :obj:`Dict`:
            - having all inputs as dict .

            Encoder Only Models like Bert, Albert, Roberta, GPT2 etc requires inputs in following format:
            - a dictionary with one or several input Tensors.

            b. **Outputs**.

            TF 2.0 models returns output formats as :obj:`Dict`:
            - having all outputs as dict .

        Args:
            inputs: Dict of tf.Tensor (:obj:`"input_ids": tf.int32 (batch_size x sequence_length)`,
                                       :obj:`"input_type_ids": Optional. tf.int32 (batch_size x sequence_length)`,
                                       :obj:`"input_mask": Optional. tf.int32 (batch_size x sequence_length)`,
                                       :obj:`"all_cache_key": tf.float32
                                       (num_hidden_layers ,
                                       batch_size ,
                                       num_attention_heads ,
                                       sequence_length,
                                       attention_head_size)`,
                                       :obj:`"all_cache_key": tf.float32
                                       (num_hidden_layers ,
                                       batch_size ,
                                       num_attention_heads ,
                                       sequence_length,
                                       attention_head_size)`,
                                       :obj:`"past_length": tf.int32 (1 x sequence_length)`)

        Returns:
            result: Dict of tf.Tensor.

            (:obj:`"cls_output": tf.float32 (batch_size x sequence_length)`,
            :obj:`"token_embeddings": tf.float32 (batch_size x sequence_length x embedding_size)`,
            :obj:`"last_token_logits": tf.float32 (batch_size x  vocab_size)`,
            :obj:`"all_cache_key": tf.float32
            (num_hidden_layers ,
            batch_size ,
            num_attention_heads ,
            sequence_length,
            attention_head_size)`,
            :obj:`"all_cache_key": tf.float32
            (num_hidden_layers ,
            batch_size ,
            num_attention_heads ,
            sequence_length,
            attention_head_size)`
            )

        """


CALL_DECODER_AUTO_REGRESSIVE_DOCSTRING = r"""
        A Decoder Auto Regressive model in general accepts :obj:`input_ids`, optional (:obj:`input_mask`,
        :obj:`input_type_ids`), :obj:`all_cache_key`, :obj:`all_cache_value`.

        This is required for caching. But a user need not to be worry about this, as this will be
        encapsulated/abstracted inside :class:`TextDecoder`. The main difference between Encoder Auto Regressive
        is, here no need of obj:`past_length`, as variable sequence length is not an issue.

        .. note::
            a. **Inputs**.

            TF 2.0 models accepts input formats as :obj:`Dict`:
            - having all inputs as dict .

            Encoder Only Models like Bert, Albert, Roberta, GPT2 etc requires inputs in following format:
            - a dictionary with one or several input Tensors.

            b. **Outputs**.

            TF 2.0 models returns output formats as :obj:`Dict`:
            - having all outputs as dict .

        Args:
            inputs: Dict of tf.Tensor (:obj:`"input_ids": tf.int32 (batch_size x sequence_length)`,
                                       :obj:`"input_type_ids": Optional. tf.int32 (batch_size x sequence_length)`,
                                       :obj:`"input_mask": Optional. tf.int32 (batch_size x sequence_length)`,
                                       :obj:`"all_cache_key": tf.float32
                                       (num_hidden_layers ,
                                       batch_size ,
                                       num_attention_heads ,
                                       sequence_length,
                                       attention_head_size)`,
                                       :obj:`"all_cache_key": tf.float32
                                       (num_hidden_layers ,
                                       batch_size ,
                                       num_attention_heads ,
                                       sequence_length,
                                       attention_head_size)`)

        Returns:
            result: Dict of tf.Tensor.

            (:obj:`"cls_output": tf.float32 (batch_size x sequence_length)`,
            :obj:`"token_embeddings": tf.float32 (batch_size x sequence_length x embedding_size)`,
            :obj:`"last_token_logits": tf.float32 (batch_size x  vocab_size)`,
            :obj:`"all_cache_key": tf.float32
            (num_hidden_layers ,
            batch_size ,
            num_attention_heads ,
            sequence_length,
            attention_head_size)`,
            :obj:`"all_cache_key": tf.float32
            (num_hidden_layers ,
            batch_size ,
            num_attention_heads ,
            sequence_length,
            attention_head_size)`
            )

        """

MAIN_CALL_DOCSTRING = r"""
        All calls to the model will be routed from here. This routing is more of like a :obj:`Factory` concept.
        This routing will be decided, when the model is initialized. This cannot be changed dynamically.
        For more details, check :class:`~tf_transformers.core.LegacyLayer.get_call_method`.

        .. note::
            A call cannot be changed **dynamically**.

            if :obj:`is_training=True/False` and if :obj:`use_decoder=True`:  return :meth:`call_decoder`.

            if :obj:`is_training=True/False` and if :obj:`use_decoder=False`: return :meth:`call_encoder`.

            if :obj:`is_training=False` and if :obj:`use_decoder=True` and if
            :obj:`use_auto_regressive=True`: return :meth:`call_decoder_auto_regressive`.

            if :obj:`is_training=False` and if :obj:`use_decoder=False` and if
            :obj:`use_auto_regressive=True`: return :meth:`call_encoder_auto_regressive`.

        """
