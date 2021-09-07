import sentencepiece
import tensorflow as tf
import tensorflow_text as tf_text
from absl import logging

# tok2 = T5Tokenizer(**kwargs)
# tok2.unique_no_split_tokens = tok2.all_special_tokens


def extend_sentencepicemodel(in_file, out_file, special_tokens):
    """Extend the vocab for Sentencepice model

    Args:
        in_file : path of the model
        out_file : output path of the model
        special_tokens : list of special tokens
    """
    from tf_transformers.text import sentencepiece_model_pb2

    mp = sentencepiece_model_pb2.ModelProto()
    mp.ParseFromString(open(in_file, "rb").read())

    for token in special_tokens:
        new_token = sentencepiece_model_pb2.ModelProto().SentencePiece()
        new_token.piece = token
        new_token.score = 0
        mp.pieces.append(new_token)
        print(f'added {token}...')

    with open(out_file, 'wb') as f:
        f.write(mp.SerializeToString())


def get_vocab(model_proto):
    """Get vocab from sentencpiece model"""
    sp_model = sentencepiece.SentencePieceProcessor()
    sp_model.LoadFromSerializedProto(model_proto)
    vocab = {sp_model.IdToPiece(i): i for i in range(sp_model.GetPieceSize())}
    return vocab


class SentencepieceTokenizer(tf.keras.layers.Layer):
    """Wraps `tf_text.SentencepieceTokenizer` as a Keras Layer.
    Attributes:
    tokenize_with_offsets: If true, calls
      `SentencepieceTokenizer.tokenize_with_offsets()`
      instead of plain `.tokenize()` and outputs a triple of
      `(tokens, start_offsets, limit_offsets)`.
    """

    def __init__(
        self,
        *,
        lower_case,
        special_tokens,
        model_file_path=None,
        model_serialized_proto=None,
        add_cls_sep=False,
        cls_token=None,
        sep_token=None,
        tokenize_with_offsets=False,
        nbest_size: int = 0,
        alpha: float = 1.0,
        strip_diacritics=False,
        **kwargs,
    ):
        """Initializes a SentencepieceTokenizer layer.
        Args:
          lower_case: A Python boolean indicating whether to lowercase the string
            before tokenization. NOTE: New models are encouraged to build `*_cf`
            (case folding) normalization into the Sentencepiece model itself and
            avoid this extra step.
          special_tokens: A list of special tokens , must present in model. If not pass None.
          model_file_path: A Python string with the path of the sentencepiece model.
            Exactly one of `model_file_path` and `model_serialized_proto` can be
            specified. In either case, the Keras model config for this layer will
            store the actual proto (not a filename passed here).
          model_serialized_proto: The sentencepiece model serialized proto string.
          add_cls_sep: To add [CLS] and [SEP] with the tokenized text
          cls_token: cls token string
          sep_token: sep token string
          tokenize_with_offsets: A Python boolean. If true, this layer calls
            `SentencepieceTokenizer.tokenize_with_offsets()` instead of
            plain `.tokenize()` and outputs a triple of
            `(tokens, start_offsets, limit_offsets)` insead of just tokens.
            Note that when following `strip_diacritics` is set to True, returning
            offsets is not supported now.
          nbest_size: A scalar for sampling:
            nbest_size = {0,1}: No sampling is performed. (default)
            nbest_size > 1: samples from the nbest_size results.
            nbest_size < 0: assuming that nbest_size is infinite and samples
               from the all hypothesis (lattice) using
               forward-filtering-and-backward-sampling algorithm.
          alpha: A scalar for a smoothing parameter. Inverse temperature for
            probability rescaling.
          strip_diacritics: Whether to strip diacritics or not. Note that stripping
            diacritics requires additional text normalization and dropping bytes,
            which makes it impossible to keep track of the offsets now. Hence
            when `strip_diacritics` is set to True, we don't yet support
            `tokenize_with_offsets`. NOTE: New models are encouraged to put this
            into custom normalization rules for the Sentencepiece model itself to
            avoid this extra step and the limitation regarding offsets.
          **kwargs: standard arguments to `Layer()`.
        Raises:
          ImportError: if importing tensorflow_text failed.
        """
        super().__init__(**kwargs)
        if bool(model_file_path) == bool(model_serialized_proto):
            raise ValueError("Exact one of `model_file_path` and " "`model_serialized_proto` can be specified.")
        # TODO(b/181866850): Support tokenize_with_offsets for strip_diacritics=True
        if tokenize_with_offsets and strip_diacritics:
            raise ValueError("`tokenize_with_offsets` is not supported when " "`strip_diacritics` is set to True.")
        if model_file_path:
            self._model_serialized_proto = tf.io.gfile.GFile(model_file_path, "rb").read()
        else:
            self._model_serialized_proto = model_serialized_proto

        self._vocab = get_vocab(self._model_serialized_proto)
        self._lower_case = lower_case
        self.add_cls_sep = add_cls_sep
        self.tokenize_with_offsets = tokenize_with_offsets

        if special_tokens:
            self._special_tokens_dict = self._create_special_tokens_dict(special_tokens)
        else:
            logging.info("If no special tokens , please set `special_tokens=None`")
            self._special_tokens_dict = {}

        if self.add_cls_sep:
            if self.tokenize_with_offsets:
                raise ValueError("`add_cls_sep` not supported with `tokenize_with_offsets")
            assert cls_token is not None
            assert sep_token is not None
            assert cls_token in self._special_tokens_dict
            assert sep_token in self._special_tokens_dict
            self._cls_id = self._vocab[cls_token]
            self._sep_id = self._vocab[sep_token]

        self._nbest_size = nbest_size
        self._alpha = alpha
        self._strip_diacritics = strip_diacritics
        self._tokenizer = self._create_tokenizer()

    def _create_tokenizer(self):
        return tf_text.SentencepieceTokenizer(
            model=self._model_serialized_proto, out_type=tf.int32, nbest_size=self._nbest_size, alpha=self._alpha
        )

    @property
    def vocab_size(self):
        return self._tokenizer.vocab_size()

    def call(self, inputs):
        """Calls `text.SentencepieceTokenizer` on inputs.
        Args:
          inputs: A string Tensor of shape `(batch_size,)`.
        Returns:
          One or three of RaggedTensors if tokenize_with_offsets is False or True,
          respectively. These are
          tokens: A RaggedTensor of shape `[batch_size, (pieces)]` and type `int32`.
            `tokens[i,j]` contains the j-th piece in the i-th input.
          start_offsets, limit_offsets: If `tokenize_with_offsets` is True,
            RaggedTensors of type `int64` with the same indices as tokens.
            Element `[i,j]` contains the byte offset at the start, or past the
            end, resp., for the j-th piece in the i-th input.
        """

        inputs = tf.squeeze(inputs['text'], axis=0)
        if self._strip_diacritics:
            if self.tokenize_with_offsets:
                raise ValueError(
                    "`tokenize_with_offsets` is not supported yet when "
                    "`strip_diacritics` is set to True (b/181866850)."
                )
            inputs = tf_text.normalize_utf8(inputs, "NFD")
            inputs = tf.strings.regex_replace(inputs, r"\p{Mn}", "")

        if self._lower_case:
            inputs = tf_text.case_fold_utf8(inputs)

        # Prepare to reshape the result to work around broken shape inference.
        batch_size = tf.shape(inputs)[0]

        def _reshape(rt):
            values = rt.values
            row_splits = rt.row_splits
            row_splits = tf.reshape(row_splits, [batch_size + 1])
            return tf.RaggedTensor.from_row_splits(values, row_splits)

        # Call the tokenizer.
        if self.tokenize_with_offsets:
            tokens, start_offsets, limit_offsets = self._tokenizer.tokenize_with_offsets(inputs)

            tokens = _reshape(tokens)
            start_offsets = _reshape(start_offsets)
            limit_offsets = _reshape(limit_offsets)

            return {'input_ids': tokens, 'start_offsets': start_offsets, 'limit_offsets': limit_offsets}
        else:
            tokens = self._tokenizer.tokenize(inputs)
            tokens = _reshape(tokens)
            if self.add_cls_sep:
                tokens = self._add_cls_sep(tokens)
            return {'input_ids': tokens}

    def get_config(self):
        # Skip in tf.saved_model.save(); fail if called direcly.
        # raise NotImplementedError("TODO(b/170480226): implement")
        pass

    def _add_cls_sep(self, tokens):
        return tf_text.combine_segments([tokens], start_of_sequence_id=self._cls_id, end_of_segment_id=self._sep_id)[0]

    def get_special_tokens_dict(self):
        """Returns dict of token ids, keyed by standard names for their purpose.
        Returns:
          A dict from Python strings to Python integers. Each key is a standard
          name for a special token describing its use. (For example, "padding_id"
          is what Sentencepiece calls "<pad>" but others may call "[PAD]".)
          The corresponding value is the integer token id. If a special token
          is not found, its entry is omitted from the dict.
          The supported keys and tokens are:
            * start_of_sequence_id: looked up from "[CLS]"
            * end_of_segment_id: looked up from "[SEP]"
            * padding_id: looked up from "<pad>"
            * mask_id: looked up from "[MASK]"
            * vocab_size: one past the largest token id used
        """
        return self._special_tokens_dict

    def _create_special_tokens_dict(self, special_tokens):
        """Create special tokens"""
        special_vocab = {}
        for tok in special_tokens:
            if tok not in self._vocab:
                raise ValueError("Special token `{}` not found in vocab".format(tok))
            special_vocab[tok] = self._vocab[tok]
        return special_vocab

    def get_model(self):
        """Convert Keras Layer to Model"""
        from tf_transformers.core import LegacyModel

        inputs = {
            "text": tf.keras.layers.Input(
                shape=(None,),
                batch_size=1,
                dtype=tf.string,
                name="text",
            )
        }
        layer_outputs = self(inputs)
        model = LegacyModel(inputs=inputs, outputs=layer_outputs, name='sentencepice')
        return model
