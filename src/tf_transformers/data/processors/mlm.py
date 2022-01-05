import tensorflow as tf
import tensorflow_text as tf_text

def dynamic_masking_from_features(
    max_seq_len: int, max_predictions_per_batch: int, vocab_size: int, cls_id:int, sep_id: int, unk_id: int, pad_id, mask_id: int
):
    """Dynamic Masking from input_ids (saved as tfrecord)

    Args:
        max_seq_len (:obj:`int`): Maximum Sequence Length
        max_predictions_per_batch (:obj:`int`): Maximum predictions per batch
        vocab_size (:obj:`int`): Vocabulary size
        cls_id (:obj:`int`): CLS token
        sep_id (:obj:`int`): SEP token
        unk_id (:obj:`int`): UNK token
        pad_id (:obj:`int`): PAD token
        mask_id (:obj:`int`): MASK token

    Returns:
        Function which return Tuple (inputs, labels)
    """
    # Truncate inputs to a maximum length.
    trimmer = tf_text.RoundRobinTrimmer(max_seq_length=max_seq_len-2)

    # Random Selector
    random_selector = tf_text.RandomItemSelector(
        max_selections_per_batch=max_predictions_per_batch,
        selection_rate=0.1,
        unselectable_ids=[cls_id, sep_id, unk_id, pad_id],
    )

    # Mask Value chooser (Encapsulates the BERT MLM token selection logic)
    mask_values_chooser = tf_text.MaskValuesChooser(vocab_size, mask_id, 0.8)

    def map_mlm(item):
        """Dynamic MLM, better to call after batching"""
        segments = item['input_ids']
        trimmed_segments = trimmer.trim([segments])

        # We replace trimmer with slice [:_MAX_SEQ_LEN-2] operation # 2 to add CLS and SEP
        # input_ids = item['input_ids'][:_MAX_SEQ_LEN-2]

        # Combine segments, get segment ids and add special tokens.
        segments_combined, segment_ids = tf_text.combine_segments(
            trimmed_segments, start_of_sequence_id=cls_id, end_of_segment_id=sep_id
        )

        # We replace segment with concat
        # input_ids = tf.concat([[_START_TOKEN], input_ids, [_END_TOKEN]], axis=0)

        # Apply dynamic masking
        masked_token_ids, masked_pos, masked_lm_ids = tf_text.mask_language_model(
            segments_combined, item_selector=random_selector, mask_values_chooser=mask_values_chooser
        )

        # Prepare and pad combined segment inputs
        input_word_ids, input_mask = tf_text.pad_model_inputs(masked_token_ids, max_seq_length=max_seq_len)
        input_type_ids, _ = tf_text.pad_model_inputs(segment_ids, max_seq_length=max_seq_len)

        # Prepare and pad masking task inputs
        # Masked lm weights will mask the weights
        masked_lm_positions, masked_lm_weights = tf_text.pad_model_inputs(
            masked_pos, max_seq_length=max_predictions_per_batch
        )
        masked_lm_ids, _ = tf_text.pad_model_inputs(masked_lm_ids, max_seq_length=max_predictions_per_batch)

        inputs = {}
        inputs['input_ids'] = input_word_ids
        inputs['input_type_ids'] = input_type_ids
        inputs['input_mask'] = input_mask
        inputs['masked_lm_positions'] = masked_lm_positions

        labels = {}
        labels['masked_lm_labels'] = masked_lm_ids
        labels['masked_lm_weights'] = masked_lm_weights  # Mask

        return (inputs, labels)

    return map_mlm


def dynamic_prefix_lm_from_features(max_seq_len, cls_id, sep_id):
    """Prefix Causal LM"""
    import warnings
    warnings.warn("This function `dynamic_prefix_lm_from_features` will be deprecated soon.")
    def dynamic_map_prefix(item):
        input_ids = item['input_ids']
        input_ids = input_ids[: max_seq_len - 1]  # we need -2 for cls and sep, but in causal LM we shift one pos
        # so we use -1, length input_ids = max_seq_len + 1
        # Add cls sep
        input_ids = tf.concat([[cls_id], input_ids, [sep_id]], axis=0)
        labels = input_ids[1:]  # exclude first word till last
        input_ids = input_ids[:-1]  # exclude last word

        input_seq_length = tf.shape(input_ids)[0]
        sentence_length = tf.random.uniform(minval=1, maxval=input_seq_length, shape=(1,), dtype=tf.int32)[0]
        remaining_length = input_seq_length - sentence_length
        input_mask = tf.concat(
            [tf.ones(shape=(sentence_length,), dtype=tf.int32), tf.zeros(shape=(remaining_length,), dtype=tf.int32)],
            axis=0,
        )
        # Opposite to input_mask
        labels_mask = tf.concat(
            [tf.zeros(shape=(sentence_length,), dtype=tf.int32), tf.ones(shape=(remaining_length,), dtype=tf.int32)],
            axis=0,
        )

        # input type ids
        input_type_ids = tf.zeros_like(input_ids)

        inputs = {
            'input_ids': input_ids,
            'input_type_ids': input_type_ids,
            'input_mask': input_mask,
            'masked_lm_positions': tf.range(tf.shape(input_ids)[0]),
        }

        outputs = {'masked_lm_labels': labels, 'masked_lm_weights': labels_mask}
        return inputs, outputs

    return dynamic_map_prefix


def dynamic_causal_lm_from_features(max_seq_len, cls_id, sep_id):
    """Causal LM"""

    def dynamic_map_causal(item):
        input_ids = item['input_ids']
        input_ids = input_ids[: max_seq_len - 1]  # we need -2 for cls and sep, but in causal LM we shift one pos
        # so we use -1, length input_ids = max_seq_len + 1
        # Add cls sep
        input_ids = tf.concat([[cls_id], input_ids, [sep_id]], axis=0)
        labels = input_ids[1:]  # exclude first word till last
        input_ids = input_ids[:-1]  # exclude last word
        labels_mask = tf.ones_like(input_ids)
        input_mask = labels_mask
        # input type ids
        input_type_ids = tf.zeros_like(input_ids)

        inputs = {
            'input_ids': input_ids,
            'input_type_ids': input_type_ids,
            'input_mask': input_mask,
            'masked_lm_positions': tf.range(tf.shape(input_ids)[0]),
        }

        outputs = {'masked_lm_labels': labels, 'masked_lm_weights': labels_mask}

        return inputs, outputs

    return dynamic_map_causal
