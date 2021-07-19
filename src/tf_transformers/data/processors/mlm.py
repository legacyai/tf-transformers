import tensorflow_text as tf_text


def dynamic_masking_from_features(
    max_seq_len, max_predictions_per_batch, vocab_size, cls_id, sep_id, unk_id, pad_id, mask_id
):

    """Dynamic Masking from input_ids (saved as tfrecord)"""
    # Truncate inputs to a maximum length.
    trimmer = tf_text.RoundRobinTrimmer(max_seq_length=max_seq_len)

    # Random Selector
    random_selector = tf_text.RandomItemSelector(
        max_selections_per_batch=max_predictions_per_batch,
        selection_rate=0.2,
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
