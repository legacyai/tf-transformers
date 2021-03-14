import tensorflow as tf


def assign_zeros_to_K_V(matrix, input_ids_orig, batch_size, max_len):
    """In the case of variable batch decoding, we need to assign zeros
     to -1 postions (padded)
    This is required only in step 0 (1st step) of the decoder

    Arguments:
        matrix: [num_layers, batch_size, num_attention_heads,
        sequence_length, attention_state]
        input_ids_orig: [batch_size x sequence_length]
        batch_size: batch_size
        max_len: max_len of the batch

    Returns:
        matrix: After assigning zeros to all padded positions (-1)

    """

    # Split matrxi by axis 1 (by batch_size)
    matrix_splits = tf.split(matrix, num_or_size_splits=batch_size, axis=1)
    # calculate sum of -1 entries evrywhere (0 if no need of padding)
    padded_positions = tf.reduce_sum(tf.cast(tf.equal(input_ids_orig, -1), tf.float32), axis=1)
    for index, pos in enumerate(padded_positions):

        if pos == 0:
            continue

        matrix_batch = matrix_splits[index].numpy()
        # max_len - pos, (if pos = 3 (3 -1 entries) , max_len = 11, we get 8)
        # So we need to assign zeros to 8: positions
        index_to_start_from = tf.cast(max_len - pos, tf.int32)
        matrix_batch[:, :, :, index_to_start_from.numpy() :, :] = 0.0
        matrix_splits[index] = matrix_batch

    matrix = tf.concat(matrix_splits, axis=1)
    return matrix


def _log_prob_from_logits(logits, axis=2):
    return logits - tf.reduce_logsumexp(logits, axis=axis, keepdims=True)


def _gather_beams(state, beam_indices, batch_size, new_beam_size):
    """Gather beams from nested structure of tensors.

    Each tensor in nested represents a batch of beams, where beam refers to a
    single search state (beam search involves searching through multiple states
    in parallel).

    This function is used to gather the top beams, specified by
    beam_indices, from the nested tensors.

    Args:
        nested: Nested structure (tensor, list, tuple or dict) containing tensors
          with shape [batch_size, beam_size, ...].
        beam_indices: int32 tensor with shape [batch_size, new_beam_size]. Each
         value in beam_indices must be between [0, beam_size), and are not
         necessarily unique.
        batch_size: int size of batch
        new_beam_size: int number of beams to be pulled from the nested tensors.

    Returns:
        Nested structure containing tensors with shape
        [batch_size, new_beam_size, ...]
    """
    # Computes the i'th coodinate that contains the batch index for gather_nd.
    # Batch pos is a tensor like [[0,0,0,0,],[1,1,1,1],..].
    batch_pos = tf.range(batch_size * new_beam_size) // new_beam_size
    batch_pos = tf.reshape(batch_pos, [batch_size, new_beam_size])

    # Create coordinates to be passed to tf.gather_nd. Stacking creates a tensor
    # with shape [batch_size, beam_size, 2], where the last dimension contains
    # the (i, j) gathering coordinates.
    coordinates = tf.stack([batch_pos, beam_indices], axis=2)

    return tf.gather_nd(state, coordinates), coordinates


def top_k_logits(logits, k):
    """From OpenAI implementation"""

    values, _ = tf.nn.top_k(logits, k=k)
    # min_values = values[:, -1, tf.newaxis]
    min_values = tf.expand_dims(tf.reduce_min(values, axis=1), 1)
    return tf.where(
        logits < min_values,
        tf.ones_like(logits, dtype=logits.dtype) * -1e10,
        logits,
    )


def top_p_logits(logits, p):
    """Nucleus sampling"""
    # batch, _ = logits.shape.as_list()
    batch = tf.shape(logits)[0]
    sorted_logits = tf.sort(logits, direction="DESCENDING", axis=-1)
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
    indices = tf.stack(
        [
            tf.range(0, batch),
            # number of indices to include
            tf.maximum(tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32), axis=-1) - 1, 0),
        ],
        axis=-1,
    )
    min_values = tf.gather_nd(sorted_logits, indices)
    return tf.where(
        logits < tf.expand_dims(min_values, 1),
        tf.ones_like(logits) * -1e10,
        logits,
    )
