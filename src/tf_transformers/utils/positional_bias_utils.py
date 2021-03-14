import math

import tensorflow as tf


def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    """
    Adapted from Mesh Tensorflow:
    https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

    Translate relative position to a bucket number for relative attention.
    The relative position is defined as memory_position - query_position, i.e.
    the distance in tokens from the attending position to the attended-to
    position.  If bidirectional=False, then positive relative positions are
    invalid.
    We use smaller buckets for small absolute relative_position and larger buckets
    for larger absolute relative_positions.  All relative positions >=max_distance
    map to the same bucket.  All relative positions <=-max_distance map to the
    same bucket.  This should allow for more graceful generalization to longer
    sequences than the model has been trained on.
    Args:
        relative_position: an int32 Tensor
        bidirectional: a boolean - whether the attention is bidirectional
        num_buckets: an integer
        max_distance: an integer
    Returns:
        a Tensor with the same shape as relative_position, containing int32
        values in the range [0, num_buckets)
    """
    ret = 0
    n = -relative_position
    if bidirectional:
        num_buckets //= 2
        ret += tf.dtypes.cast(tf.math.less(n, 0), tf.int32) * num_buckets
        n = tf.math.abs(n)
    else:
        n = tf.math.maximum(n, 0)
    # now n is in the range [0, inf)
    max_exact = num_buckets // 2
    is_small = tf.math.less(n, max_exact)
    val_if_large = max_exact + tf.dtypes.cast(
        tf.math.log(tf.dtypes.cast(n, tf.float32) / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact),
        tf.int32,
    )
    val_if_large = tf.math.minimum(val_if_large, num_buckets - 1)
    ret += tf.where(is_small, n, val_if_large)
    return ret


def compute_positional_bias(qlen, klen, bidirectional=True, num_buckets=32):
    """ Compute binned relative position bias """
    context_position = tf.range(qlen)[:, None]
    memory_position = tf.range(klen)[None, :]
    relative_position = memory_position - context_position  # shape (qlen, klen)
    rp_bucket = _relative_position_bucket(
        relative_position,
        bidirectional=bidirectional,
        num_buckets=num_buckets,
    )
    return rp_bucket
