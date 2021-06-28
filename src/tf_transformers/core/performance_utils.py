import tensorflow as tf

# Map string to TensorFlow dtype
DTYPE_MAP = {
    "fp16": tf.float16,
    "bf16": tf.bfloat16,
    "fp32": tf.float32,
}


def get_tf_dtype(dtype):
    return DTYPE_MAP[dtype]


def is_float16(dtype):
    if dtype in [tf.float16, tf.bfloat16]:
        return True
    return False


def set_mixed_precision_policy(dtype):
    """Sets mix precision policy."""
    if dtype == tf.float16:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    elif dtype == tf.bfloat16:
        tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
    elif dtype == tf.float32:
        tf.keras.mixed_precision.set_global_policy('float32')
    else:
        raise ValueError('Unexpected dtype: %s' % dtype)


def configure_optimizer(optimizer, use_float16=False, use_graph_rewrite=False, loss_scale='dynamic'):
    """Configures optimizer object with performance options."""
    if use_float16:
        if loss_scale == 'dynamic':
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        else:
            # loss_scale is a number. We interpret that as a fixed loss scale.
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, dynamic=False, initial_scale=loss_scale)

    if use_graph_rewrite:
        # Note: the model dtype must be 'float32', which will ensure
        # tf.keras.mixed_precision and enable_mixed_precision_graph_rewrite do not
        # double up.
        optimizer = tf.compat.v1.mixed_precision.enable_mixed_precision_graph_rewrite(optimizer)
    return optimizer
