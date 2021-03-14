import six
import tensorflow as tf

from tf_transformers import activations


# TODO(hongkuny): consider moving custom string-map lookup to keras api.
def get_activation(identifier):
    """Maps a identifier to a Python function, e.g., "relu" => `tf.nn.relu`.

    It checks string first and if it is one of customized activation not in TF,
    the corresponding activation will be returned. For non-customized activation
    names and callable identifiers, always fallback to tf.keras.activations.get.

    Args:
      identifier: String name of the activation function or callable.

    Returns:
      A Python function corresponding to the activation function.
    """
    if isinstance(identifier, six.string_types):
        name_to_fn = {
            "gelu": activations.gelu,
            "simple_swish": activations.simple_swish,
            "hard_swish": activations.hard_swish,
            "identity": activations.identity,
            "relu": tf.keras.activations.relu,
        }
        identifier = str(identifier).lower()
        if identifier in name_to_fn:
            return tf.keras.activations.get(name_to_fn[identifier])
    return tf.keras.activations.get(identifier)
