from functools import wraps

import tensorflow as tf


def pad_dataset(tokenizer_fn):
    """We will pad the data.
    Based on name of the dataset, we will pad it accordingly

    Args:
        tokenizer_fn ([type]): [A function which returns dict of list of list]

    Returns:
        [type]: [description]
    """

    @wraps(tokenizer_fn)
    def pad_fn(*args, **kw):
        tokenized_dict = tokenizer_fn(*args, **kw)
        tokenized_dict_ragged = {name: tf.ragged.constant(tensor) for name, tensor in tokenized_dict.items()}
        tokenized_dict_padded = {}
        for name, tensor in tokenized_dict_ragged.items():
            if isinstance(tensor, tf.RaggedTensor):
                if name in ["input_ids", "encoder_input_ids"]:
                    tokenized_dict_padded[name] = tensor.to_tensor(-1)
                elif name in [
                    "input_mask",
                    "input_type_ids",
                    "encoder_input_mask",
                    "encoder_input_type_ids",
                ]:
                    tokenized_dict_padded[name] = tensor.to_tensor(0)
            else:
                tokenized_dict_padded[name] = tensor
        return tokenized_dict_padded

    return pad_fn


def pad_dataset_normal(tokenizer_fn):
    """We will pad the data.
    Based on name of the dataset, we will pad it accordingly

    Args:
        tokenizer_fn ([type]): [A function which returns dict of list of list]

    Returns:
        [type]: [description]
    """

    @wraps(tokenizer_fn)
    def pad_fn(*args, **kw):
        tokenized_dict = tokenizer_fn(*args, **kw)
        tokenized_dict_ragged = {name: tf.ragged.constant(tensor) for name, tensor in tokenized_dict.items()}
        tokenized_dict_padded = {}
        for name, tensor in tokenized_dict_ragged.items():
            if isinstance(tensor, tf.RaggedTensor):
                if name in ["input_ids", "encoder_input_ids"]:
                    tokenized_dict_padded[name] = tensor.to_tensor(0)
                elif name in [
                    "input_mask",
                    "input_type_ids",
                    "encoder_input_mask",
                    "encoder_input_type_ids",
                ]:
                    tokenized_dict_padded[name] = tensor.to_tensor(0)
            else:
                tokenized_dict_padded[name] = tensor
        return tokenized_dict_padded

    return pad_fn


def separate_x_y(dict, x_keys, y_keys):
    """Separate dataset into a tuple (X, Y)

    Args:
        dict ([type]): Each entry in tf dataset
        x_keys ([type]): List of key values
        y_keys ([type]): List of ky values

    Returns:
        tuple of each entry
    """
    X = {}
    Y = {}
    for k, v in dict.items():
        if k in x_keys:
            X[k] = v
            continue
        if k in y_keys:
            Y[k] = v
    return (X, Y)


def pad_ragged(dataset):
    """
    Pad dataset of dict .

    """
    dataset_padded = {}
    for item, tensor in dataset.items():
        if isinstance(tensor, tf.RaggedTensor):
            dataset_padded[item] = tensor.to_tensor()
        else:
            dataset_padded[item] = tensor
    return dataset_padded
