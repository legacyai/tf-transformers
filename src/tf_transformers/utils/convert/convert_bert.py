import tensorflow as tf
from absl import logging

logging.set_verbosity("INFO")


def convert_bert_hf_to_tf_transformers(model_hf, model_tf_transformers, config):
    """
    Args:
        model_hf: HuggingFace Model (TF)
        model_tf_transformers: tf_transformers model/layer
        config: dict

    """
    # From vars (Transformer variables)
    from_model_vars = [
        "tf_bert_model/bert/encoder/layer_._{}/attention/self/query/kernel:0",
        "tf_bert_model/bert/encoder/layer_._{}/attention/self/query/bias:0",
        "tf_bert_model/bert/encoder/layer_._{}/attention/self/key/kernel:0",
        "tf_bert_model/bert/encoder/layer_._{}/attention/self/key/bias:0",
        "tf_bert_model/bert/encoder/layer_._{}/attention/self/value/kernel:0",
        "tf_bert_model/bert/encoder/layer_._{}/attention/self/value/bias:0",
        "tf_bert_model/bert/encoder/layer_._{}/attention/output/dense/kernel:0",
        "tf_bert_model/bert/encoder/layer_._{}/attention/output/dense/bias:0",
        "tf_bert_model/bert/encoder/layer_._{}/attention/output/LayerNorm/gamma:0",
        "tf_bert_model/bert/encoder/layer_._{}/attention/output/LayerNorm/beta:0",
        "tf_bert_model/bert/encoder/layer_._{}/intermediate/dense/kernel:0",
        "tf_bert_model/bert/encoder/layer_._{}/intermediate/dense/bias:0",
        "tf_bert_model/bert/encoder/layer_._{}/output/dense/kernel:0",
        "tf_bert_model/bert/encoder/layer_._{}/output/dense/bias:0",
        "tf_bert_model/bert/encoder/layer_._{}/output/LayerNorm/gamma:0",
        "tf_bert_model/bert/encoder/layer_._{}/output/LayerNorm/beta:0",
    ]

    # To vars (Transformer variables)
    to_model_vars = [
        "tf_transformers/bert/transformer/layer_{}/self_attention/query/kernel:0",
        "tf_transformers/bert/transformer/layer_{}/self_attention/query/bias:0",
        "tf_transformers/bert/transformer/layer_{}/self_attention/key/kernel:0",
        "tf_transformers/bert/transformer/layer_{}/self_attention/key/bias:0",
        "tf_transformers/bert/transformer/layer_{}/self_attention/value/kernel:0",
        "tf_transformers/bert/transformer/layer_{}/self_attention/value/bias:0",
        "tf_transformers/bert/transformer/layer_{}/self_attention_output/kernel:0",
        "tf_transformers/bert/transformer/layer_{}/self_attention_output/bias:0",
        "tf_transformers/bert/transformer/layer_{}/self_attention_layer_norm/gamma:0",
        "tf_transformers/bert/transformer/layer_{}/self_attention_layer_norm/beta:0",
        "tf_transformers/bert/transformer/layer_{}/intermediate/kernel:0",
        "tf_transformers/bert/transformer/layer_{}/intermediate/bias:0",
        "tf_transformers/bert/transformer/layer_{}/output/kernel:0",
        "tf_transformers/bert/transformer/layer_{}/output/bias:0",
        "tf_transformers/bert/transformer/layer_{}/output_layer_norm/gamma:0",
        "tf_transformers/bert/transformer/layer_{}/output_layer_norm/beta:0",
    ]

    # Simple Assertion
    assert len(from_model_vars) == len(to_model_vars)

    # This dictionary maps from -> to dict names
    mapping_dict = {}
    for index in range(len(from_model_vars)):
        for i in range(config["num_hidden_layers"]):
            mapping_dict[from_model_vars[index].format(i)] = to_model_vars[index].format(i)

    # Word Embeddings
    mapping_dict[
        "tf_bert_model/bert/embeddings/word_embeddings/weight:0"
    ] = "tf_transformers/bert/word_embeddings/embeddings:0"
    # Positional Embedding
    mapping_dict[
        "tf_bert_model/bert/embeddings/position_embeddings/embeddings:0"
    ] = "tf_transformers/bert/positional_embeddings/embeddings:0"
    # Type Embeddings
    mapping_dict[
        "tf_bert_model/bert/embeddings/token_type_embeddings/embeddings:0"
    ] = "tf_transformers/bert/type_embeddings/embeddings:0"
    mapping_dict[
        "tf_bert_model/bert/embeddings/LayerNorm/gamma:0"
    ] = "tf_transformers/bert/embeddings/layer_norm/gamma:0"
    mapping_dict["tf_bert_model/bert/embeddings/LayerNorm/beta:0"] = "tf_transformers/bert/embeddings/layer_norm/beta:0"
    mapping_dict["tf_bert_model/bert/pooler/dense/kernel:0"] = "tf_transformers/bert/pooler_transform/kernel:0"
    mapping_dict["tf_bert_model/bert/pooler/dense/bias:0"] = "tf_transformers/bert/pooler_transform/bias:0"

    # HF model variable name to variable values, for fast retrieval
    from_to_variable_dict = {var.name: var for var in model_hf.variables}
    del model_hf
    logging.info("Deleteing huggingface model for saving memory")

    # We need variable name to the index where it is stored inside tf_transformers model
    tf_transformers_model_index_dict = {}
    for index, var in enumerate(model_tf_transformers.variables):
        tf_transformers_model_index_dict[var.name] = index

    # Start assigning HF values to tf_transformers
    # assigned_map and assigned_map_values are used for sanity check if needed
    assigned_map = []
    assigned_map_values = []
    for original_var, legacy_var in mapping_dict.items():

        index = tf_transformers_model_index_dict[legacy_var]

        if "query/kernel:0" in legacy_var or "key/kernel:0" in legacy_var or "value/kernel:0" in legacy_var:
            # huggingface (2D) to tf_transformers (3D)
            model_tf_transformers.variables[index].assign(
                tf.reshape(
                    from_to_variable_dict.get(original_var),
                    (
                        config["embedding_size"],
                        config["num_attention_heads"],
                        config["attention_head_size"],
                    ),
                )
            )
            assigned_map.append((original_var, legacy_var))
            continue
        if "query/bias:0" in legacy_var or "key/bias:0" in legacy_var or "value/bias:0" in legacy_var:
            # huggingface (2D) to tf_transformers (3D)
            model_tf_transformers.variables[index].assign(
                tf.reshape(
                    from_to_variable_dict.get(original_var),
                    (
                        config["num_attention_heads"],
                        config["attention_head_size"],
                    ),
                )
            )
            assigned_map.append((original_var, legacy_var))
            continue

        model_tf_transformers.variables[index].assign(from_to_variable_dict.get(original_var))
        assigned_map.append((original_var, legacy_var))

    logging.info("Done assigning variables weights. Total {}".format(len(assigned_map)))
