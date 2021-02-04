import tensorflow as tf
from absl import logging

logging.set_verbosity("INFO")


def convert_roberta_hf_to_tf_transformers(model_hf, model_tf_transformers, config):
    """
    Args:
        model_hf: HuggingFace Model (TF)
        model_tf_transformers: tf_transformers model/layer
        config: dict

    """
    # From vars (Transformer variables)
    from_model_vars = [
        "tf_roberta_model/roberta/encoder/layer_._{}/attention/self/query/kernel:0",
        "tf_roberta_model/roberta/encoder/layer_._{}/attention/self/query/bias:0",
        "tf_roberta_model/roberta/encoder/layer_._{}/attention/self/key/kernel:0",
        "tf_roberta_model/roberta/encoder/layer_._{}/attention/self/key/bias:0",
        "tf_roberta_model/roberta/encoder/layer_._{}/attention/self/value/kernel:0",
        "tf_roberta_model/roberta/encoder/layer_._{}/attention/self/value/bias:0",
        "tf_roberta_model/roberta/encoder/layer_._{}/attention/output/dense/kernel:0",
        "tf_roberta_model/roberta/encoder/layer_._{}/attention/output/dense/bias:0",
        "tf_roberta_model/roberta/encoder/layer_._{}/attention/output/LayerNorm/gamma:0",
        "tf_roberta_model/roberta/encoder/layer_._{}/attention/output/LayerNorm/beta:0",
        "tf_roberta_model/roberta/encoder/layer_._{}/intermediate/dense/kernel:0",
        "tf_roberta_model/roberta/encoder/layer_._{}/intermediate/dense/bias:0",
        "tf_roberta_model/roberta/encoder/layer_._{}/output/dense/kernel:0",
        "tf_roberta_model/roberta/encoder/layer_._{}/output/dense/bias:0",
        "tf_roberta_model/roberta/encoder/layer_._{}/output/LayerNorm/gamma:0",
        "tf_roberta_model/roberta/encoder/layer_._{}/output/LayerNorm/beta:0",
    ]

    # To vars (Transformer variables)
    to_model_vars = [
        "tf_transformers/roberta/transformer/layer_{}/self_attention/query/kernel:0",
        "tf_transformers/roberta/transformer/layer_{}/self_attention/query/bias:0",
        "tf_transformers/roberta/transformer/layer_{}/self_attention/key/kernel:0",
        "tf_transformers/roberta/transformer/layer_{}/self_attention/key/bias:0",
        "tf_transformers/roberta/transformer/layer_{}/self_attention/value/kernel:0",
        "tf_transformers/roberta/transformer/layer_{}/self_attention/value/bias:0",
        "tf_transformers/roberta/transformer/layer_{}/self_attention_output/kernel:0",
        "tf_transformers/roberta/transformer/layer_{}/self_attention_output/bias:0",
        "tf_transformers/roberta/transformer/layer_{}/self_attention_layer_norm/gamma:0",
        "tf_transformers/roberta/transformer/layer_{}/self_attention_layer_norm/beta:0",
        "tf_transformers/roberta/transformer/layer_{}/intermediate/kernel:0",
        "tf_transformers/roberta/transformer/layer_{}/intermediate/bias:0",
        "tf_transformers/roberta/transformer/layer_{}/output/kernel:0",
        "tf_transformers/roberta/transformer/layer_{}/output/bias:0",
        "tf_transformers/roberta/transformer/layer_{}/output_layer_norm/gamma:0",
        "tf_transformers/roberta/transformer/layer_{}/output_layer_norm/beta:0",
    ]

    # Simple Assertion
    assert len(from_model_vars) == len(to_model_vars)
    mapping_dict = {}

    for index in range(len(from_model_vars)):
        for i in range(config["num_hidden_layers"]):
            mapping_dict[from_model_vars[index].format(i)] = to_model_vars[index].format(i)

    # Word Embeddings
    mapping_dict[
        "tf_roberta_model/roberta/embeddings/word_embeddings/weight:0"
    ] = "tf_transformers/roberta/word_embeddings/embeddings:0"
    # Positional Embedding
    mapping_dict[
        "tf_roberta_model/roberta/embeddings/position_embeddings/embeddings:0"
    ] = "tf_transformers/roberta/positional_embeddings/embeddings:0"
    # Type Embeddings
    mapping_dict[
        "tf_roberta_model/roberta/embeddings/token_type_embeddings/embeddings:0"
    ] = "tf_transformers/roberta/type_embeddings/embeddings:0"
    mapping_dict[
        "tf_roberta_model/roberta/embeddings/LayerNorm/gamma:0"
    ] = "tf_transformers/roberta/embeddings/layer_norm/gamma:0"
    mapping_dict[
        "tf_roberta_model/roberta/embeddings/LayerNorm/beta:0"
    ] = "tf_transformers/roberta/embeddings/layer_norm/beta:0"
    mapping_dict["tf_roberta_model/roberta/pooler/dense/kernel:0"] = "tf_transformers/roberta/pooler_transform/kernel:0"
    mapping_dict["tf_roberta_model/roberta/pooler/dense/bias:0"] = "tf_transformers/roberta/pooler_transform/bias:0"

    from_to_variable_dict = {var.name: var for var in model_hf.variables}
    del model_hf
    logging.info("Deleteing huggingface model for saving memory")

    tf_transformers_model_index_dict = {}
    for index, var in enumerate(model_tf_transformers.variables):
        tf_transformers_model_index_dict[var.name] = index

    # legacy_ai <-- HuggingFace
    assigned_map = []
    # assigned_map_values = []
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

        # Robert ave 514 embeddings, we are taking [2:] (512)
        if "positional_embeddings" in legacy_var:
            logging.info("We slice Positional Embeddings from 514 to 512")
            model_tf_transformers.variables[index].assign(from_to_variable_dict.get(original_var)[2:])
            assigned_map.append((original_var, legacy_var))
            continue

        model_tf_transformers.variables[index].assign(from_to_variable_dict.get(original_var))
        assigned_map.append((original_var, legacy_var))

    logging.info("Done assigning variables weights . Total {}".format(len(assigned_map)))
