import tensorflow as tf
from absl import logging

logging.set_verbosity("INFO")


def convert_t5_hf_to_tf_transformers(model_hf, model_tf_transformers, config):
    # Encoder Side
    # From vars (Transformer variables)
    from_model_vars = [
        "tf_t5model/encoder/block_._{}/layer_._0/SelfAttention/q/kernel:0",
        "tf_t5model/encoder/block_._{}/layer_._0/SelfAttention/k/kernel:0",
        "tf_t5model/encoder/block_._{}/layer_._0/SelfAttention/v/kernel:0",
        "tf_t5model/encoder/block_._{}/layer_._0/SelfAttention/o/kernel:0",
        "tf_t5model/encoder/block_._{}/layer_._0/layer_norm/weight:0",
        "tf_t5model/encoder/block_._{}/layer_._1/DenseReluDense/wi/kernel:0",
        "tf_t5model/encoder/block_._{}/layer_._1/DenseReluDense/wo/kernel:0",
        "tf_t5model/encoder/block_._{}/layer_._1/layer_norm/weight:0",
    ]

    to_model_vars = [
        "tf_transformers/t5_encoder/transformer/layer_{}/self_attention/query/kernel:0",
        "tf_transformers/t5_encoder/transformer/layer_{}/self_attention/key/kernel:0",
        "tf_transformers/t5_encoder/transformer/layer_{}/self_attention/value/kernel:0",
        "tf_transformers/t5_encoder/transformer/layer_{}/self_attention_output/kernel:0",
        "tf_transformers/t5_encoder/transformer/layer_{}/pre_attention_norm/weight:0",
        "tf_transformers/t5_encoder/transformer/layer_{}/intermediate/kernel:0",
        "tf_transformers/t5_encoder/transformer/layer_{}/output/kernel:0",
        "tf_transformers/t5_encoder/transformer/layer_{}/self_attention_layer_norm/weight:0",
    ]

    # Simple Assertion
    assert len(from_model_vars) == len(to_model_vars)
    mapping_dict = {}

    for index in range(len(from_model_vars)):
        for i in range(config["num_hidden_layers"]):
            mapping_dict[from_model_vars[index].format(i)] = to_model_vars[index].format(i)

    # Only Layer 0
    mapping_dict[
        "tf_t5model/encoder/block_._0/layer_._0/SelfAttention/relative_attention_bias/embeddings:0"
    ] = "tf_transformers/t5_encoder/transformer/layer_0/self_attention/relative_attention_bias/embeddings:0"
    # Word Embedding
    mapping_dict["shared/shared/weight:0"] = "tf_transformers/t5_encoder/word_embeddings/embeddings:0"
    # Final Layer Norm weight
    mapping_dict["tf_t5model/encoder/final_layer_norm/weight:0"] = "tf_transformers/t5_encoder/last_layer_norm/weight:0"

    from_to_variable_dict = {var.name: var for var in model_hf.variables}
    # del model_hf
    logging.info("Deleteing huggingface model for saving memory")

    tf_transformers_model_index_dict = {}
    for index, var in enumerate(model_tf_transformers.variables):
        tf_transformers_model_index_dict[var.name] = index

    # legacy_ai <-- hub
    assigned_map = []
    assigned_map_values = []
    for original_var, legacy_var in mapping_dict.items():

        index = tf_transformers_model_index_dict[legacy_var]
        # If not in mapping_dict, then mostly it is from attention layer
        if "query/kernel:0" in legacy_var or "key/kernel:0" in legacy_var or "value/kernel:0" in legacy_var:
            # hub (2D) to tf_transformers (3D)
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

        model_tf_transformers.variables[index].assign(from_to_variable_dict.get(original_var))
        assigned_map.append((original_var, legacy_var))

    logging.info("Done assigning ENCODER variables weights {}".format(len(assigned_map)))

    # Decoder Side
    # From vars (Transformer variables)
    from_model_vars = [
        "tf_t5model/decoder/block_._{}/layer_._0/SelfAttention/q/kernel:0",
        "tf_t5model/decoder/block_._{}/layer_._0/SelfAttention/k/kernel:0",
        "tf_t5model/decoder/block_._{}/layer_._0/SelfAttention/v/kernel:0",
        "tf_t5model/decoder/block_._{}/layer_._0/SelfAttention/o/kernel:0",
        "tf_t5model/decoder/block_._{}/layer_._0/layer_norm/weight:0",
        "tf_t5model/decoder/block_._{}/layer_._1/EncDecAttention/q/kernel:0",
        "tf_t5model/decoder/block_._{}/layer_._1/EncDecAttention/k/kernel:0",
        "tf_t5model/decoder/block_._{}/layer_._1/EncDecAttention/v/kernel:0",
        "tf_t5model/decoder/block_._{}/layer_._1/EncDecAttention/o/kernel:0",
        "tf_t5model/decoder/block_._{}/layer_._1/layer_norm/weight:0",
        "tf_t5model/decoder/block_._{}/layer_._2/DenseReluDense/wi/kernel:0",
        "tf_t5model/decoder/block_._{}/layer_._2/DenseReluDense/wo/kernel:0",
        "tf_t5model/decoder/block_._{}/layer_._2/layer_norm/weight:0",
    ]

    to_model_vars = [
        "tf_transformers/t5_decoder/transformer/layer_{}/self_attention/query/kernel:0",
        "tf_transformers/t5_decoder/transformer/layer_{}/self_attention/key/kernel:0",
        "tf_transformers/t5_decoder/transformer/layer_{}/self_attention/value/kernel:0",
        "tf_transformers/t5_decoder/transformer/layer_{}/self_attention_output/kernel:0",
        "tf_transformers/t5_decoder/transformer/layer_{}/pre_attention_norm/weight:0",
        "tf_transformers/t5_decoder/transformer/layer_{}/cross_attention/query/kernel:0",
        "tf_transformers/t5_decoder/transformer/layer_{}/cross_attention/key/kernel:0",
        "tf_transformers/t5_decoder/transformer/layer_{}/cross_attention/value/kernel:0",
        "tf_transformers/t5_decoder/transformer/layer_{}/cross_attention_output/kernel:0",
        "tf_transformers/t5_decoder/transformer/layer_{}/pre_cross_attention_norm/weight:0",
        "tf_transformers/t5_decoder/transformer/layer_{}/intermediate/kernel:0",
        "tf_transformers/t5_decoder/transformer/layer_{}/output/kernel:0",
        "tf_transformers/t5_decoder/transformer/layer_{}/self_attention_layer_norm/weight:0",
    ]

    # Simple Assertion
    assert len(from_model_vars) == len(to_model_vars)
    mapping_dict = {}

    for index in range(len(from_model_vars)):
        for i in range(config["num_hidden_layers"]):
            mapping_dict[from_model_vars[index].format(i)] = to_model_vars[index].format(i)

    # Only Layer 0
    mapping_dict[
        "tf_t5model/decoder/block_._0/layer_._0/SelfAttention/relative_attention_bias/embeddings:0"
    ] = "tf_transformers/t5_decoder/transformer/layer_0/self_attention/relative_attention_bias/embeddings:0"
    mapping_dict[
        "tf_t5model/decoder/block_._0/layer_._1/EncDecAttention/relative_attention_bias/embeddings:0"
    ] = "tf_transformers/t5_decoder/transformer/layer_0/cross_attention/relative_attention_bias/embeddings:0"
    # Final Layer Norm weight
    mapping_dict["tf_t5model/decoder/final_layer_norm/weight:0"] = "tf_transformers/t5_decoder/last_layer_norm/weight:0"

    from_to_variable_dict = {var.name: var for var in model_hf.variables}
    # del model_hf
    logging.info("Deleteing huggingface model for saving memory")

    tf_transformers_model_index_dict = {}
    for index, var in enumerate(model_tf_transformers.variables):
        tf_transformers_model_index_dict[var.name] = index

    # legacy_ai <-- hub
    assigned_map = []
    assigned_map_values = []
    for original_var, legacy_var in mapping_dict.items():

        index = tf_transformers_model_index_dict[legacy_var]
        # If not in mapping_dict, then mostly it is from attention layer
        if "query/kernel:0" in legacy_var or "key/kernel:0" in legacy_var or "value/kernel:0" in legacy_var:
            # hub (2D) to tf_transformers (3D)
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
        if (
            original_var
            == "tf_t5model/decoder/block_._0/layer_._1/EncDecAttention/relative_attention_bias/embeddings:0"
        ):
            if original_var not in from_to_variable_dict:
                model_tf_transformers.variables[index].assign(tf.zeros_like(model_tf_transformers.variables[index]))
                assigned_map.append((original_var, legacy_var))
                continue

        model_tf_transformers.variables[index].assign(from_to_variable_dict.get(original_var))
        assigned_map.append((original_var, legacy_var))

    logging.info("Done assigning DECODER variables weights {}".format(len(assigned_map)))
