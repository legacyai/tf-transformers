import tensorflow as tf
from absl import logging

logging.set_verbosity("INFO")


def convert_albert_hf_to_tf_transformers(model_hf, model_tf_transformers, config):
    """
    Args:
        model_hf: HuggingFace Model (TF)
        model_tf_transformers: tf_transformers model/layer
        config: dict

    """
    # From vars (Transformer variables)
    from_model_vars = [
        "tf_albert_model/albert/embeddings/word_embeddings/weight:0",
        "tf_albert_model/albert/embeddings/token_type_embeddings/embeddings:0",
        "tf_albert_model/albert/embeddings/position_embeddings/embeddings:0",
        "tf_albert_model/albert/embeddings/LayerNorm/gamma:0",
        "tf_albert_model/albert/embeddings/LayerNorm/beta:0",
        "tf_albert_model/albert/encoder/embedding_hidden_mapping_in/kernel:0",
        "tf_albert_model/albert/encoder/embedding_hidden_mapping_in/bias:0",
        "tf_albert_model/albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/query/kernel:0",
        "tf_albert_model/albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/query/bias:0",
        "tf_albert_model/albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/key/kernel:0",
        "tf_albert_model/albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/key/bias:0",
        "tf_albert_model/albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/value/kernel:0",
        "tf_albert_model/albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/value/bias:0",
        "tf_albert_model/albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/dense/kernel:0",
        "tf_albert_model/albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/dense/bias:0",
        "tf_albert_model/albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/LayerNorm/gamma:0",
        "tf_albert_model/albert/encoder/albert_layer_groups_._0/albert_layers_._0/attention/LayerNorm/beta:0",
        "tf_albert_model/albert/encoder/albert_layer_groups_._0/albert_layers_._0/ffn/kernel:0",
        "tf_albert_model/albert/encoder/albert_layer_groups_._0/albert_layers_._0/ffn/bias:0",
        "tf_albert_model/albert/encoder/albert_layer_groups_._0/albert_layers_._0/ffn_output/kernel:0",
        "tf_albert_model/albert/encoder/albert_layer_groups_._0/albert_layers_._0/ffn_output/bias:0",
        "tf_albert_model/albert/encoder/albert_layer_groups_._0/albert_layers_._0/full_layer_layer_norm/gamma:0",
        "tf_albert_model/albert/encoder/albert_layer_groups_._0/albert_layers_._0/full_layer_layer_norm/beta:0",
        "tf_albert_model/albert/pooler/kernel:0",
        "tf_albert_model/albert/pooler/bias:0",
    ]

    # To vars (Transformer variables)
    to_model_vars = [
        "tf_transformers/albert/word_embeddings/embeddings:0",
        "tf_transformers/albert/type_embeddings/embeddings:0",
        "tf_transformers/albert/positional_embeddings/embeddings:0",
        "tf_transformers/albert/embeddings/layer_norm/gamma:0",
        "tf_transformers/albert/embeddings/layer_norm/beta:0",
        "tf_transformers/albert/embedding_projection/kernel:0",
        "tf_transformers/albert/embedding_projection/bias:0",
        "tf_transformers/albert/transformer/layer/self_attention/query/kernel:0",
        "tf_transformers/albert/transformer/layer/self_attention/query/bias:0",
        "tf_transformers/albert/transformer/layer/self_attention/key/kernel:0",
        "tf_transformers/albert/transformer/layer/self_attention/key/bias:0",
        "tf_transformers/albert/transformer/layer/self_attention/value/kernel:0",
        "tf_transformers/albert/transformer/layer/self_attention/value/bias:0",
        "tf_transformers/albert/transformer/layer/self_attention_output/kernel:0",
        "tf_transformers/albert/transformer/layer/self_attention_output/bias:0",
        "tf_transformers/albert/transformer/layer/self_attention_layer_norm/gamma:0",
        "tf_transformers/albert/transformer/layer/self_attention_layer_norm/beta:0",
        "tf_transformers/albert/transformer/layer/intermediate/kernel:0",
        "tf_transformers/albert/transformer/layer/intermediate/bias:0",
        "tf_transformers/albert/transformer/layer/output/kernel:0",
        "tf_transformers/albert/transformer/layer/output/bias:0",
        "tf_transformers/albert/transformer/layer/output_layer_norm/gamma:0",
        "tf_transformers/albert/transformer/layer/output_layer_norm/beta:0",
        "tf_transformers/albert/pooler_transform/kernel:0",
        "tf_transformers/albert/pooler_transform/bias:0",
    ]

    # Simple Assertion
    assert len(from_model_vars) == len(to_model_vars)
    mapping_dict = {}

    for index in range(len(from_model_vars)):
        for i in range(config["num_hidden_layers"]):
            mapping_dict[from_model_vars[index].format(i)] = to_model_vars[index].format(i)

    from_to_variable_dict = {var.name: var for var in model_hf.variables}
    logging.info("Deleteing huggingface model for saving memory")

    tf_transformers_model_index_dict = {}
    for index, var in enumerate(model_tf_transformers.variables):
        tf_transformers_model_index_dict[var.name] = index

    # legacy_ai <-- HuggingFace
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
            # assigned_map_values.append\
            # ((tf.reduce_sum(from_to_variable_dict.get(original_var)).numpy(), \
            # tf.reduce_sum(model.variables[index]).numpy()))
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
            # assigned_map_values.append((tf.reduce_sum(\
            # from_to_variable_dict.get(original_var)).numpy(),\
            #  tf.reduce_sum(model.variables[index]).numpy()))
            continue

        model_tf_transformers.variables[index].assign(from_to_variable_dict.get(original_var))
        assigned_map.append((original_var, legacy_var))
        # assigned_map_values.append((tf.reduce_sum(\
        # from_to_variable_dict.get(original_var)).numpy(),\
        #  tf.reduce_sum(model.variables[index]).numpy()))

    logging.info("Done assigning variables weights. Total {}".format(len(assigned_map)))
