import tensorflow as tf
from absl import logging

logging.set_verbosity("INFO")


def convert_gpt2_hf_to_tf_transformers(model_hf, model_tf_transformers, config):
    """
    Args:
        model_hf: HuggingFace Model (TF)
        model_tf_transformers: tf_transformers model/layer
        config: dict

    """
    # From vars (Transformer variables)
    from_model_vars = [
        "tfgp_t2model/transformer/h_._{}/ln_1/gamma:0",
        "tfgp_t2model/transformer/h_._{}/ln_1/beta:0",
        "tfgp_t2model/transformer/h_._{}/attn/c_attn/weight:0",
        "tfgp_t2model/transformer/h_._{}/attn/c_attn/bias:0",
        "tfgp_t2model/transformer/h_._{}/attn/c_proj/weight:0",
        "tfgp_t2model/transformer/h_._{}/attn/c_proj/bias:0",
        "tfgp_t2model/transformer/h_._{}/ln_2/gamma:0",
        "tfgp_t2model/transformer/h_._{}/ln_2/beta:0",
        "tfgp_t2model/transformer/h_._{}/mlp/c_fc/weight:0",
        "tfgp_t2model/transformer/h_._{}/mlp/c_fc/bias:0",
        "tfgp_t2model/transformer/h_._{}/mlp/c_proj/weight:0",
        "tfgp_t2model/transformer/h_._{}/mlp/c_proj/bias:0",
    ]

    # To vars (Transformer variables)
    to_model_vars = [
        "tf_transformers/gpt2/transformer/layer_{}/ln_1/layer_norm/gamma:0",
        "tf_transformers/gpt2/transformer/layer_{}/ln_1/layer_norm/beta:0",
        "tf_transformers/gpt2/transformer/layer_{}/self_attention/qkv/kernel:0",
        "tf_transformers/gpt2/transformer/layer_{}/self_attention/qkv/bias:0",
        "tf_transformers/gpt2/transformer/layer_{}/self_attention_output/kernel:0",
        "tf_transformers/gpt2/transformer/layer_{}/self_attention_output/bias:0",
        "tf_transformers/gpt2/transformer/layer_{}/self_attention_layer_norm/gamma:0",
        "tf_transformers/gpt2/transformer/layer_{}/self_attention_layer_norm/beta:0",
        "tf_transformers/gpt2/transformer/layer_{}/intermediate/kernel:0",
        "tf_transformers/gpt2/transformer/layer_{}/intermediate/bias:0",
        "tf_transformers/gpt2/transformer/layer_{}/output/kernel:0",
        "tf_transformers/gpt2/transformer/layer_{}/output/bias:0",
    ]

    # Simple Assertion
    assert len(from_model_vars) == len(to_model_vars)
    mapping_dict = {}

    for index in range(len(from_model_vars)):
        for i in range(config["num_hidden_layers"]):
            mapping_dict[from_model_vars[index].format(i)] = to_model_vars[index].format(i)

    # Word Embeddings
    mapping_dict["tfgp_t2model/transformer/wte/weight:0"] = "tf_transformers/gpt2/word_embeddings/embeddings:0"
    # Positional Embedding
    mapping_dict[
        "tfgp_t2model/transformer/wpe/embeddings:0"
    ] = "tf_transformers/gpt2/positional_embeddings/embeddings:0"
    mapping_dict["tfgp_t2model/transformer/ln_f/gamma:0"] = "tf_transformers/gpt2/ln_f/layer_norm/gamma:0"
    mapping_dict["tfgp_t2model/transformer/ln_f/beta:0"] = "tf_transformers/gpt2/ln_f/layer_norm/beta:0"

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

        from_shape = from_to_variable_dict.get(original_var).shape
        to_shape = model_tf_transformers.variables[index].shape

        if len(from_shape) == 2:
            if len(to_shape) == 1:
                model_tf_transformers.variables[index].assign(tf.squeeze(from_to_variable_dict.get(original_var)))
                continue

        model_tf_transformers.variables[index].assign(from_to_variable_dict.get(original_var))
        assigned_map.append((original_var, legacy_var))

    logging.info("Done assigning variables weights . Total {}".format(len(assigned_map)))
