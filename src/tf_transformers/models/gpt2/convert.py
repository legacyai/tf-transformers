import numpy as np
import tensorflow as tf


def convert_gpt2_pt(model, config, model_name):
    """PT converter
    Args:
        model_hf: HuggingFace Model (TF)
        model: tf_transformers model/layer
        config: dict
    Returns:
        a function
    """

    import torch
    import transformers

    transformers.logging.set_verbosity_error()

    # From vars (Transformer variables)
    from_model_vars = [
        "h.{}.ln_1.weight",
        "h.{}.ln_1.bias",
        "h.{}.attn.c_attn.weight",
        "h.{}.attn.c_attn.bias",
        "h.{}.attn.c_proj.weight",
        "h.{}.attn.c_proj.bias",
        "h.{}.ln_2.weight",
        "h.{}.ln_2.bias",
        "h.{}.mlp.c_fc.weight",
        "h.{}.mlp.c_fc.bias",
        "h.{}.mlp.c_proj.weight",
        "h.{}.mlp.c_proj.bias",
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
    mapping_dict["wte.weight"] = "tf_transformers/gpt2/word_embeddings/embeddings:0"
    # Positional Embedding
    mapping_dict["wpe.weight"] = "tf_transformers/gpt2/positional_embeddings/embeddings:0"
    mapping_dict["ln_f.weight"] = "tf_transformers/gpt2/ln_f/layer_norm/gamma:0"
    mapping_dict["ln_f.bias"] = "tf_transformers/gpt2/ln_f/layer_norm/beta:0"

    # BertModel
    from transformers import GPT2Model

    model_hf = GPT2Model.from_pretrained(model_name)

    # HF model variable name to variable values, for fast retrieval
    from_to_variable_dict = {name: var.detach().numpy() for name, var in model_hf.named_parameters()}

    # We need variable name to the index where it is stored inside tf_transformers model
    tf_transformers_model_index_dict = {}
    for index, var in enumerate(model.variables):
        tf_transformers_model_index_dict[var.name] = index
        # In auto_regressive mode, positional embeddings variable name has
        # cond extra name. So, in case someone converts in that mode,
        # replace above mapping here, only for positional embeddings
        if var.name == "tf_transformers/gpt2/cond/positional_embeddings/embeddings:0":
            mapping_dict["wpe.weight"] = "tf_transformers/gpt2/cond/positional_embeddings/embeddings:0"

    # Start assigning HF values to tf_transformers
    # assigned_map and assigned_map_values are used for sanity check if needed
    assigned_map = []
    # assigned_map_values = []
    for original_var, legacy_var in mapping_dict.items():
        index = tf_transformers_model_index_dict[legacy_var]
        from_shape = from_to_variable_dict.get(original_var).shape
        to_shape = model.variables[index].shape

        if len(from_shape) == 2:
            if len(to_shape) == 1:
                model.variables[index].assign(np.squeeze(from_to_variable_dict.get(original_var)))
                continue

        model.variables[index].assign(from_to_variable_dict.get(original_var))
        assigned_map.append((original_var, legacy_var))

    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    text = "This is a long sentence to check how close models are."
    inputs = tokenizer(text, return_tensors="pt")
    outputs_hf = model_hf(**inputs)
    outputs_hf = torch.sum(outputs_hf["last_hidden_state"], dim=-1).detach().numpy()
    inputs_tf = {}
    inputs_tf["input_ids"] = tf.cast(tf.constant(inputs["input_ids"].numpy()), tf.int32)
    outputs_tf = model(inputs_tf)
    outputs_tf = tf.reduce_sum(outputs_tf["token_embeddings"], axis=-1).numpy()
    tf.debugging.assert_near(outputs_hf, outputs_tf, rtol=1.0)


def convert_gpt2_tf(model, config, model_name):
    """TF converter
    Args:
        model_hf: HuggingFace Model (TF)
        model: tf_transformers model/layer
        config: dict
    Returns:
        a function
    """

    import transformers

    transformers.logging.set_verbosity_error()

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

    # GPT2Model
    from transformers import TFGPT2Model

    tf.keras.backend.clear_session()
    model_hf = TFGPT2Model.from_pretrained(model_name)

    # HF model variable name to variable values, for fast retrieval
    from_to_variable_dict = {var.name: var for var in model_hf.variables}

    # We need variable name to the index where it is stored inside tf_transformers model
    tf_transformers_model_index_dict = {}
    for index, var in enumerate(model.variables):
        tf_transformers_model_index_dict[var.name] = index
        # In auto_regressive mode, positional embeddings variable name has
        # cond extra name. So, in case someone converts in that mode,
        # replace above mapping here, only for positional embeddings
        if var.name == "tf_transformers/gpt2/cond/positional_embeddings/embeddings:0":
            mapping_dict[
                "tfgp_t2model/transformer/wpe/embeddings:0"
            ] = "tf_transformers/gpt2/cond/positional_embeddings/embeddings:0"

    # Start assigning HF values to tf_transformers
    # assigned_map and assigned_map_values are used for sanity check if needed
    assigned_map = []
    # assigned_map_values = []
    for original_var, legacy_var in mapping_dict.items():
        index = tf_transformers_model_index_dict[legacy_var]
        from_shape = from_to_variable_dict.get(original_var).shape
        to_shape = model.variables[index].shape

        if len(from_shape) == 2:
            if len(to_shape) == 1:
                model.variables[index].assign(tf.squeeze(from_to_variable_dict.get(original_var)))
                continue

        model.variables[index].assign(from_to_variable_dict.get(original_var))
        assigned_map.append((original_var, legacy_var))

    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    text = "This is a long sentence to check how close models are."
    inputs = tokenizer(text, return_tensors="tf")
    outputs_hf = model_hf(**inputs)
    outputs_hf = tf.reduce_sum(outputs_hf["last_hidden_state"], axis=-1).numpy()
    del model_hf

    inputs_tf = {}
    inputs_tf["input_ids"] = inputs["input_ids"]
    outputs_tf = model(inputs_tf)
    outputs_tf = tf.reduce_sum(outputs_tf["token_embeddings"], axis=-1).numpy()
    tf.debugging.assert_near(outputs_hf, outputs_tf, rtol=1.0)
