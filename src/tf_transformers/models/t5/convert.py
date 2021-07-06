import numpy as np
import tensorflow as tf


def convert_t5_pt(model, config, model_name):
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

    from_model_vars = [
        "encoder.block.{}.layer.0.SelfAttention.q.weight",
        "encoder.block.{}.layer.0.SelfAttention.k.weight",
        "encoder.block.{}.layer.0.SelfAttention.v.weight",
        "encoder.block.{}.layer.0.SelfAttention.o.weight",
        "encoder.block.{}.layer.0.layer_norm.weight",
        "encoder.block.{}.layer.1.DenseReluDense.wi.weight",
        "encoder.block.{}.layer.1.DenseReluDense.wo.weight",
        "encoder.block.{}.layer.1.layer_norm.weight",
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
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
    ] = "tf_transformers/t5_encoder/transformer/layer_0/self_attention/relative_attention_bias/embeddings:0"
    # Word Embedding
    mapping_dict["shared.weight"] = "tf_transformers/t5_encoder/word_embeddings/embeddings:0"
    # Final Layer Norm weight
    mapping_dict["encoder.final_layer_norm.weight"] = "tf_transformers/t5_encoder/last_layer_norm/weight:0"

    # T5Model
    from transformers import T5Model as PTT5Model

    model_hf = PTT5Model.from_pretrained(model_name)
    # HF model variable name to variable values, for fast retrieval
    from_to_variable_dict = {name: var.detach().numpy() for name, var in model_hf.named_parameters()}

    tf_transformers_model_index_dict = {}
    for index, var in enumerate(model.variables):
        tf_transformers_model_index_dict[var.name] = index

    # legacy_ai <-- hub
    assigned_map = []
    # assigned_map_values = []
    for original_var, legacy_var in mapping_dict.items():
        index = tf_transformers_model_index_dict[legacy_var]
        # If not in mapping_dict, then mostly it is from attention layer
        if "query/kernel:0" in legacy_var or "key/kernel:0" in legacy_var or "value/kernel:0" in legacy_var:
            # hub (2D) to tf_transformers (3D)
            model.variables[index].assign(
                np.reshape(
                    np.transpose(from_to_variable_dict.get(original_var)),
                    (
                        config["embedding_size"],
                        config["num_attention_heads"],
                        config["attention_head_size"],
                    ),
                )
            )
            assigned_map.append((original_var, legacy_var))
            continue

        elif "kernel:0" in legacy_var:
            if list(model.variables[index].shape) == list(from_to_variable_dict.get(original_var).shape):
                model.variables[index].assign(np.transpose(from_to_variable_dict.get(original_var)))
                assigned_map.append((original_var, legacy_var))
                continue
            else:
                model.variables[index].assign(np.transpose(from_to_variable_dict.get(original_var)))
                assigned_map.append((original_var, legacy_var))
                continue
        model.variables[index].assign(from_to_variable_dict.get(original_var))
        assigned_map.append((original_var, legacy_var))

    # Decoder Side
    # From vars (Transformer variables)

    from_model_vars = [
        "decoder.block.{}.layer.0.SelfAttention.q.weight",
        "decoder.block.{}.layer.0.SelfAttention.k.weight",
        "decoder.block.{}.layer.0.SelfAttention.v.weight",
        "decoder.block.{}.layer.0.SelfAttention.o.weight",
        "decoder.block.{}.layer.0.layer_norm.weight",
        "decoder.block.{}.layer.1.EncDecAttention.q.weight",
        "decoder.block.{}.layer.1.EncDecAttention.k.weight",
        "decoder.block.{}.layer.1.EncDecAttention.v.weight",
        "decoder.block.{}.layer.1.EncDecAttention.o.weight",
        "decoder.block.{}.layer.1.layer_norm.weight",
        "decoder.block.{}.layer.2.DenseReluDense.wi.weight",
        "decoder.block.{}.layer.2.DenseReluDense.wo.weight",
        "decoder.block.{}.layer.2.layer_norm.weight",
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
        "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
    ] = "tf_transformers/t5_decoder/transformer/layer_0/self_attention/relative_attention_bias/embeddings:0"
    # Final Layer Norm weight
    mapping_dict["decoder.final_layer_norm.weight"] = "tf_transformers/t5_decoder/last_layer_norm/weight:0"

    # HF model variable name to variable values, for fast retrieval
    from_to_variable_dict = {name: var.detach().numpy() for name, var in model_hf.named_parameters()}

    tf_transformers_model_index_dict = {}
    for index, var in enumerate(model.variables):
        tf_transformers_model_index_dict[var.name] = index
        if (
            var.name
            == "tf_transformers/t5_decoder/transformer/layer_0/cross_attention/relative_attention_bias/embeddings:0"
        ):
            model.variables[index].assign(tf.zeros_like(model.variables[index]))
            continue

    # legacy_ai <-- hub
    assigned_map = []
    # assigned_map_values = []
    for original_var, legacy_var in mapping_dict.items():
        index = tf_transformers_model_index_dict[legacy_var]
        # If not in mapping_dict, then mostly it is from attention layer
        if "query/kernel:0" in legacy_var or "key/kernel:0" in legacy_var or "value/kernel:0" in legacy_var:
            # hub (2D) to tf_transformers (3D)
            model.variables[index].assign(
                np.reshape(
                    np.transpose(from_to_variable_dict.get(original_var)),
                    (
                        config["embedding_size"],
                        config["num_attention_heads"],
                        config["attention_head_size"],
                    ),
                )
            )
            assigned_map.append((original_var, legacy_var))
            continue
        elif "kernel:0" in legacy_var:
            if list(model.variables[index].shape) == list(from_to_variable_dict.get(original_var).shape):
                model.variables[index].assign(np.transpose(from_to_variable_dict.get(original_var)))
                assigned_map.append((original_var, legacy_var))
                continue
            else:
                model.variables[index].assign(np.transpose(from_to_variable_dict.get(original_var)))
                assigned_map.append((original_var, legacy_var))
                continue

        model.variables[index].assign(from_to_variable_dict.get(original_var))
        assigned_map.append((original_var, legacy_var))

    from transformers import T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    text = "This is a long sentence to check how close models are."
    inputs = tokenizer(text, return_tensors="pt")
    outputs_hf = model_hf(inputs["input_ids"], decoder_input_ids=inputs["input_ids"])
    outputs_hf = torch.sum(outputs_hf["last_hidden_state"], dim=-1).detach().numpy()

    inputs = tokenizer(text, return_tensors="tf")
    inputs_tf = {}
    inputs_tf["encoder_input_ids"] = inputs["input_ids"]
    inputs_tf["encoder_input_mask"] = inputs["attention_mask"]
    inputs_tf["decoder_input_ids"] = inputs["input_ids"]
    outputs_tf = model(inputs_tf)
    outputs_tf = tf.reduce_sum(outputs_tf["token_embeddings"], axis=-1).numpy()
    tf.debugging.assert_near(outputs_hf, outputs_tf, rtol=1.0)


def convert_t5_tf(model, config, model_name):
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

    # T5Model
    from transformers import TFT5Model

    model_hf = TFT5Model.from_pretrained(model_name)

    from_to_variable_dict = {var.name: var for var in model_hf.variables}

    tf_transformers_model_index_dict = {}
    for index, var in enumerate(model.variables):
        tf_transformers_model_index_dict[var.name] = index

    # legacy_ai <-- hub
    assigned_map = []
    # assigned_map_values = []
    for original_var, legacy_var in mapping_dict.items():

        index = tf_transformers_model_index_dict[legacy_var]
        # If not in mapping_dict, then mostly it is from attention layer
        if "query/kernel:0" in legacy_var or "key/kernel:0" in legacy_var or "value/kernel:0" in legacy_var:
            # hub (2D) to tf_transformers (3D)
            model.variables[index].assign(
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

        model.variables[index].assign(from_to_variable_dict.get(original_var))
        assigned_map.append((original_var, legacy_var))

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

    tf_transformers_model_index_dict = {}
    for index, var in enumerate(model.variables):
        tf_transformers_model_index_dict[var.name] = index

    # legacy_ai <-- hub
    assigned_map = []
    # assigned_map_values = []
    for original_var, legacy_var in mapping_dict.items():

        index = tf_transformers_model_index_dict[legacy_var]
        # If not in mapping_dict, then mostly it is from attention layer
        if "query/kernel:0" in legacy_var or "key/kernel:0" in legacy_var or "value/kernel:0" in legacy_var:
            # hub (2D) to tf_transformers (3D)
            model.variables[index].assign(
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
                model.variables[index].assign(tf.zeros_like(model.variables[index]))
                assigned_map.append((original_var, legacy_var))
                continue

        model.variables[index].assign(from_to_variable_dict.get(original_var))
        assigned_map.append((original_var, legacy_var))

    from transformers import T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    text = "This is a long sentence to check how close models are."
    inputs = tokenizer(text, return_tensors="tf")
    outputs_hf = model_hf(inputs["input_ids"], decoder_input_ids=inputs["input_ids"])
    outputs_hf = tf.reduce_sum(outputs_hf["last_hidden_state"], axis=-1).numpy()

    inputs_tf = {}
    inputs_tf["encoder_input_ids"] = inputs["input_ids"]
    inputs_tf["encoder_input_mask"] = inputs["attention_mask"]
    inputs_tf["decoder_input_ids"] = inputs["input_ids"]
    outputs_tf = model(inputs_tf)
    outputs_tf = tf.reduce_sum(outputs_tf["token_embeddings"], axis=-1).numpy()
    tf.debugging.assert_near(outputs_hf, outputs_tf, rtol=1.0)
