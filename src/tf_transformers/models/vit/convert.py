import tensorflow as tf


def convert_vit_pt(model, config, model_name):
    """PT converter
    Args:
        model_hf: HuggingFace Model (TF)
        model: tf_transformers model/layer
        config: dict
    Returns:
        a function
    """
    import numpy as np
    import torch

    # From vars (Transformer variables)
    from_model_vars = [
        "vit.encoder.layer.{}.attention.attention.query.weight",
        "vit.encoder.layer.{}.attention.attention.query.bias",
        "vit.encoder.layer.{}.attention.attention.key.weight",
        "vit.encoder.layer.{}.attention.attention.key.bias",
        "vit.encoder.layer.{}.attention.attention.value.weight",
        "vit.encoder.layer.{}.attention.attention.value.bias",
        "vit.encoder.layer.{}.attention.output.dense.weight",
        "vit.encoder.layer.{}.attention.output.dense.bias",
        "vit.encoder.layer.{}.layernorm_before.weight",
        "vit.encoder.layer.{}.layernorm_before.bias",
        "vit.encoder.layer.{}.intermediate.dense.weight",
        "vit.encoder.layer.{}.intermediate.dense.bias",
        "vit.encoder.layer.{}.output.dense.weight",
        "vit.encoder.layer.{}.output.dense.bias",
        "vit.encoder.layer.{}.layernorm_after.weight",
        "vit.encoder.layer.{}.layernorm_after.bias",
    ]

    # To vars (Transformer variables)
    to_model_vars = [
        "tf_transformers/vit/transformer/layer_{}/self_attention/query/kernel:0",
        "tf_transformers/vit/transformer/layer_{}/self_attention/query/bias:0",
        "tf_transformers/vit/transformer/layer_{}/self_attention/key/kernel:0",
        "tf_transformers/vit/transformer/layer_{}/self_attention/key/bias:0",
        "tf_transformers/vit/transformer/layer_{}/self_attention/value/kernel:0",
        "tf_transformers/vit/transformer/layer_{}/self_attention/value/bias:0",
        "tf_transformers/vit/transformer/layer_{}/self_attention_output/kernel:0",
        "tf_transformers/vit/transformer/layer_{}/self_attention_output/bias:0",
        "tf_transformers/vit/transformer/layer_{}/pre_attention_norm/gamma:0",
        "tf_transformers/vit/transformer/layer_{}/pre_attention_norm/beta:0",
        "tf_transformers/vit/transformer/layer_{}/intermediate/kernel:0",
        "tf_transformers/vit/transformer/layer_{}/intermediate/bias:0",
        "tf_transformers/vit/transformer/layer_{}/output/kernel:0",
        "tf_transformers/vit/transformer/layer_{}/output/bias:0",
        "tf_transformers/vit/transformer/layer_{}/self_attention_layer_norm/gamma:0",
        "tf_transformers/vit/transformer/layer_{}/self_attention_layer_norm/beta:0",
    ]

    # Simple Assertion
    assert len(from_model_vars) == len(to_model_vars)

    # This dictionary maps from -> to dict names
    mapping_dict = {}
    for index in range(len(from_model_vars)):
        for i in range(config["num_hidden_layers"]):
            mapping_dict[from_model_vars[index].format(i)] = to_model_vars[index].format(i)

    # CLS Token
    mapping_dict["vit.embeddings.cls_token"] = "tf_transformers/vit/cls_token:0"
    # Positional Embedding
    mapping_dict["vit.embeddings.position_embeddings"] = "tf_transformers/vit/positional_embeddings/embeddings:0"
    # Patch Embeddings
    mapping_dict[
        "vit.embeddings.patch_embeddings.projection.weight"
    ] = "tf_transformers/vit/patch_embeddings/conv2d/kernel:0"
    mapping_dict[
        "vit.embeddings.patch_embeddings.projection.bias"
    ] = "tf_transformers/vit/patch_embeddings/conv2d/bias:0"

    mapping_dict["vit.layernorm.weight"] = "tf_transformers/vit/last_layer_norm/gamma:0"
    mapping_dict["vit.layernorm.bias"] = "tf_transformers/vit/last_layer_norm/beta:0"
    mapping_dict["classifier.weight"] = "tf_transformers/vit/classifier_layer/kernel:0"
    mapping_dict["classifier.bias"] = "tf_transformers/vit/classifier_layer/bias:0"

    # Randomly initialize by HF
    # mapping_dict["pooler.dense.weight"] = "tf_transformers/vit/pooler_transform/kernel:0"
    # mapping_dict["pooler.dense.bias"] = "tf_transformers/vit/pooler_transform/bias:0"
    from transformers import ViTForImageClassification

    model_hf = ViTForImageClassification.from_pretrained(model_name)
    # HF model variable name to variable values, for fast retrieval
    from_to_variable_dict = {name: var.detach().numpy() for name, var in model_hf.named_parameters()}
    # We need variable name to the index where it is stored inside tf_transformers model
    tf_transformers_model_index_dict = {}
    for index, var in enumerate(model.variables):
        tf_transformers_model_index_dict[var.name] = index

        # In auto_regressive mode, positional embeddings variable name has
        # cond extra name. So, in case someone converts in that mode,
        # replace above mapping here, only for positional embeddings
    #         if var.name == "tf_transformers/bert/cond/positional_embeddings/embeddings:0":
    #             mapping_dict[
    #                 "embeddings.position_embeddings.weight"
    #             ] = "tf_transformers/bert/cond/positional_embeddings/embeddings:0"

    # Start assigning HF values to tf_transformers
    # assigned_map and assigned_map_values are used for sanity check if needed
    assigned_map = []
    # assigned_map_values = []
    for original_var, legacy_var in mapping_dict.items():
        index = tf_transformers_model_index_dict[legacy_var]
        # Assign Patch embeddings
        if "patch_embeddings/conv2d/kernel:0" in legacy_var:
            model.variables[index].assign(np.transpose(from_to_variable_dict.get(original_var), [2, 3, 1, 0]))
            assigned_map.append((original_var, legacy_var))
            continue

        if "query/kernel:0" in legacy_var or "key/kernel:0" in legacy_var or "value/kernel:0" in legacy_var:

            # huggingface (2D) to tf_transformers (3D)
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
        if "query/bias:0" in legacy_var or "key/bias:0" in legacy_var or "value/bias:0" in legacy_var:

            # huggingface (2D) to tf_transformers (3D)
            model.variables[index].assign(
                np.reshape(
                    from_to_variable_dict.get(original_var),
                    (
                        config["num_attention_heads"],
                        config["attention_head_size"],
                    ),
                )
            )
            assigned_map.append((original_var, legacy_var))
            continue

        if "self_attention_output/kernel:0" in legacy_var:
            # huggingface (3D) to tf_transformers (2D)
            model.variables[index].assign(
                np.reshape(
                    np.transpose(from_to_variable_dict.get(original_var)),
                    (config["embedding_size"], config["num_attention_heads"] * config["attention_head_size"]),
                )
            )
            assigned_map.append((original_var, legacy_var))
            continue

        if "self_attention_output/bias:0" in legacy_var:
            # huggingface (3D) to tf_transformers (2D)
            model.variables[index].assign(
                np.reshape(
                    from_to_variable_dict.get(original_var),
                    (-1),
                )
            )
            assigned_map.append((original_var, legacy_var))
            continue

        if "intermediate/kernel:0" in legacy_var or "output/kernel:0" in legacy_var:
            # huggingface (torch transpose
            model.variables[index].assign(np.transpose(from_to_variable_dict.get(original_var)))

            assigned_map.append((original_var, legacy_var))
            continue

        if "classifier_layer/kernel:0" in legacy_var:
            model.variables[index].assign(np.transpose(from_to_variable_dict.get(original_var)))
            continue

        model.variables[index].assign(from_to_variable_dict.get(original_var))
        assigned_map.append((original_var, legacy_var))

    from transformers import ViTModel

    model_hf = ViTModel.from_pretrained(model_name)

    batch_size = 2
    num_channels = config["num_channels"]
    image_size = config["image_size"]
    pixel_values = torch.rand(([batch_size, num_channels, image_size, image_size]))
    inputs_tf = tf.convert_to_tensor(pixel_values.numpy())
    # Reshape (b x channels x h x w)
    inputs_tf = tf.transpose(inputs_tf, [0, 2, 3, 1])

    outputs = model_hf(pixel_values)['last_hidden_state'].detach().numpy()
    outputs_tf = model({"input_ids": inputs_tf})["token_embeddings"].numpy()
    # Slightly bigger rtol
    tf.debugging.assert_near(tf.reduce_sum(outputs), tf.reduce_sum(outputs_tf), rtol=5.0)
