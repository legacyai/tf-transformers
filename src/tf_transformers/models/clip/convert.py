import tensorflow as tf


def convert_text_model_pt(model, config, model_hf):
    import numpy as np

    mapping_dict = {}

    # From vars (Transformer variables)
    from_model_vars = [
        'text_model.encoder.layers.{}.layer_norm1.weight',
        'text_model.encoder.layers.{}.layer_norm1.bias',
        'text_model.encoder.layers.{}.self_attn.k_proj.weight',
        'text_model.encoder.layers.{}.self_attn.k_proj.bias',
        'text_model.encoder.layers.{}.self_attn.v_proj.weight',
        'text_model.encoder.layers.{}.self_attn.v_proj.bias',
        'text_model.encoder.layers.{}.self_attn.q_proj.weight',
        'text_model.encoder.layers.{}.self_attn.q_proj.bias',
        'text_model.encoder.layers.{}.self_attn.out_proj.weight',
        'text_model.encoder.layers.{}.self_attn.out_proj.bias',
        'text_model.encoder.layers.{}.mlp.fc1.weight',
        'text_model.encoder.layers.{}.mlp.fc1.bias',
        'text_model.encoder.layers.{}.mlp.fc2.weight',
        'text_model.encoder.layers.{}.mlp.fc2.bias',
        'text_model.encoder.layers.{}.layer_norm2.weight',
        'text_model.encoder.layers.{}.layer_norm2.bias',
    ]

    # To vars (Transformer variables)
    to_model_vars = [
        'tf_transformers/clip_text/transformer/layer_{}/pre_attention_norm/gamma:0',
        'tf_transformers/clip_text/transformer/layer_{}/pre_attention_norm/beta:0',
        'tf_transformers/clip_text/transformer/layer_{}/self_attention/key/kernel:0',
        'tf_transformers/clip_text/transformer/layer_{}/self_attention/key/bias:0',
        'tf_transformers/clip_text/transformer/layer_{}/self_attention/value/kernel:0',
        'tf_transformers/clip_text/transformer/layer_{}/self_attention/value/bias:0',
        'tf_transformers/clip_text/transformer/layer_{}/self_attention/query/kernel:0',
        'tf_transformers/clip_text/transformer/layer_{}/self_attention/query/bias:0',
        'tf_transformers/clip_text/transformer/layer_{}/self_attention_output/kernel:0',
        'tf_transformers/clip_text/transformer/layer_{}/self_attention_output/bias:0',
        'tf_transformers/clip_text/transformer/layer_{}/intermediate/kernel:0',
        'tf_transformers/clip_text/transformer/layer_{}/intermediate/bias:0',
        'tf_transformers/clip_text/transformer/layer_{}/output/kernel:0',
        'tf_transformers/clip_text/transformer/layer_{}/output/bias:0',
        'tf_transformers/clip_text/transformer/layer_{}/self_attention_layer_norm/gamma:0',
        'tf_transformers/clip_text/transformer/layer_{}/self_attention_layer_norm/beta:0',
    ]

    # Simple Assertion
    assert len(from_model_vars) == len(to_model_vars)

    # This dictionary maps from -> to dict names
    mapping_dict = {}
    for index in range(len(from_model_vars)):
        for i in range(config["num_hidden_layers"]):
            mapping_dict[from_model_vars[index].format(i)] = to_model_vars[index].format(i)

    # Word Embedding 49408 x 512 --> 49408 x 512
    mapping_dict[
        "text_model.embeddings.token_embedding.weight"
    ] = "tf_transformers/clip_text/word_embeddings/embeddings:0"

    # Positional Embedding 77 x 512 --> 77 x 512
    mapping_dict[
        "text_model.embeddings.position_embedding.weight"
    ] = "tf_transformers/clip_text/positional_embeddings/embeddings:0"

    # Last norm
    mapping_dict['text_model.final_layer_norm.weight'] = 'tf_transformers/clip_text/last_layer_norm/gamma:0'
    mapping_dict['text_model.final_layer_norm.bias'] = 'tf_transformers/clip_text/last_layer_norm/beta:0'

    # Last layer Norm
    # mapping_dict["vison_model.layernorm.weight"] = "tf_transformers/vit/last_layer_norm/gamma:0"
    # mapping_dict["text_model.layernorm.bias"] = "tf_transformers/vit/last_layer_norm/beta:0"

    from_to_variable_dict = {name: var.detach().numpy() for name, var in model_hf.named_parameters()}

    tf_transformers_model_index_dict = {}
    for index, var in enumerate(model.variables):
        tf_transformers_model_index_dict[var.name] = index

    assigned_map = []
    # assigned_map_values = []
    for original_var, legacy_var in mapping_dict.items():
        index = tf_transformers_model_index_dict[legacy_var]

        if "query/kernel:0" in legacy_var or "key/kernel:0" in legacy_var or "value/kernel:0" in legacy_var:

            # huggingface (2D) to tf_transformers (3D)
            model.variables[index].assign(
                tf.reshape(
                    tf.transpose(from_to_variable_dict.get(original_var)),
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

        if 'intermediate/kernel:0' in legacy_var or 'output/kernel:0' in legacy_var:
            # huggingface (2D) to tf_transformers (3D)
            model.variables[index].assign(
                tf.transpose(from_to_variable_dict.get(original_var)),
            )
            assigned_map.append((original_var, legacy_var))
            continue

        model.variables[index].assign(from_to_variable_dict.get(original_var))
        assigned_map.append((original_var, legacy_var))


def convert_image_model_pt(model, config, model_hf):
    import numpy as np

    mapping_dict = {}

    # From vars (Transformer variables)
    from_model_vars = [
        'vision_model.encoder.layers.{}.layer_norm1.weight',
        'vision_model.encoder.layers.{}.layer_norm1.bias',
        'vision_model.encoder.layers.{}.self_attn.k_proj.weight',
        'vision_model.encoder.layers.{}.self_attn.k_proj.bias',
        'vision_model.encoder.layers.{}.self_attn.v_proj.weight',
        'vision_model.encoder.layers.{}.self_attn.v_proj.bias',
        'vision_model.encoder.layers.{}.self_attn.q_proj.weight',
        'vision_model.encoder.layers.{}.self_attn.q_proj.bias',
        'vision_model.encoder.layers.{}.self_attn.out_proj.weight',
        'vision_model.encoder.layers.{}.self_attn.out_proj.bias',
        'vision_model.encoder.layers.{}.mlp.fc1.weight',
        'vision_model.encoder.layers.{}.mlp.fc1.bias',
        'vision_model.encoder.layers.{}.mlp.fc2.weight',
        'vision_model.encoder.layers.{}.mlp.fc2.bias',
        'vision_model.encoder.layers.{}.layer_norm2.weight',
        'vision_model.encoder.layers.{}.layer_norm2.bias',
    ]

    # To vars (Transformer variables)
    to_model_vars = [
        'tf_transformers/clip_image/transformer/layer_{}/pre_attention_norm/gamma:0',
        'tf_transformers/clip_image/transformer/layer_{}/pre_attention_norm/beta:0',
        'tf_transformers/clip_image/transformer/layer_{}/self_attention/key/kernel:0',
        'tf_transformers/clip_image/transformer/layer_{}/self_attention/key/bias:0',
        'tf_transformers/clip_image/transformer/layer_{}/self_attention/value/kernel:0',
        'tf_transformers/clip_image/transformer/layer_{}/self_attention/value/bias:0',
        'tf_transformers/clip_image/transformer/layer_{}/self_attention/query/kernel:0',
        'tf_transformers/clip_image/transformer/layer_{}/self_attention/query/bias:0',
        'tf_transformers/clip_image/transformer/layer_{}/self_attention_output/kernel:0',
        'tf_transformers/clip_image/transformer/layer_{}/self_attention_output/bias:0',
        'tf_transformers/clip_image/transformer/layer_{}/intermediate/kernel:0',
        'tf_transformers/clip_image/transformer/layer_{}/intermediate/bias:0',
        'tf_transformers/clip_image/transformer/layer_{}/output/kernel:0',
        'tf_transformers/clip_image/transformer/layer_{}/output/bias:0',
        'tf_transformers/clip_image/transformer/layer_{}/self_attention_layer_norm/gamma:0',
        'tf_transformers/clip_image/transformer/layer_{}/self_attention_layer_norm/beta:0',
    ]

    # Simple Assertion
    assert len(from_model_vars) == len(to_model_vars)

    # This dictionary maps from -> to dict names
    mapping_dict = {}
    for index in range(len(from_model_vars)):
        for i in range(config["num_hidden_layers"]):
            mapping_dict[from_model_vars[index].format(i)] = to_model_vars[index].format(i)

    # CLS Token 768 --> 1 x 1 x 768
    mapping_dict["vision_model.embeddings.class_embedding"] = "tf_transformers/clip_image/cls_token:0"
    # Positional Embedding 50 x 768 --> 1 x 50 x 768
    mapping_dict[
        "vision_model.embeddings.position_embedding.weight"
    ] = "tf_transformers/clip_image/positional_embeddings/embeddings:0"
    # Patch Embeddings [768, 3, 32, 32] --> [32, 32, 3, 768]
    mapping_dict[
        "vision_model.embeddings.patch_embedding.weight"
    ] = "tf_transformers/clip_image/patch_embeddings/conv2d/kernel:0"

    # Embedding Norm
    mapping_dict['vision_model.pre_layrnorm.weight'] = 'tf_transformers/clip_image/embeddings/layer_norm/gamma:0'
    mapping_dict['vision_model.pre_layrnorm.bias'] = 'tf_transformers/clip_image/embeddings/layer_norm/beta:0'

    # Last norm
    mapping_dict['vision_model.post_layernorm.weight'] = 'tf_transformers/clip_image/last_layer_norm/gamma:0'
    mapping_dict['vision_model.post_layernorm.bias'] = 'tf_transformers/clip_image/last_layer_norm/beta:0'

    # logits scale
    mapping_dict['logit_scale'] = 'logits_scale:0'

    # Projections (Image and Text)
    mapping_dict['visual_projection.weight'] = 'tf_transformers/clip_image/visual_projection/kernel:0'
    mapping_dict['text_projection.weight'] = 'tf_transformers/clip_text/text_projection/kernel:0'

    from_to_variable_dict = {name: var.detach().numpy() for name, var in model_hf.named_parameters()}

    tf_transformers_model_index_dict = {}
    for index, var in enumerate(model.variables):
        tf_transformers_model_index_dict[var.name] = index

    assigned_map = []
    # assigned_map_values = []
    for original_var, legacy_var in mapping_dict.items():
        index = tf_transformers_model_index_dict[legacy_var]

        # Assign CLS token
        if "tf_transformers/clip_image/cls_token" in legacy_var:
            model.variables[index].assign(tf.expand_dims([from_to_variable_dict.get(original_var)], 0))
            assigned_map.append((original_var, legacy_var))
            continue

        # Assing positional embedding
        if "tf_transformers/clip_image/positional_embeddings" in legacy_var:
            model.variables[index].assign(tf.expand_dims(from_to_variable_dict.get(original_var), 0))
            assigned_map.append((original_var, legacy_var))
            continue
        # Assign Patch embeddings
        if "patch_embeddings/conv2d/kernel:0" in legacy_var:
            model.variables[index].assign(tf.transpose(from_to_variable_dict.get(original_var), [2, 3, 1, 0]))
            assigned_map.append((original_var, legacy_var))
            continue

        if "query/kernel:0" in legacy_var or "key/kernel:0" in legacy_var or "value/kernel:0" in legacy_var:

            # huggingface (2D) to tf_transformers (3D)
            model.variables[index].assign(
                tf.reshape(
                    tf.transpose(from_to_variable_dict.get(original_var)),
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

        if (
            'intermediate/kernel:0' in legacy_var
            or 'output/kernel:0' in legacy_var
            or 'visual_projection/kernel:0' in legacy_var
            or 'text_projection/kernel:0' in legacy_var
        ):
            # huggingface (2D) to tf_transformers (3D)
            model.variables[index].assign(
                tf.transpose(from_to_variable_dict.get(original_var)),
            )
            assigned_map.append((original_var, legacy_var))
            continue

        model.variables[index].assign(from_to_variable_dict.get(original_var))
        assigned_map.append((original_var, legacy_var))


def convert_clip_pt(model, config, model_name):
    """PT converter
    Args:
        model_hf: HuggingFace Model (TF)
        model: tf_transformers model/layer
        config: dict
    Returns:
        a function
    """

    import torch
    from transformers import CLIPModel

    model_hf = CLIPModel.from_pretrained(model_name)

    vision_config = config['vision_config']
    text_config = config['text_config']

    # Convert vision model
    convert_image_model_pt(model, vision_config, model_hf)

    # Convert text model
    convert_text_model_pt(model, text_config, model_hf)

    batch_size = 2
    num_channels = 3
    image_size = vision_config['image_size']

    pixel_values = torch.rand(([batch_size, num_channels, image_size, image_size]))
    inputs_tf = tf.convert_to_tensor(pixel_values.numpy())
    # Reshape (b x channels x h x w)
    inputs_tf = tf.transpose(inputs_tf, [0, 2, 3, 1])

    input_ids = torch.randint(0, 10, (2, 7))
    input_ids_tf = tf.convert_to_tensor(input_ids.numpy())

    inputs_hf = {'pixel_values': pixel_values, 'input_ids': input_ids}
    inputs_tf = {'input_pixels': inputs_tf, 'input_ids': input_ids_tf}

    outputs_hf = model_hf(**inputs_hf)
    outputs_tf = model(inputs_tf)
    # Slightly bigger rtol
    tf.debugging.assert_near(outputs_tf['logits_per_text'], outputs_hf.logits_per_text.detach().numpy(), rtol=5.0)
    tf.debugging.assert_near(outputs_tf['logits_per_image'], outputs_hf.logits_per_image.detach().numpy(), rtol=5.0)
