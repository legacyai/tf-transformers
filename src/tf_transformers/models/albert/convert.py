import tensorflow as tf
import numpy as np


def assert_model_results(model):
    def get_expected_text(model_name):
        print("Model name", model_name)
        if model_name == "bert_base_uncased":
            expected_text = ". i want to buy the car because it is cheap.."
        if model_name == "bert_base_cased" or model_name == "bert_large_cased":
            expected_text = ".. want to buy the car because it is cheap.."
        if model_name == "bert_large_cased":
            expected_text = ".. want to buy the car because it is cheap.."
        return expected_text

    def assert_bert(model_name):
        from transformers import BertTokenizer

        model_name = model_name.replace("_", "-")
        tokenizer = BertTokenizer.from_pretrained(model_name)
        text = "[CLS] i want to [MASK] the car because it is cheap. [SEP]"
        input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

        input_ids = tf.constant([input_ids])

        inputs = {}
        inputs["input_ids"] = input_ids
        inputs["input_mask"] = tf.ones_like(input_ids)
        inputs["input_type_ids"] = tf.zeros_like(input_ids)

        results = model(inputs)
        expected_text = get_expected_text(model_name)
        decoded_text = tokenizer.decode(tf.argmax(results["token_logits"], axis=2)[0].numpy())
        assert expected_text == decoded_text

    def assert_model(model_name):
        assert_bert(model_name)

    return assert_model


def convert_albert_pt(model, config, model_name):
    """TF converter
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
        "embeddings.word_embeddings.weight",
        "embeddings.token_type_embeddings.weight",
        "embeddings.position_embeddings.weight",
        "embeddings.LayerNorm.weight",
        "embeddings.LayerNorm.bias",
        "encoder.embedding_hidden_mapping_in.weight",
        "encoder.embedding_hidden_mapping_in.bias",
        "encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight",
        "encoder.albert_layer_groups.0.albert_layers.0.attention.query.bias",
        "encoder.albert_layer_groups.0.albert_layers.0.attention.key.weight",
        "encoder.albert_layer_groups.0.albert_layers.0.attention.key.bias",
        "encoder.albert_layer_groups.0.albert_layers.0.attention.value.weight",
        "encoder.albert_layer_groups.0.albert_layers.0.attention.value.bias",
        "encoder.albert_layer_groups.0.albert_layers.0.attention.dense.weight",
        "encoder.albert_layer_groups.0.albert_layers.0.attention.dense.bias",
        "encoder.albert_layer_groups.0.albert_layers.0.attention.LayerNorm.weight",
        "encoder.albert_layer_groups.0.albert_layers.0.attention.LayerNorm.bias",
        "encoder.albert_layer_groups.0.albert_layers.0.ffn.weight",
        "encoder.albert_layer_groups.0.albert_layers.0.ffn.bias",
        "encoder.albert_layer_groups.0.albert_layers.0.ffn_output.weight",
        "encoder.albert_layer_groups.0.albert_layers.0.ffn_output.bias",
        "encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.weight",
        "encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.bias",
        "pooler.weight",
        "pooler.bias",
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

    # BertModel
    from transformers import AlbertModel

    model_hf = AlbertModel.from_pretrained(model_name)

    # HF model variable name to variable values, for fast retrieval
    from_to_variable_dict = {name: var.detach().numpy() for name, var in model_hf.named_parameters()}

    tf_transformers_model_index_dict = {}
    for index, var in enumerate(model.variables):
        tf_transformers_model_index_dict[var.name] = index

    # legacy_ai <-- HuggingFace
    assigned_map = []
    assigned_map_values = []
    for original_var, legacy_var in mapping_dict.items():

        index = tf_transformers_model_index_dict[legacy_var]

        if "embedding_projection/kernel:0" in legacy_var:
            model.variables[index].assign(np.transpose(from_to_variable_dict.get(original_var)))
            continue

        if "query/kernel:0" in legacy_var or "key/kernel:0" in legacy_var or "value/kernel:0" in legacy_var:
            # huggingface (2D) to tf_transformers (3D)
            model.variables[index].assign(
                np.reshape(
                    np.transpose(from_to_variable_dict.get(original_var)),
                    (
                        config["embedding_projection_size"],
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
            # assigned_map_values.append((tf.reduce_sum(\
            # from_to_variable_dict.get(original_var)).numpy(),\
            #  tf.reduce_sum(model.variables[index]).numpy()))
            continue

        if "self_attention_output/kernel:0" in legacy_var:
            # huggingface (3D) to tf_transformers (2D)
            model.variables[index].assign(
                np.reshape(
                    np.transpose(from_to_variable_dict.get(original_var)),
                    (
                        config["embedding_projection_size"],
                        config["num_attention_heads"] * config["attention_head_size"],
                    ),
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

        model.variables[index].assign(from_to_variable_dict.get(original_var))
        assigned_map.append((original_var, legacy_var))

    from transformers import AlbertTokenizer

    tokenizer = AlbertTokenizer.from_pretrained(model_name)
    text = "[CLS] i want to [MASK] the car because it is cheap. [SEP]"
    inputs = tokenizer(text, return_tensors="pt")
    outputs_pt = model_hf(**inputs)
    outputs_pt = torch.argmax(outputs_pt.last_hidden_state, dim=2)[0].numpy()

    # BertMLM
    from transformers import AlbertForMaskedLM

    model_hf = AlbertForMaskedLM.from_pretrained(model_name)
    hf_vars = [
        "predictions.bias",
        "predictions.dense.weight",
        "predictions.dense.bias",
        "predictions.LayerNorm.weight",
        "predictions.LayerNorm.bias",
    ]

    tf_vars = [
        "tf_transformers/albert/logits_bias/bias:0",
        "tf_transformers/albert/mlm/transform/dense/kernel:0",
        "tf_transformers/albert/mlm/transform/dense/bias:0",
        "tf_transformers/albert/mlm/transform/LayerNorm/gamma:0",
        "tf_transformers/albert/mlm/transform/LayerNorm/beta:0",
    ]
    mapping_dict = dict(zip(tf_vars, hf_vars))
    # HF model variable name to variable values, for fast retrieval
    hf_variable_dict = {name: var.detach().numpy() for name, var in model_hf.named_parameters() if name in hf_vars}
    for var in model.variables:
        if var.name in tf_vars:
            hf_var_name = mapping_dict[var.name]

            if "dense/kernel:0" in var.name:
                var.assign(np.transpose(hf_variable_dict[hf_var_name]))
                continue
            var.assign(hf_variable_dict[hf_var_name])

    inputs = tokenizer(text, return_tensors="pt")
    outputs_pt_mlm = model_hf(**inputs)
    text_pt = tokenizer.decode(torch.argmax(outputs_pt_mlm[0], dim=2)[0])
    del model_hf

    inputs = tokenizer(text, return_tensors="tf")
    inputs_tf = {}
    inputs_tf["input_ids"] = inputs["input_ids"]
    inputs_tf["input_type_ids"] = inputs["token_type_ids"]
    inputs_tf["input_mask"] = inputs["attention_mask"]
    outputs_tf = model(inputs_tf)
    text_tf = tokenizer.decode(tf.argmax(outputs_tf["token_logits"], axis=2)[0])

    assert text_pt == text_tf
    outputs_tf = tf.argmax(outputs_tf["token_embeddings"], axis=2)[0].numpy()
    assert np.allclose(outputs_pt, outputs_tf, rtol=1.0) == True


def convert_albert_tf(model, config, model_name):
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

    # BertModel
    from transformers import TFAlbertModel

    tf.keras.backend.clear_session()
    model_hf = TFAlbertModel.from_pretrained(model_name)

    from_to_variable_dict = {var.name: var for var in model_hf.variables}

    tf_transformers_model_index_dict = {}
    for index, var in enumerate(model.variables):
        tf_transformers_model_index_dict[var.name] = index

    # legacy_ai <-- HuggingFace
    assigned_map = []
    assigned_map_values = []
    for original_var, legacy_var in mapping_dict.items():

        index = tf_transformers_model_index_dict[legacy_var]

        if "query/kernel:0" in legacy_var or "key/kernel:0" in legacy_var or "value/kernel:0" in legacy_var:
            # huggingface (2D) to tf_transformers (3D)
            model.variables[index].assign(
                tf.reshape(
                    from_to_variable_dict.get(original_var),
                    (
                        config["embedding_projection_size"],
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
            model.variables[index].assign(
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

        model.variables[index].assign(from_to_variable_dict.get(original_var))
        assigned_map.append((original_var, legacy_var))

    from transformers import AlbertTokenizer

    tokenizer = AlbertTokenizer.from_pretrained(model_name)
    text = "[CLS] i want to [MASK] the car because it is cheap. [SEP]"
    inputs = tokenizer(text, return_tensors="tf")
    outputs_hf = model_hf(**inputs)
    outputs_hf = tf.argmax(outputs_hf.last_hidden_state, axis=2)[0].numpy()

    # BertMLM
    from transformers import TFAlbertForMaskedLM

    tf.keras.backend.clear_session()
    model_hf = TFAlbertForMaskedLM.from_pretrained(model_name)
    hf_vars = [
        "tf_albert_for_masked_lm/predictions/bias:0",
        "tf_albert_for_masked_lm/predictions/dense/kernel:0",
        "tf_albert_for_masked_lm/predictions/dense/bias:0",
        "tf_albert_for_masked_lm/predictions/LayerNorm/gamma:0",
        "tf_albert_for_masked_lm/predictions/LayerNorm/beta:0",
    ]

    tf_vars = [
        "tf_transformers/albert/logits_bias/bias:0",
        "tf_transformers/albert/mlm/transform/dense/kernel:0",
        "tf_transformers/albert/mlm/transform/dense/bias:0",
        "tf_transformers/albert/mlm/transform/LayerNorm/gamma:0",
        "tf_transformers/albert/mlm/transform/LayerNorm/beta:0",
    ]
    mapping_dict = dict(zip(tf_vars, hf_vars))
    # HF model variable name to variable values, for fast retrieval
    hf_variable_dict = {var.name: var for var in model_hf.variables}
    for var in model.variables:
        if var.name in tf_vars:
            hf_var_name = mapping_dict[var.name]
            var.assign(hf_variable_dict[hf_var_name])

    inputs = tokenizer(text, return_tensors="tf")
    outputs_hf_mlm = model_hf(**inputs)
    text_hf = tokenizer.decode(tf.argmax(outputs_hf_mlm[0], axis=2)[0])
    del model_hf

    inputs_tf = {}
    inputs_tf["input_ids"] = inputs["input_ids"]
    inputs_tf["input_type_ids"] = inputs["token_type_ids"]
    inputs_tf["input_mask"] = inputs["attention_mask"]
    outputs_tf = model(inputs_tf)
    text_tf = tokenizer.decode(tf.argmax(outputs_tf["token_logits"], axis=2)[0])

    assert text_hf == text_tf
    outputs_tf = tf.argmax(outputs_tf["token_embeddings"], axis=2)[0].numpy()
    assert np.allclose(outputs_hf, outputs_tf, rtol=1.0) == True
