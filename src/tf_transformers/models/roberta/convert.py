import numpy as np
import tensorflow as tf


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


def convert_roberta_pt(model, config, model_name):
    """PT converter
    Args:
        model_hf: HuggingFace Model (TF)
        model: tf_transformers model/layer
        config: dict
    Returns:
        a function
    """

    import torch

    # From vars (Transformer variables)
    from_model_vars = [
        "encoder.layer.{}.attention.self.query.weight",
        "encoder.layer.{}.attention.self.query.bias",
        "encoder.layer.{}.attention.self.key.weight",
        "encoder.layer.{}.attention.self.key.bias",
        "encoder.layer.{}.attention.self.value.weight",
        "encoder.layer.{}.attention.self.value.bias",
        "encoder.layer.{}.attention.output.dense.weight",
        "encoder.layer.{}.attention.output.dense.bias",
        "encoder.layer.{}.attention.output.LayerNorm.weight",
        "encoder.layer.{}.attention.output.LayerNorm.bias",
        "encoder.layer.{}.intermediate.dense.weight",
        "encoder.layer.{}.intermediate.dense.bias",
        "encoder.layer.{}.output.dense.weight",
        "encoder.layer.{}.output.dense.bias",
        "encoder.layer.{}.output.LayerNorm.weight",
        "encoder.layer.{}.output.LayerNorm.bias",
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

    # This dictionary maps from -> to dict names
    mapping_dict = {}
    for index in range(len(from_model_vars)):
        for i in range(config["num_hidden_layers"]):
            mapping_dict[from_model_vars[index].format(i)] = to_model_vars[index].format(i)

    # Word Embeddings
    mapping_dict["embeddings.word_embeddings.weight"] = "tf_transformers/roberta/word_embeddings/embeddings:0"
    # Positional Embedding
    mapping_dict["embeddings.position_embeddings.weight"] = "tf_transformers/roberta/positional_embeddings/embeddings:0"
    # Type Embeddings
    mapping_dict["embeddings.token_type_embeddings.weight"] = "tf_transformers/roberta/type_embeddings/embeddings:0"
    mapping_dict["embeddings.LayerNorm.weight"] = "tf_transformers/roberta/embeddings/layer_norm/gamma:0"
    mapping_dict["embeddings.LayerNorm.bias"] = "tf_transformers/roberta/embeddings/layer_norm/beta:0"
    mapping_dict["pooler.dense.weight"] = "tf_transformers/roberta/pooler_transform/kernel:0"
    mapping_dict["pooler.dense.bias"] = "tf_transformers/roberta/pooler_transform/bias:0"

    from transformers import RobertaModel

    model_hf = RobertaModel.from_pretrained(model_name)
    # HF model variable name to variable values, for fast retrieval
    from_to_variable_dict = {name: var.detach().numpy() for name, var in model_hf.named_parameters()}

    # We need variable name to the index where it is stored inside tf_transformers model
    tf_transformers_model_index_dict = {}
    for index, var in enumerate(model.variables):
        tf_transformers_model_index_dict[var.name] = index

        # In auto_regressive mode, positional embeddings variable name has
        # cond extra name. So, in case someone converts in that mode,
        # replace above mapping here, only for positional embeddings
        if var.name == "tf_transformers/roberta/cond/positional_embeddings/embeddings:0":
            mapping_dict[
                "embeddings.position_embeddings.weight"
            ] = "tf_transformers/roberta/cond/positional_embeddings/embeddings:0"
    # Start assigning HF values to tf_transformers
    # assigned_map and assigned_map_values are used for sanity check if needed
    assigned_map = []
    # assigned_map_values = []
    for original_var, legacy_var in mapping_dict.items():
        index = tf_transformers_model_index_dict[legacy_var]

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

        # Robert ave 514 embeddings, we are taking [2:] (512)
        if "positional_embeddings" in legacy_var:
            model.variables[index].assign(from_to_variable_dict.get(original_var)[2:])
            assigned_map.append((original_var, legacy_var))
            continue

        model.variables[index].assign(from_to_variable_dict.get(original_var))
        assigned_map.append((original_var, legacy_var))

    from transformers import RobertaTokenizer

    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    text = "[CLS] i want to [MASK] the car because it is cheap. [SEP]"
    inputs = tokenizer(text, return_tensors="pt")
    outputs_pt = model_hf(**inputs)
    outputs_pt = torch.argmax(outputs_pt.last_hidden_state, dim=2)[0].numpy()

    from transformers import RobertaForMaskedLM

    model_hf = RobertaForMaskedLM.from_pretrained(model_name)
    hf_vars = [
        "lm_head.bias",
        "lm_head.dense.weight",
        "lm_head.dense.bias",
        "lm_head.layer_norm.weight",
        "lm_head.layer_norm.bias",
    ]

    tf_vars = [
        "tf_transformers/roberta/logits_bias/bias:0",
        "tf_transformers/roberta/mlm/transform/dense/kernel:0",
        "tf_transformers/roberta/mlm/transform/dense/bias:0",
        "tf_transformers/roberta/mlm/transform/LayerNorm/gamma:0",
        "tf_transformers/roberta/mlm/transform/LayerNorm/beta:0",
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
    inputs_tf["input_type_ids"] = tf.zeros_like(inputs_tf["input_ids"])
    inputs_tf["input_mask"] = inputs["attention_mask"]
    outputs_tf = model(inputs_tf)
    text_tf = tokenizer.decode(tf.argmax(outputs_tf["token_logits"], axis=2)[0])

    assert text_pt == text_tf
    outputs_tf = tf.argmax(outputs_tf["token_embeddings"], axis=2)[0].numpy()
    tf.debugging.assert_equal(outputs_pt, outputs_tf)


def convert_roberta_tf(model, config, model_name):
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

    # BertModel
    from transformers import TFRobertaModel

    tf.keras.backend.clear_session()
    model_hf = TFRobertaModel.from_pretrained(model_name)

    # HF model variable name to variable values, for fast retrieval
    from_to_variable_dict = {var.name: var for var in model_hf.variables}

    # We need variable name to the index where it is stored inside tf_transformers model
    tf_transformers_model_index_dict = {}
    for index, var in enumerate(model.variables):
        tf_transformers_model_index_dict[var.name] = index
        # In auto_regressive mode, positional embeddings variable name has
        # cond extra name. So, in case someone converts in that mode,
        # replace above mapping here, only for positional embeddings
        if var.name == "tf_transformers/roberta/cond/positional_embeddings/embeddings:0":
            mapping_dict[
                "tf_roberta_model/roberta/embeddings/position_embeddings/embeddings:0"
            ] = "tf_transformers/roberta/cond/positional_embeddings/embeddings:0"

    # Start assigning HF values to tf_transformers
    # assigned_map and assigned_map_values are used for sanity check if needed
    assigned_map = []
    # assigned_map_values = []
    for original_var, legacy_var in mapping_dict.items():
        index = tf_transformers_model_index_dict[legacy_var]

        if "query/kernel:0" in legacy_var or "key/kernel:0" in legacy_var or "value/kernel:0" in legacy_var:

            # huggingface (2D) to tf_transformers (3D)
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
            continue

        if "self_attention_output/kernel:0" in legacy_var:
            # huggingface (3D) to tf_transformers (2D)
            model.variables[index].assign(
                tf.reshape(
                    from_to_variable_dict.get(original_var),
                    (config["embedding_size"], config["num_attention_heads"] * config["attention_head_size"]),
                )
            )
            assigned_map.append((original_var, legacy_var))
            continue

        if "self_attention_output/bias:0" in legacy_var:
            # huggingface (3D) to tf_transformers (2D)
            model.variables[index].assign(
                tf.reshape(
                    from_to_variable_dict.get(original_var),
                    (-1),
                )
            )
            assigned_map.append((original_var, legacy_var))
            continue

        # Robert ave 514 embeddings, we are taking [2:] (512)
        if "positional_embeddings" in legacy_var:
            model.variables[index].assign(from_to_variable_dict.get(original_var)[2:])
            assigned_map.append((original_var, legacy_var))
            continue

        model.variables[index].assign(from_to_variable_dict.get(original_var))
        assigned_map.append((original_var, legacy_var))

    from transformers import RobertaTokenizer

    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    text = "[CLS] i want to [MASK] the car because it is cheap. [SEP]"
    inputs = tokenizer(text, return_tensors="tf")
    outputs_hf = model_hf(**inputs)
    outputs_hf = tf.argmax(outputs_hf.last_hidden_state, axis=2)[0].numpy()

    # BertMLM
    from transformers import TFRobertaForMaskedLM

    model_hf = TFRobertaForMaskedLM.from_pretrained(model_name)
    hf_vars = [
        "tf_roberta_for_masked_lm/lm_head/bias:0",
        "tf_roberta_for_masked_lm/lm_head/dense/kernel:0",
        "tf_roberta_for_masked_lm/lm_head/dense/bias:0",
        "tf_roberta_for_masked_lm/lm_head/layer_norm/gamma:0",
        "tf_roberta_for_masked_lm/lm_head/layer_norm/beta:0",
    ]

    tf_vars = [
        "tf_transformers/roberta/logits_bias/bias:0",
        "tf_transformers/roberta/mlm/transform/dense/kernel:0",
        "tf_transformers/roberta/mlm/transform/dense/bias:0",
        "tf_transformers/roberta/mlm/transform/LayerNorm/gamma:0",
        "tf_transformers/roberta/mlm/transform/LayerNorm/beta:0",
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
    inputs_tf["input_type_ids"] = tf.zeros_like(inputs_tf["input_ids"])
    inputs_tf["input_mask"] = inputs["attention_mask"]
    outputs_tf = model(inputs_tf)
    text_tf = tokenizer.decode(tf.argmax(outputs_tf["token_logits"], axis=2)[0])

    assert text_hf == text_tf
    outputs_tf = tf.argmax(outputs_tf["token_embeddings"], axis=2)[0].numpy()
    tf.debugging.assert_equal(outputs_hf, outputs_tf)
