import numpy as np
import tensorflow as tf
from absl import logging

from tf_transformers.core import keras_utils


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


def convert_bart_pt(model, config, model_name):
    """PT converter
    Args:
        model_hf: HuggingFace Model (TF)
        model: tf_transformers model/layer
        config: dict
    Returns:
        a function
    """

    # When dropout, use_auto_regressive is enabled assertion won't work
    SKIP_ASSERT = False
    try:
        # LegacyLayer
        local_config = model._config_dict['decoder']
    except Exception as e:
        # LegacyModel
        import traceback

        print(traceback.format_exc())
        logging.error(e)
        local_config = model.model_config['decoder']

    if local_config['use_dropout']:
        logging.warn("Note: As `use_dropout` is True we will skip Assertions, please verify the model.")
        SKIP_ASSERT = True
    if local_config['use_auto_regressive']:
        raise ValueError(
            "Please save  model checkpoint without `use_auto_regressive` and then reload it with `use_auto_regressive`."
        )
        SKIP_ASSERT = True

    import torch

    from_model_vars = [
        'encoder.layers.{}.self_attn.q_proj.weight',
        'encoder.layers.{}.self_attn.q_proj.bias',
        'encoder.layers.{}.self_attn.k_proj.weight',
        'encoder.layers.{}.self_attn.k_proj.bias',
        'encoder.layers.{}.self_attn.v_proj.weight',
        'encoder.layers.{}.self_attn.v_proj.bias',
        'encoder.layers.{}.self_attn.out_proj.weight',
        'encoder.layers.{}.self_attn.out_proj.bias',
        'encoder.layers.{}.self_attn_layer_norm.weight',
        'encoder.layers.{}.self_attn_layer_norm.bias',
        'encoder.layers.{}.fc1.weight',
        'encoder.layers.{}.fc1.bias',
        'encoder.layers.{}.fc2.weight',
        'encoder.layers.{}.fc2.bias',
        'encoder.layers.{}.final_layer_norm.weight',
        'encoder.layers.{}.final_layer_norm.bias',
    ]

    to_model_vars = [
        'tf_transformers/bart_encoder/transformer/layer_{}/self_attention/query/kernel:0',
        'tf_transformers/bart_encoder/transformer/layer_{}/self_attention/query/bias:0',
        'tf_transformers/bart_encoder/transformer/layer_{}/self_attention/key/kernel:0',
        'tf_transformers/bart_encoder/transformer/layer_{}/self_attention/key/bias:0',
        'tf_transformers/bart_encoder/transformer/layer_{}/self_attention/value/kernel:0',
        'tf_transformers/bart_encoder/transformer/layer_{}/self_attention/value/bias:0',
        'tf_transformers/bart_encoder/transformer/layer_{}/self_attention_output/kernel:0',
        'tf_transformers/bart_encoder/transformer/layer_{}/self_attention_output/bias:0',
        'tf_transformers/bart_encoder/transformer/layer_{}/self_attention_layer_norm/gamma:0',
        'tf_transformers/bart_encoder/transformer/layer_{}/self_attention_layer_norm/beta:0',
        'tf_transformers/bart_encoder/transformer/layer_{}/intermediate/kernel:0',
        'tf_transformers/bart_encoder/transformer/layer_{}/intermediate/bias:0',
        'tf_transformers/bart_encoder/transformer/layer_{}/output/kernel:0',
        'tf_transformers/bart_encoder/transformer/layer_{}/output/bias:0',
        'tf_transformers/bart_encoder/transformer/layer_{}/output_layer_norm/gamma:0',
        'tf_transformers/bart_encoder/transformer/layer_{}/output_layer_norm/beta:0',
    ]

    # Simple Assertion
    assert len(from_model_vars) == len(to_model_vars)
    mapping_dict = {}

    for index in range(len(from_model_vars)):
        for i in range(config["num_hidden_layers"]):
            mapping_dict[from_model_vars[index].format(i)] = to_model_vars[index].format(i)

    # Word Embedding
    mapping_dict["shared.weight"] = "tf_transformers/bart_encoder/word_embeddings/embeddings:0"
    # Positional Embedding
    mapping_dict["encoder.embed_positions.weight"] = "tf_transformers/bart_encoder/positional_embeddings/embeddings:0"
    # Embedding Norm
    mapping_dict['encoder.layernorm_embedding.weight'] = 'tf_transformers/bart_encoder/embeddings/layer_norm/gamma:0'
    mapping_dict['encoder.layernorm_embedding.bias'] = 'tf_transformers/bart_encoder/embeddings/layer_norm/beta:0'

    # BartModel

    from transformers import BartModel

    model_hf = BartModel.from_pretrained(model_name)

    from_to_variable_dict = {name: var.detach().numpy() for name, var in model_hf.named_parameters()}

    tf_transformers_model_index_dict = {}
    for index, var in enumerate(model.variables):
        tf_transformers_model_index_dict[var.name] = index

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

        if (
            "intermediate/kernel:0" in legacy_var
            or "output/kernel:0" in legacy_var
            or "self_attention_output/kernel:0" in legacy_var
        ):
            # huggingface (torch transpose
            model.variables[index].assign(np.transpose(from_to_variable_dict.get(original_var)))
            assigned_map.append((original_var, legacy_var))
            continue

        # Bart have 1026 embeddings, we are taking [2:] (1024)
        if "positional_embeddings" in legacy_var:
            model.variables[index].assign(from_to_variable_dict.get(original_var)[2:])
            assigned_map.append((original_var, legacy_var))
            continue

        model.variables[index].assign(from_to_variable_dict.get(original_var))
        assigned_map.append((original_var, legacy_var))

    # Decoder Side
    # From vars (Transformer variables)
    from_model_vars = [
        'decoder.layers.{}.self_attn.q_proj.weight',
        'decoder.layers.{}.self_attn.q_proj.bias',
        'decoder.layers.{}.self_attn.k_proj.weight',
        'decoder.layers.{}.self_attn.k_proj.bias',
        'decoder.layers.{}.self_attn.v_proj.weight',
        'decoder.layers.{}.self_attn.v_proj.bias',
        'decoder.layers.{}.self_attn.out_proj.weight',
        'decoder.layers.{}.self_attn.out_proj.bias',
        'decoder.layers.{}.self_attn_layer_norm.weight',
        'decoder.layers.{}.self_attn_layer_norm.bias',
        'decoder.layers.{}.encoder_attn.q_proj.weight',
        'decoder.layers.{}.encoder_attn.q_proj.bias',
        'decoder.layers.{}.encoder_attn.k_proj.weight',
        'decoder.layers.{}.encoder_attn.k_proj.bias',
        'decoder.layers.{}.encoder_attn.v_proj.weight',
        'decoder.layers.{}.encoder_attn.v_proj.bias',
        'decoder.layers.{}.encoder_attn.out_proj.weight',
        'decoder.layers.{}.encoder_attn.out_proj.bias',
        'decoder.layers.{}.encoder_attn_layer_norm.weight',
        'decoder.layers.{}.encoder_attn_layer_norm.bias',
        'decoder.layers.{}.fc1.weight',
        'decoder.layers.{}.fc1.bias',
        'decoder.layers.{}.fc2.weight',
        'decoder.layers.{}.fc2.bias',
        'decoder.layers.{}.final_layer_norm.weight',
        'decoder.layers.{}.final_layer_norm.bias',
    ]

    to_model_vars = [
        'tf_transformers/bart_decoder/transformer/layer_{}/self_attention/query/kernel:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/self_attention/query/bias:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/self_attention/key/kernel:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/self_attention/key/bias:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/self_attention/value/kernel:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/self_attention/value/bias:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/self_attention_output/kernel:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/self_attention_output/bias:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/self_attention_layer_norm/gamma:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/self_attention_layer_norm/beta:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/cross_attention/query/kernel:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/cross_attention/query/bias:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/cross_attention/key/kernel:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/cross_attention/key/bias:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/cross_attention/value/kernel:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/cross_attention/value/bias:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/cross_attention_output/kernel:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/cross_attention_output/bias:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/cross_attention_layer_norm/gamma:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/cross_attention_layer_norm/beta:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/intermediate/kernel:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/intermediate/bias:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/output/kernel:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/output/bias:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/output_layer_norm/gamma:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/output_layer_norm/beta:0',
    ]

    # Simple Assertion
    assert len(from_model_vars) == len(to_model_vars)
    mapping_dict = {}

    for index in range(len(from_model_vars)):
        for i in range(config["num_hidden_layers"]):
            mapping_dict[from_model_vars[index].format(i)] = to_model_vars[index].format(i)

    # Positional Embedding
    mapping_dict["decoder.embed_positions.weight"] = "tf_transformers/bart_decoder/positional_embeddings/embeddings:0"
    # Embedding Norm
    mapping_dict['decoder.layernorm_embedding.weight'] = 'tf_transformers/bart_decoder/embeddings/layer_norm/gamma:0'
    mapping_dict['decoder.layernorm_embedding.bias'] = 'tf_transformers/bart_decoder/embeddings/layer_norm/beta:0'

    from_to_variable_dict = {name: var.detach().numpy() for name, var in model_hf.named_parameters()}

    tf_transformers_model_index_dict = {}
    for index, var in enumerate(model.variables):
        tf_transformers_model_index_dict[var.name] = index

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

        if (
            "intermediate/kernel:0" in legacy_var
            or "output/kernel:0" in legacy_var
            or "self_attention_output/kernel:0" in legacy_var
        ):
            # huggingface (torch transpose
            model.variables[index].assign(np.transpose(from_to_variable_dict.get(original_var)))

            assigned_map.append((original_var, legacy_var))
            continue

        # Bart ave 1026 embeddings, we are taking [2:] (1024)
        if "positional_embeddings" in legacy_var:
            model.variables[index].assign(from_to_variable_dict.get(original_var)[2:])
            assigned_map.append((original_var, legacy_var))
            continue

        model.variables[index].assign(from_to_variable_dict.get(original_var))
        assigned_map.append((original_var, legacy_var))

    if SKIP_ASSERT is False:
        from transformers import BartTokenizer

        tokenizer = BartTokenizer.from_pretrained(model_name)
        text = "This is a long sentence to check how close models are."
        inputs = tokenizer(text, return_tensors="pt")
        decoder_input_ids = torch.tensor([[2, 3, 175, 879]])
        with torch.no_grad():
            outputs_hf = model_hf(inputs["input_ids"], decoder_input_ids=decoder_input_ids)
        outputs_hf = torch.sum(outputs_hf["last_hidden_state"], dim=-1).numpy()

        inputs_tf = {}
        inputs = tokenizer(text, return_tensors="tf")
        inputs_tf["encoder_input_ids"] = inputs["input_ids"]
        inputs_tf["encoder_input_mask"] = inputs["attention_mask"]
        decoder_input_ids = tf.constant([[2, 3, 175, 879]])
        inputs_tf["decoder_input_ids"] = decoder_input_ids
        outputs_tf = model(inputs_tf)
        outputs_tf = tf.reduce_sum(outputs_tf["token_embeddings"], axis=-1).numpy()

        # Output embeddings check .
        if keras_utils.get_policy_name() == 'float32':
            tf.debugging.assert_near(outputs_hf, outputs_tf, rtol=1.0)


def convert_bart_tf(model, config, model_name):
    """TF converter
    Args:
        model_hf: HuggingFace Model (TF)
        model: tf_transformers model/layer
        config: dict
    Returns:
        a function
    """

    # When dropout, use_auto_regressive is enabled assertion won't work
    SKIP_ASSERT = False
    try:
        # LegacyLayer
        local_config = model._config_dict['decoder']
    except Exception as e:
        # LegacyModel
        import traceback

        print(traceback.format_exc())
        logging.error(e)
        local_config = model.model_config['decoder']

    if local_config['use_dropout']:
        logging.warn("Note: As `use_dropout` is True we will skip Assertions, please verify the model.")
        SKIP_ASSERT = True
    if local_config['use_auto_regressive']:
        raise ValueError(
            "Please save  model checkpoint without `use_auto_regressive` and then reload it with `use_auto_regressive`."
        )
        SKIP_ASSERT = True

    import transformers

    transformers.logging.set_verbosity_error()

    from_model_vars = [
        'tf_bart_model/model/encoder/layers.{}/self_attn/q_proj/kernel:0',
        'tf_bart_model/model/encoder/layers.{}/self_attn/q_proj/bias:0',
        'tf_bart_model/model/encoder/layers.{}/self_attn/k_proj/kernel:0',
        'tf_bart_model/model/encoder/layers.{}/self_attn/k_proj/bias:0',
        'tf_bart_model/model/encoder/layers.{}/self_attn/v_proj/kernel:0',
        'tf_bart_model/model/encoder/layers.{}/self_attn/v_proj/bias:0',
        'tf_bart_model/model/encoder/layers.{}/self_attn/out_proj/kernel:0',
        'tf_bart_model/model/encoder/layers.{}/self_attn/out_proj/bias:0',
        'tf_bart_model/model/encoder/layers.{}/self_attn_layer_norm/gamma:0',
        'tf_bart_model/model/encoder/layers.{}/self_attn_layer_norm/beta:0',
        'tf_bart_model/model/encoder/layers.{}/fc1/kernel:0',
        'tf_bart_model/model/encoder/layers.{}/fc1/bias:0',
        'tf_bart_model/model/encoder/layers.{}/fc2/kernel:0',
        'tf_bart_model/model/encoder/layers.{}/fc2/bias:0',
        'tf_bart_model/model/encoder/layers.{}/final_layer_norm/gamma:0',
        'tf_bart_model/model/encoder/layers.{}/final_layer_norm/beta:0',
    ]

    to_model_vars = [
        'tf_transformers/bart_encoder/transformer/layer_{}/self_attention/query/kernel:0',
        'tf_transformers/bart_encoder/transformer/layer_{}/self_attention/query/bias:0',
        'tf_transformers/bart_encoder/transformer/layer_{}/self_attention/key/kernel:0',
        'tf_transformers/bart_encoder/transformer/layer_{}/self_attention/key/bias:0',
        'tf_transformers/bart_encoder/transformer/layer_{}/self_attention/value/kernel:0',
        'tf_transformers/bart_encoder/transformer/layer_{}/self_attention/value/bias:0',
        'tf_transformers/bart_encoder/transformer/layer_{}/self_attention_output/kernel:0',
        'tf_transformers/bart_encoder/transformer/layer_{}/self_attention_output/bias:0',
        'tf_transformers/bart_encoder/transformer/layer_{}/self_attention_layer_norm/gamma:0',
        'tf_transformers/bart_encoder/transformer/layer_{}/self_attention_layer_norm/beta:0',
        'tf_transformers/bart_encoder/transformer/layer_{}/intermediate/kernel:0',
        'tf_transformers/bart_encoder/transformer/layer_{}/intermediate/bias:0',
        'tf_transformers/bart_encoder/transformer/layer_{}/output/kernel:0',
        'tf_transformers/bart_encoder/transformer/layer_{}/output/bias:0',
        'tf_transformers/bart_encoder/transformer/layer_{}/output_layer_norm/gamma:0',
        'tf_transformers/bart_encoder/transformer/layer_{}/output_layer_norm/beta:0',
    ]

    # Simple Assertion
    assert len(from_model_vars) == len(to_model_vars)
    mapping_dict = {}

    for index in range(len(from_model_vars)):
        for i in range(config["num_hidden_layers"]):
            mapping_dict[from_model_vars[index].format(i)] = to_model_vars[index].format(i)

    # Word Embedding
    mapping_dict["model.shared/model.shared/weight:0"] = "tf_transformers/bart_encoder/word_embeddings/embeddings:0"
    # Positional Embedding
    mapping_dict[
        "tf_bart_model/model/encoder/embed_positions/weight:0"
    ] = "tf_transformers/bart_encoder/positional_embeddings/embeddings:0"
    # Embedding Norm
    mapping_dict[
        'tf_bart_model/model/encoder/layernorm_embedding/gamma:0'
    ] = 'tf_transformers/bart_encoder/embeddings/layer_norm/gamma:0'
    mapping_dict[
        'tf_bart_model/model/encoder/layernorm_embedding/beta:0'
    ] = 'tf_transformers/bart_encoder/embeddings/layer_norm/beta:0'

    # BartModel

    from transformers import TFBartModel

    model_hf = TFBartModel.from_pretrained(model_name)

    from_to_variable_dict = {var.name: var for var in model_hf.variables}

    tf_transformers_model_index_dict = {}
    for index, var in enumerate(model.variables):
        tf_transformers_model_index_dict[var.name] = index

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

        # Bart ave 1026 embeddings, we are taking [2:] (1024)
        if "positional_embeddings" in legacy_var:
            model.variables[index].assign(from_to_variable_dict.get(original_var)[2:])
            assigned_map.append((original_var, legacy_var))
            continue

        model.variables[index].assign(from_to_variable_dict.get(original_var))
        assigned_map.append((original_var, legacy_var))

    # Decoder Side
    # From vars (Transformer variables)
    from_model_vars = [
        'tf_bart_model/model/decoder/layers.{}/self_attn/q_proj/kernel:0',
        'tf_bart_model/model/decoder/layers.{}/self_attn/q_proj/bias:0',
        'tf_bart_model/model/decoder/layers.{}/self_attn/k_proj/kernel:0',
        'tf_bart_model/model/decoder/layers.{}/self_attn/k_proj/bias:0',
        'tf_bart_model/model/decoder/layers.{}/self_attn/v_proj/kernel:0',
        'tf_bart_model/model/decoder/layers.{}/self_attn/v_proj/bias:0',
        'tf_bart_model/model/decoder/layers.{}/self_attn/out_proj/kernel:0',
        'tf_bart_model/model/decoder/layers.{}/self_attn/out_proj/bias:0',
        'tf_bart_model/model/decoder/layers.{}/self_attn_layer_norm/gamma:0',
        'tf_bart_model/model/decoder/layers.{}/self_attn_layer_norm/beta:0',
        'tf_bart_model/model/decoder/layers.{}/encoder_attn/q_proj/kernel:0',
        'tf_bart_model/model/decoder/layers.{}/encoder_attn/q_proj/bias:0',
        'tf_bart_model/model/decoder/layers.{}/encoder_attn/k_proj/kernel:0',
        'tf_bart_model/model/decoder/layers.{}/encoder_attn/k_proj/bias:0',
        'tf_bart_model/model/decoder/layers.{}/encoder_attn/v_proj/kernel:0',
        'tf_bart_model/model/decoder/layers.{}/encoder_attn/v_proj/bias:0',
        'tf_bart_model/model/decoder/layers.{}/encoder_attn/out_proj/kernel:0',
        'tf_bart_model/model/decoder/layers.{}/encoder_attn/out_proj/bias:0',
        'tf_bart_model/model/decoder/layers.{}/encoder_attn_layer_norm/gamma:0',
        'tf_bart_model/model/decoder/layers.{}/encoder_attn_layer_norm/beta:0',
        'tf_bart_model/model/decoder/layers.{}/fc1/kernel:0',
        'tf_bart_model/model/decoder/layers.{}/fc1/bias:0',
        'tf_bart_model/model/decoder/layers.{}/fc2/kernel:0',
        'tf_bart_model/model/decoder/layers.{}/fc2/bias:0',
        'tf_bart_model/model/decoder/layers.{}/final_layer_norm/gamma:0',
        'tf_bart_model/model/decoder/layers.{}/final_layer_norm/beta:0',
    ]

    to_model_vars = [
        'tf_transformers/bart_decoder/transformer/layer_{}/self_attention/query/kernel:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/self_attention/query/bias:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/self_attention/key/kernel:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/self_attention/key/bias:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/self_attention/value/kernel:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/self_attention/value/bias:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/self_attention_output/kernel:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/self_attention_output/bias:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/self_attention_layer_norm/gamma:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/self_attention_layer_norm/beta:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/cross_attention/query/kernel:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/cross_attention/query/bias:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/cross_attention/key/kernel:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/cross_attention/key/bias:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/cross_attention/value/kernel:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/cross_attention/value/bias:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/cross_attention_output/kernel:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/cross_attention_output/bias:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/cross_attention_layer_norm/gamma:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/cross_attention_layer_norm/beta:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/intermediate/kernel:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/intermediate/bias:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/output/kernel:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/output/bias:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/output_layer_norm/gamma:0',
        'tf_transformers/bart_decoder/transformer/layer_{}/output_layer_norm/beta:0',
    ]

    # Simple Assertion
    assert len(from_model_vars) == len(to_model_vars)
    mapping_dict = {}

    for index in range(len(from_model_vars)):
        for i in range(config["num_hidden_layers"]):
            mapping_dict[from_model_vars[index].format(i)] = to_model_vars[index].format(i)

    # Positional Embedding
    mapping_dict[
        "tf_bart_model/model/decoder/embed_positions/weight:0"
    ] = "tf_transformers/bart_decoder/positional_embeddings/embeddings:0"
    # Embedding Norm
    mapping_dict[
        'tf_bart_model/model/decoder/layernorm_embedding/gamma:0'
    ] = 'tf_transformers/bart_decoder/embeddings/layer_norm/gamma:0'
    mapping_dict[
        'tf_bart_model/model/decoder/layernorm_embedding/beta:0'
    ] = 'tf_transformers/bart_decoder/embeddings/layer_norm/beta:0'

    from_to_variable_dict = {var.name: var for var in model_hf.variables}

    tf_transformers_model_index_dict = {}
    for index, var in enumerate(model.variables):
        tf_transformers_model_index_dict[var.name] = index

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

        # Bart ave 1026 embeddings, we are taking [2:] (1024)
        if "positional_embeddings" in legacy_var:
            model.variables[index].assign(from_to_variable_dict.get(original_var)[2:])
            assigned_map.append((original_var, legacy_var))
            continue

        model.variables[index].assign(from_to_variable_dict.get(original_var))
        assigned_map.append((original_var, legacy_var))

    if SKIP_ASSERT is False:
        from transformers import BartTokenizer

        tokenizer = BartTokenizer.from_pretrained(model_name)
        text = "This is a long sentence to check how close models are."
        inputs = tokenizer(text, return_tensors="tf")
        decoder_input_ids = tf.constant([[2, 3, 175, 879]])
        outputs_hf = model_hf(inputs["input_ids"], decoder_input_ids=decoder_input_ids)
        outputs_hf = tf.reduce_sum(outputs_hf["last_hidden_state"], axis=-1).numpy()

        inputs_tf = {}
        inputs_tf["encoder_input_ids"] = inputs["input_ids"]
        inputs_tf["encoder_input_mask"] = inputs["attention_mask"]
        inputs_tf["decoder_input_ids"] = decoder_input_ids
        outputs_tf = model(inputs_tf)
        outputs_tf = tf.reduce_sum(outputs_tf["token_embeddings"], axis=-1).numpy()

        # Output embeddings check .
        if keras_utils.get_policy_name() == 'float32':
            tf.debugging.assert_near(outputs_hf, outputs_tf, rtol=1.0)
