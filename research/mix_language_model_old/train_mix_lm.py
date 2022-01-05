import glob
import json

import hydra
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf
from transformers import T5Tokenizer

from tf_transformers.core import TPUTrainer
from tf_transformers.data import TFReader
from tf_transformers.data.processors.mlm import (
    dynamic_causal_lm_from_features,
    dynamic_masking_from_features,
    dynamic_prefix_lm_from_features,
)
from tf_transformers.data.utils import separate_x_y
from tf_transformers.layers.mask import prefix_mask
from tf_transformers.losses import cross_entropy_loss
from tf_transformers.optimization import create_optimizer
from tf_transformers.text import SentencepieceTokenizer
from tf_transformers.utils import tf_utils


def load_tokenizer(cfg):
    """Load tf text based tokenizer"""
    model_file_path = cfg.tokenizer.model_file_path
    do_lower_case = cfg.tokenizer.do_lower_case
    special_tokens = cfg.tokenizer.special_tokens

    tokenizer_layer = SentencepieceTokenizer(
        model_file_path=model_file_path, lower_case=do_lower_case, special_tokens=special_tokens
    )

    return tokenizer_layer


def get_tfdataset_from_tfrecords(tfrecord_path_list):
    """Get tf dataset from tfrecords"""
    all_files = []
    for tfrecord_path in tfrecord_path_list:
        all_files.extend(glob.glob("{}/*.tfrecord".format(tfrecord_path)))
    schema = json.load(open("{}/schema.json".format(tfrecord_path)))
    tf_reader = TFReader(schema=schema, tfrecord_files=all_files)
    train_dataset = tf_reader.read_record()
    return train_dataset


def get_dataset(
    tfrecord_path_list,
    max_seq_len,
    max_predictions_per_batch,
    vocab_size,
    cls_token_id,
    sep_token_id,
    unk_token_id,
    pad_token_id,
    mask_token_id,
    batch_size,
    min_sen_len,
):
    """Get dataset after mlm from TFRecords"""

    def filter_by_length(x, min_sen_len):
        """Filter by minimum sentence length (subwords)"""
        return tf.squeeze(tf.greater_equal(tf.shape(x["input_ids"]), tf.constant(min_sen_len)), axis=0)

    def filter_by_batch(x, y, batch_size):
        """Filter by batch size"""
        x_batch = tf.shape(x["input_ids"])[0]
        return tf.equal(x_batch, tf.constant(batch_size))

    def prepare_3d_input_mask_mlm(input_mask):
        """Prepare 3D mask from 2D"""
        batch_size = tf.shape(input_mask)[0]
        seq_length = tf.shape(input_mask)[1]

        to_mask = tf.cast(tf.reshape(input_mask, [batch_size, 1, seq_length]), dtype=input_mask.dtype)
        broadcast_ones = tf.ones(shape=[batch_size, seq_length, 1], dtype=input_mask.dtype)

        mask = broadcast_ones * to_mask

        return tf.cast(mask, tf.float32)

    def attention_mask_square(nd):
        """1's in the lower triangle, counting from the lower right corner.

        Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
        """
        dtype = tf_utils.get_dtype()
        ns = nd
        i = tf.range(nd)[:, None]
        j = tf.range(ns)
        m = i >= j - ns + nd
        return tf.cast(m, dtype)

    def mask_causal_mask(input_ids):
        input_ids = tf.expand_dims(input_ids, 0)
        from_shape = tf_utils.get_shape_list(input_ids, expected_rank=[2, 3])
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]

        # 2D Lower Triangular Mask
        from_mask = attention_mask_square(from_seq_length)

        # Replicate 2D `N` times
        mask = tf.cast(tf.ones([batch_size, 1, 1]), from_mask.dtype) * from_mask

        return tf.cast(tf.squeeze(mask, axis=0), tf.float32)

    # Dynamic MLM
    dynamic_mlm_fn = dynamic_masking_from_features(
        max_seq_len,
        max_predictions_per_batch,
        vocab_size,
        cls_token_id,
        sep_token_id,
        unk_token_id,
        pad_token_id,
        mask_token_id,
    )

    # Dynamic Prefix LM
    dynamic_prefix_lm = dynamic_prefix_lm_from_features(max_seq_len, cls_token_id, sep_token_id)

    # Dynamic Causal LM
    dynamic_causal_lm = dynamic_causal_lm_from_features(max_seq_len, cls_token_id, sep_token_id)

    train_dataset = get_tfdataset_from_tfrecords(tfrecord_path_list)

    if min_sen_len and min_sen_len > 0:
        train_dataset = train_dataset.filter(lambda x: filter_by_length(x, min_sen_len))

    # prob check has to be inside map
    # otherwise things become deterministic
    def get_dataset_based_on_prob(item):
        """Map function"""

        def add_mark(x, mode, prob):
            """Check are we getting all if conditions with equal probability"""
            x["mode"] = [mode]
            x["prob"] = [prob]
            return x

        def map_mlm(x):
            """MLM"""
            x["input_ids"] = tf.RaggedTensor.from_tensor(tf.expand_dims(x["input_ids"], axis=0))
            x_copy, y_copy = dynamic_mlm_fn(x)
            # Squeeze
            x = {}
            for name, v_tensor in x_copy.items():
                x[name] = tf.squeeze(v_tensor, axis=0)
            for name, v_tensor in y_copy.items():
                x[name] = tf.squeeze(v_tensor, axis=0)
            x["3d_mask"] = tf.squeeze(prepare_3d_input_mask_mlm(x_copy["input_mask"]), axis=0)

            return x

        def map_pcmlm(x):
            """Prefix Causal LM"""
            x, y = dynamic_prefix_lm(x)
            x["3d_mask"] = prefix_mask(tf.expand_dims(x["input_mask"], axis=0))
            for name, v_tensor in y.items():
                x[name] = v_tensor
            return x

        def map_cmlm(x):
            """Causal LM"""
            x, y = dynamic_causal_lm(x)
            x["3d_mask"] = mask_causal_mask(x["input_mask"])
            for name, v_tensor in y.items():
                x[name] = v_tensor
            return x

        prob = tf.random.uniform(shape=())
        # Keep a copy like this importatnt
        # otherwise transformation in first if cond might affect other
        input_ids = item["input_ids"]

        # Do MLM
        if prob <= 0.33:
            x = map_mlm(item)
            # Cast
            x["masked_lm_positions"] = tf.cast(x["masked_lm_positions"], dtype=tf.int32)
            x["masked_lm_weights"] = tf.cast(x["masked_lm_weights"], dtype=tf.int32)
            x["input_mask"] = tf.cast(x["3d_mask"], tf.int32)
            del x["3d_mask"]
            # x = add_mark(x, "mlm", prob)

        # Prefix CLM
        elif prob < 0.66:
            x = map_pcmlm({"input_ids": input_ids})
            x["input_mask"] = tf.cast(x["3d_mask"], tf.int32)
            del x["3d_mask"]
            # x = add_mark(x, "prefix", prob)

        else:
            x = map_cmlm({"input_ids": input_ids})
            x["input_mask"] = tf.cast(x["3d_mask"], tf.int32)
            del x["3d_mask"]
            # x = add_mark(x, "causal", prob)
        return x

    train_dataset = train_dataset.map(get_dataset_based_on_prob, num_parallel_calls=tf.data.AUTOTUNE)
    _padded_shapes = {
        'input_ids': [max_seq_len],
        'input_type_ids': [max_seq_len],
        'input_mask': [max_seq_len, max_seq_len],
        'masked_lm_positions': [max_seq_len],
        'masked_lm_labels': [max_seq_len],
        'masked_lm_weights': [max_seq_len],
    }

    train_dataset = train_dataset.padded_batch(batch_size, padded_shapes=_padded_shapes)
    x_keys = ['input_ids', 'input_type_ids', 'input_mask', 'masked_lm_positions']
    y_keys = ['masked_lm_labels', 'masked_lm_weights']
    train_dataset = train_dataset.map(lambda x: separate_x_y(x, x_keys, y_keys))
    # train_dataset = auto_batch(
    #     train_dataset,
    #     batch_size,
    #     x_keys=["input_ids", "input_type_ids", "input_mask", "masked_lm_positions"],
    #     y_keys=["masked_lm_labels", "masked_lm_weights"],
    #     shuffle=True,
    # )
    train_dataset = train_dataset.filter(lambda x, y: filter_by_batch(x, y, batch_size))
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    return train_dataset


def get_model(vocab_size, batch_size):
    """Model"""

    def model_fn():
        config = {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "intermediate_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "embedding_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "attention_head_size": 64,
            "num_hidden_layers": 12,
            "type_vocab_size": 1,
            "vocab_size": vocab_size,
            "layer_norm_epsilon": 1e-12,
        }

        from tf_transformers.core import LegacyModel
        from tf_transformers.models import BertEncoder

        class MixEncoder(BertEncoder):
            def __init__(self, config, **kwargs):
                print(kwargs)
                super(MixEncoder, self).__init__(config, **kwargs)

            def get_model(self, initialize_only=False):
                """Convert tf.keras.Layer to a tf.keras.Model/LegacyModel.
                Args:
                    self: model (tf.keras.Layer) instance
                """

                input_ids = tf.keras.layers.Input(
                    shape=(self._sequence_length,),
                    batch_size=self._batch_size,
                    dtype=tf.int32,
                    name="input_ids",
                )
                input_mask = tf.keras.layers.Input(
                    shape=(self._sequence_length, self._sequence_length),
                    batch_size=self._batch_size,
                    dtype=tf.int32,
                    name="input_mask",
                )
                input_type_ids = tf.keras.layers.Input(
                    shape=(self._sequence_length,),
                    batch_size=self._batch_size,
                    dtype=tf.int32,
                    name="input_type_ids",
                )
                masked_lm_positions = tf.keras.layers.Input(
                    shape=(None,),
                    batch_size=self._batch_size,
                    dtype=tf.int32,
                    name="masked_lm_positions",
                )
                inputs = {}
                inputs["input_ids"] = input_ids  # Default
                # if mask_mode != 'causal', user has to provde mask
                if self._mask_mode != "causal":
                    inputs["input_mask"] = input_mask
                # If type mebddings required
                if self._type_embeddings_layer:
                    inputs["input_type_ids"] = input_type_ids
                # if masked_lm_positions
                if self._use_masked_lm_positions:
                    inputs["masked_lm_positions"] = masked_lm_positions

                layer_outputs = self(inputs)
                if initialize_only:
                    return inputs, layer_outputs

                # Adding model_config is a hack
                model = LegacyModel(inputs=inputs, outputs=layer_outputs, name=self._model_name)
                model.model_config = self._config_dict
                return model

            def call_encoder(self, inputs):
                """Forward pass of an Encoder

                Args:
                    inputs ([dict of tf.Tensor]): This is the input to the model.

                    'input_ids'         --> tf.int32 (b x s)
                    'input_mask'        --> tf.int32 (b x s) # optional
                    'input_type_ids'    --> tf.int32 (b x s) # optional

                Returns:
                    [dict of tf.Tensor]: Output from the model

                    'cls_output'        --> tf.float32 (b x s) # optional
                    'token_embeddings'  --> tf.float32 (b x s x h)
                    'all_layer_token_embeddings' --> tf.float32 (List of (b x s x h)
                                                    from all layers)
                    'all_layer_cls_output'       --> tf.float32 (List of (b x s)
                                                    from all layers)
                """

                # 1. Collect Word Embeddings
                input_ids = inputs["input_ids"]
                sequence_length = tf.shape(input_ids)[1]
                embeddings = self._embedding_layer(input_ids)
                # Add word_embeddings + position_embeddings + type_embeddings
                if self._type_embeddings_layer:
                    input_type_ids = inputs["input_type_ids"]
                    type_embeddings = self._type_embeddings_layer(input_type_ids)
                    embeddings = embeddings + type_embeddings
                if self._positional_embedding_layer:
                    positional_embeddings = self._positional_embedding_layer(tf.range(sequence_length))
                    embeddings = embeddings + positional_embeddings

                # 2. Norm + dropout
                embeddings = self._embedding_norm(embeddings)
                embeddings = self._embedding_dropout(embeddings, training=self._use_dropout)

                # 3. Attention  Mask
                attention_mask = inputs['input_mask']

                # 4. Transformer Outputs
                encoder_outputs = []
                for i in range(self._config_dict["num_hidden_layers"]):
                    layer = self._transformer_layers[i]
                    embeddings, _, _ = layer([embeddings, attention_mask])
                    encoder_outputs.append(embeddings)

                # First word of last layer outputs [CLS]
                cls_token_tensor = tf.keras.layers.Lambda(lambda x: tf.squeeze(x[:, 0:1, :], axis=1))(
                    encoder_outputs[-1]
                )
                # batch_size x embedding_size
                cls_output = self._pooler_layer(cls_token_tensor)
                # batch_size x sequence_length x embedding_size
                token_embeddings = encoder_outputs[-1]

                # check for masked lm positions
                # only for encoder forward pass. This is for MaskedLM training
                if "masked_lm_positions" in inputs:
                    masked_lm_positions = inputs["masked_lm_positions"]
                else:
                    masked_lm_positions = None

                # MaskedLM layer only project it and normalize (b x s x h)
                token_embeddings_mlm = self._masked_lm_layer(token_embeddings, masked_lm_positions)
                token_logits = tf.matmul(
                    token_embeddings_mlm,
                    tf.cast(self.get_embedding_table(), dtype=tf_utils.get_dtype()),
                    transpose_b=True,
                )
                # token_logits         =  tf.nn.bias_add(token_logits, self._masked_lm_bias)
                token_logits = self._masked_lm_bias(token_logits)
                last_token_logits = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(token_logits)

                result = {
                    "cls_output": cls_output,
                    "token_embeddings": token_embeddings,
                    "token_logits": token_logits,
                    "last_token_logits": last_token_logits,
                }

                if self._return_all_layer_outputs:
                    all_cls_output = []
                    all_token_logits = []
                    for per_layer_token_embeddings in encoder_outputs:
                        per_cls_token_tensor = tf.keras.layers.Lambda(lambda x: tf.squeeze(x[:, 0:1, :], axis=1))(
                            per_layer_token_embeddings
                        )
                        all_cls_output.append(self._pooler_layer(per_cls_token_tensor))

                        # token logits per layer
                        layer_token_embeddings_mlm = self._masked_lm_layer(
                            per_layer_token_embeddings, masked_lm_positions
                        )
                        layer_token_logits = tf.matmul(
                            layer_token_embeddings_mlm,
                            tf.cast(self.get_embedding_table(), dtype=tf_utils.get_dtype()),
                            transpose_b=True,
                        )
                        layer_token_logits = self._masked_lm_bias(layer_token_logits)
                        all_token_logits.append(layer_token_logits)

                    result["all_layer_token_embeddings"] = encoder_outputs
                    result["all_layer_cls_output"] = all_cls_output
                    result["all_layer_token_logits"] = all_token_logits

                return result

        model = MixEncoder(
            config,
            batch_size=batch_size,
            is_training=True,
            use_dropout=True,
            use_masked_lm_positions=True,
            return_all_layer_outputs=False,
        )
        model = model.get_model()

        print("Model inputs", model.input)
        print("Model outputs", model.output)
        return model

    return model_fn


def get_optimizer(learning_rate, train_steps, warmup_steps, optimizer_type):
    def optimizer_fn():
        optimizer, learning_rate_fn = create_optimizer(
            init_lr=learning_rate,
            num_train_steps=train_steps,
            num_warmup_steps=warmup_steps,
            optimizer_type=optimizer_type,
        )

        return optimizer

    return optimizer_fn


def get_loss(loss_type):

    if loss_type and loss_type == 'joint':

        def lm_loss(y_true_dict, y_pred_dict):
            """Joint loss over all layers"""
            loss_dict = {}
            loss_holder = []
            for layer_count, per_layer_output in enumerate(y_pred_dict['all_layer_token_logits']):

                loss = cross_entropy_loss(
                    labels=y_true_dict['masked_lm_labels'],
                    logits=per_layer_output,
                    label_weights=y_true_dict['masked_lm_weights'],
                )
                loss_dict['loss_{}'.format(layer_count + 1)] = loss
                loss_holder.append(loss)
            loss_dict['loss'] = tf.reduce_mean(loss_holder, axis=0)
            return loss_dict

    else:

        def lm_loss(y_true_dict, y_pred_dict):
            """Joint loss over all layers"""
            loss_dict = {}
            loss = cross_entropy_loss(
                labels=y_true_dict['masked_lm_labels'],
                logits=y_pred_dict['token_logits'],
                label_weights=y_true_dict['masked_lm_weights'],
            )
            loss_dict['loss'] = loss
            return loss_dict

    return lm_loss


def get_trainer(device_type, device_address, dtype):

    if device_type == 'tpu':
        trainer = TPUTrainer(tpu_address=device_address, dtype=dtype)
        return trainer
    if device_type == 'gpu':
        pass


def train(cfg):

    # Load tokenizer from tf text SentencePieceTokenizer
    tokenizer_sp = load_tokenizer(cfg)

    # Vocab and tokens
    model_file_path = cfg.tokenizer.model_file_path
    vocab_size = cfg.tokenizer.vocab_size
    cls_id = tokenizer_sp._vocab[cfg.tokenizer.cls_token]
    mask_id = tokenizer_sp._vocab[cfg.tokenizer.mask_token]
    sep_id = tokenizer_sp._vocab[cfg.tokenizer.sep_token]
    unk_id = tokenizer_sp._vocab[cfg.tokenizer.unk_token]
    pad_id = tokenizer_sp._vocab[cfg.tokenizer.pad_token]

    # Data
    max_seq_len = cfg.data.max_seq_len
    max_predictions_per_batch = cfg.data.max_predictions_per_batch
    batch_size = cfg.data.batch_size
    min_sen_len = cfg.data.min_sen_len

    # Train Dataset
    tfrecord_path_list = cfg.data.tfrecord_path_list
    train_dataset = get_dataset(
        tfrecord_path_list,
        max_seq_len,
        max_predictions_per_batch,
        vocab_size,
        cls_id,
        sep_id,
        unk_id,
        pad_id,
        mask_id,
        batch_size,
        min_sen_len,
    )

    # Get Model
    model_fn = get_model(vocab_size, batch_size)

    # Get Optimizer
    optimizer_fn = get_optimizer(
        cfg.model.optimizer.learning_rate,
        cfg.model.optimizer.train_steps,
        cfg.model.optimizer.warmup_steps,
        cfg.model.optimizer.optimizer_type,
    )

    # Get loss
    loss_fn = get_loss(cfg.model.loss.loss_type)
    training_loss_names = None
    if cfg.model.loss.loss_type == 'joint':
        training_loss_names = ['loss_{}'.format(i + 1) for i in range(12)]  # 12 num of hidden layers

    # Model params
    epochs = cfg.model.epochs
    steps_per_epoch = cfg.model.steps_per_epoch
    model_save_dir = cfg.model.model_save_dir
    # callback_steps = cfg.model.callback_steps

    # Set callback
    # To use new sentencepiece model in T5 use like this
    t5_kwargs = {
        'bos_token': '[CLS]',
        'eos_token': '</s>',
        'unk_token': '<unk>',
        'pad_token': '<pad>',
        'mask_token': '[MASK]',
        'vocab_file': '{}'.format(model_file_path),
    }
    tokenizer_hf = T5Tokenizer(**t5_kwargs)
    tokenizer_hf.unique_no_split_tokens = tokenizer_hf.all_special_tokens
    # mlm_callback = MLMCallback(tokenizer_hf)

    # Get trainer
    trainer = get_trainer(cfg.trainer.device_type, cfg.trainer.device_address, cfg.trainer.dtype)

    trainer.run(
        model_fn=model_fn,
        optimizer_fn=optimizer_fn,
        train_dataset=train_dataset,
        train_loss_fn=loss_fn,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        model_checkpoint_dir=model_save_dir,  # gs://tft_free/
        batch_size=batch_size,
        training_loss_names=training_loss_names,
        validation_loss_names=None,
        validation_dataset=None,
        validation_loss_fn=None,
        validation_interval_steps=None,
        steps_per_call=100,
        enable_xla=False,
        callbacks=None,
        callbacks_interval_steps=None,
        overwrite_checkpoint_dir=True,
        max_number_of_models=10,
        model_save_interval_steps=None,
        repeat_dataset=True,
    )


@hydra.main(config_path="config", config_name="train_config")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == "__main__":
    run()
