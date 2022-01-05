import tensorflow as tf
from transformers import T5Tokenizer

from tf_transformers.models import BertModel


# Callbacks
class MLMCallback:
    """Simple MLM Callback to check progress of the training"""

    def __init__(self, model, tokenizer, validation_sentences=None, top_k=10):
        """Init"""
        self.tokenizer = tokenizer
        self.model = model
        if validation_sentences is None:
            validation_sentences = [
                'Read the rest of this [MASK] to understand things in more detail.',
                'I want to buy the [MASK] because it is so cheap.',
                'The [MASK] was amazing.',
                'Sachin Tendulkar is one of the [MASK] palyers in the world.',
                '[MASK] is the capital of France.',
                'Machine Learning requires [MASK]',
                'He is working as a [MASK]',
                'She is working as a [MASK]',
            ]
            self.validation_sentences = validation_sentences
        else:
            self.validation_sentences = validation_sentences
        self.top_k = top_k

    def get_inputs(self, _use_masked_lm_positions):
        """Text to features"""
        inputs = self.tokenizer(self.validation_sentences, padding=True, return_tensors="tf")
        inputs_tf = {}
        inputs_tf["input_ids"] = inputs["input_ids"]
        inputs_tf["input_mask"] = inputs["attention_mask"]
        inputs_tf["input_type_ids"] = tf.zeros_like(inputs_tf["input_ids"])
        if _use_masked_lm_positions:
            seq_length = tf.shape(inputs_tf['input_ids'])[1]
            inputs_tf['masked_lm_positions'] = tf.zeros_like(inputs_tf["input_ids"]) + tf.range(seq_length)

        return inputs_tf

    def __call__(self, trainer_params):
        """Main Call"""
        if "masked_lm_positions" in self.model.input:
            _use_masked_lm_positions = True
        else:
            _use_masked_lm_positions = False
        inputs_tf = self.get_inputs(_use_masked_lm_positions)
        outputs_tf = self.model(inputs_tf)

        if "all_layer_token_logits" in model.output:
            # Get masked positions from each sentence
            masked_positions = tf.argmax(tf.equal(inputs_tf["input_ids"], self.tokenizer.mask_token_id), axis=1)
            for layer_count, layer_logits in enumerate(outputs_tf['all_layer_token_logits']):
                print("Layer {}".format(layer_count + 1))
                print("-------------------------------------------------------------------")
                for i, logits in enumerate(layer_logits):
                    mask_token_logits = logits[masked_positions[i]]
                    # 0 for probs and 1 for indexes from tf.nn.top_k
                    top_words = self.tokenizer.decode(tf.nn.top_k(mask_token_logits, k=self.top_k)[1].numpy())
                    print("Input ----> {}".format(self.validation_sentences[i]))
                    print("Predicted words ----> {}".format(top_words.split()))
                    print()
        else:
            # Get masked positions from each sentence
            masked_positions = tf.argmax(tf.equal(inputs_tf["input_ids"], self.tokenizer.mask_token_id), axis=1)
            for i, logits in enumerate(outputs_tf['token_logits']):
                mask_token_logits = logits[masked_positions[i]]
                # 0 for probs and 1 for indexes from tf.nn.top_k
                top_words = self.tokenizer.decode(tf.nn.top_k(mask_token_logits, k=self.top_k)[1].numpy())
                print("Input ----> {}".format(self.validation_sentences[i]))
                print("Predicted words ----> {}".format(top_words.split()))
                print()


model_file_path = '/home/sidhu/Datasets/data/t5_extended_vocab/new_spiece.model'
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

# Load model


model_path = '/home/sidhu/Projects/bert_mix_model'
vocab_size = 32002
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

model = BertModel.from_config(config, return_all_layer_outputs=True)
model.load_checkpoint(model_path)

mlm_callback = MLMCallback(model, tokenizer_hf)
mlm_callback({})

######## Text Generation

model = BertModel.from_config(config, mask_mode='causal', return_all_layer_outputs=False)
model.load_checkpoint(model_path)

text = "[CLS] Onam is celebrated "
text = "[CLS] In 2002, Musk founded SpaceX, an aerospace manufacturer and space transport services"

# text = "[CLS] India is a country with amazing culture and "
inputs_tf = tokenizer_hf(text, add_special_tokens=False, return_tensors="tf")
inputs = {}
inputs["input_ids"] = inputs_tf["input_ids"]
inputs['input_type_ids'] = tf.zeros_like(inputs['input_ids'])

predictions_non_auto_regressive = []
predictions_prob_non_auto_regressive = []

for _i in range(128):
    outputs = model(inputs)
    predicted_ids = tf.cast(tf.expand_dims(tf.argmax(outputs["last_token_logits"], axis=1), 1), tf.int32)
    inputs["input_ids"] = tf.concat([inputs["input_ids"], predicted_ids], axis=1)
    inputs["input_type_ids"] = tf.concat([inputs["input_type_ids"], [[0]]], axis=1)
    predictions_non_auto_regressive.append(predicted_ids)
    predictions_prob_non_auto_regressive.append(tf.expand_dims(tf.reduce_max(outputs["last_token_logits"], axis=1), 1))

predictions_non_auto_regressive = tf.concat(predictions_non_auto_regressive, axis=1)
predictions_prob_non_auto_regressive = tf.concat(predictions_prob_non_auto_regressive, axis=1)
tokenizer_hf.decode(inputs['input_ids'].numpy()[0])

##### Prefix

model = BertModel.from_config(config, mask_mode='prefix', return_all_layer_outputs=False)
model.load_checkpoint(model_path)

text = "[CLS] Sachin Tendulkar is one of the finest "
text = "[CLS] Chembai music festival is one of the finest "
text = "[CLS] Onam is the most important festival "
text = "[CLS] In 2002, Musk founded SpaceX, an aerospace manufacturer and space transport services"


# text = "[CLS] In 2002, Musk founded SpaceX, an aerospace manufacturer and space transport services [MASK] </s>"
inputs_tf = tokenizer_hf(text, add_special_tokens=False, return_tensors="tf")
inputs = {}
inputs["input_ids"] = inputs_tf["input_ids"]
inputs['input_mask'] = tf.ones_like(inputs['input_ids'])
inputs['input_type_ids'] = tf.zeros_like(inputs['input_ids'])

predictions_non_auto_regressive = []
predictions_prob_non_auto_regressive = []

for _i in range(128):
    outputs = model(inputs)
    predicted_ids = tf.cast(tf.expand_dims(tf.argmax(outputs["last_token_logits"], axis=1), 1), tf.int32)
    inputs["input_ids"] = tf.concat([inputs["input_ids"], predicted_ids], axis=1)
    inputs["input_mask"] = tf.concat([inputs["input_mask"], [[1]]], axis=1)
    inputs["input_type_ids"] = tf.concat([inputs["input_type_ids"], [[0]]], axis=1)
    predictions_non_auto_regressive.append(predicted_ids)
    predictions_prob_non_auto_regressive.append(tf.expand_dims(tf.reduce_max(outputs["last_token_logits"], axis=1), 1))

predictions_non_auto_regressive = tf.concat(predictions_non_auto_regressive, axis=1)
predictions_prob_non_auto_regressive = tf.concat(predictions_prob_non_auto_regressive, axis=1)
tokenizer_hf.decode(inputs['input_ids'].numpy()[0])
