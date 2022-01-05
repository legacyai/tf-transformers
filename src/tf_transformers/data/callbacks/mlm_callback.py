import tensorflow as tf


# Callbacks
class MLMCallback:
    """Simple MLM Callback to check progress of the training"""

    def __init__(self, tokenizer, validation_sentences=None, top_k=10):
        """Init"""
        self.tokenizer = tokenizer
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
        model = trainer_params['model']
        if "masked_lm_positions" in model.input:
            _use_masked_lm_positions = True
        else:
            _use_masked_lm_positions = False
        inputs_tf = self.get_inputs(_use_masked_lm_positions)
        outputs_tf = model(inputs_tf)

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
