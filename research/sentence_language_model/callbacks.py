import pandas as pd
import tensorflow as tf


# Callbacks
class MLMCallback:
    """Simple MLM Callback to check progress of the training"""

    def __init__(self, tokenizer, validation_sentences=None, top_k=10):
        """Init"""
        self.tokenizer = tokenizer
        if validation_sentences is None:
            validation_sentences = [
                'Participant holds a balance in one [MASK] and the birth date falls within the range on \
                    file in Insititution recordkeeping system for that fund.',
                'The investment risk of each target date strategy changes over time as \
                    the fund’s asset [MASK] changes.',
                'The [MASK] equity scatter plot slide is only available for plans with 100 or more DIY participants.',
                '[MASK] is the capital of U.S.A.',
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

        # Create a pandas dataframe
        df = pd.DataFrame(self.validation_sentences)
        df.columns = ['text']

        predicted_words = []
        if "all_layer_token_logits" in model.output:
            masked_indexes = tf.where(tf.equal(inputs_tf['input_ids'], self.tokenizer.mask_token_id))
            for layer_count, layer_logits in enumerate(outputs_tf['all_layer_token_logits']):

                gathered_logits = tf.gather_nd(layer_logits, masked_indexes)
                probs, indexes = tf.nn.top_k(gathered_logits, self.top_k)
                predicted_words = [text.split() for text in self.tokenizer.batch_decode(indexes)]
                df['predicted_words_layer_{}'.format(layer_count + 1)] = predicted_words
        else:
            masked_indexes = tf.where(tf.equal(inputs_tf['input_ids'], self.tokenizer.mask_token_id))
            gathered_logits = tf.gather_nd(outputs_tf['token_logits'], masked_indexes)
            probs, indexes = tf.nn.top_k(gathered_logits, self.top_k)
            predicted_words = [text.split() for text in self.tokenizer.batch_decode(indexes)]

        df['predicted_words'] = predicted_words
        wandb = trainer_params['wandb']
        global_step = trainer_params['global_step']
        # Log to wandb as a table
        if wandb:
            wandb.log({"mlm_table_step_{}".format(global_step): wandb.Table(dataframe=df)}, step=global_step)
            print(df)
        else:
            print(df)
