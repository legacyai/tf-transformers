import collections
import tensorflow as tf
from tf_transformers.data import TFProcessor


def extract_from_dict(dict_items, key):
    holder = []
    for item in dict_items:
        holder.append(item[key])
    return holder


class Token_Classification_Pipeline:
    def __init__(self, model, tokenizer, tokenizer_fn, SPECIAL_PIECE, label_map, max_seq_length, batch_size=32):

        self.get_model_fn(model)
        self.tokenizer = tokenizer
        self.tokenizer_fn = tokenizer_fn
        self.SPECIAL_PIECE = SPECIAL_PIECE
        self.label_map = label_map

        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def get_model_fn(self, model):
        self.model_fn = None
        # keras Model
        if isinstance(model, tf.keras.Model):
            self.model_fn = model
        else:
            # saved model
            if "saved_model" in str(type(model)):
                # Extract signature
                self.model_pb = model.signatures["serving_default"]

                def model_fn(x):
                    return self.model_pb(**x)

                self.model_fn = model_fn
        if self.model_fn is None:
            raise ValueError("Please check the type of your model")

    def run(self, dataset):
        token_logits = []
        for batch_inputs in dataset:
            model_outputs = self.model_fn(batch_inputs)
            token_logits.append(model_outputs["token_logits"])

        # Unstack

        token_logits_unstacked = []
        for batch_logits in token_logits:
            token_logits_unstacked.extend(tf.unstack(batch_logits))

        return token_logits_unstacked

    def convert_to_features(self, dev_examples):
        """Convert examples to features"""
        dev_features = []
        for sentence in dev_examples:
            word_tokens = sentence.split(" ")
            aligned_words, sub_words_mapped, flat_tokens, orig_to_new_index = fast_tokenize_and_align_sentence_for_ner(
                self.tokenizer,
                sentence,
                word_tokens,
                self.SPECIAL_PIECE,
                is_training=False,
                label_tokens=None,
                label_pad_token=None,
            )
            result = {}
            result["word_tokens"] = word_tokens
            result["input_ids"] = flat_tokens[: self.max_seq_length - 2]  # For CLS and SEP
            result["sub_words_mapped"] = sub_words_mapped
            dev_features.append(result)
        return dev_features

    def convert_features_to_dataset(self, dev_features):
        """Feaures to TF dataset"""
        # for TFProcessor
        def local_parser():
            for f in dev_features:
                result = tokenizer_fn(f)
                yield result

        # Create dataset
        tf_processor = TFProcessor()
        dev_dataset = tf_processor.process(parse_fn=local_parser())
        self.dev_dataset = dev_dataset = tf_processor.auto_batch(dev_dataset, batch_size=self.batch_size)
        return dev_dataset

    def post_process(self, sentences, dev_features, token_logits_unstacked):

        final_results = []
        for i in range(len(sentences)):
            slot_logits = token_logits_unstacked[i][1:-1]  # to avoid CLS and SEP
            slot_scores = tf.reduce_max(slot_logits, axis=1)
            slot_ids = tf.argmax(slot_logits, axis=1)
            # Extract only words that we want
            subword_counter = -1
            predicted_ids = []
            predicted_probs = []
            sub_words_mapped = dev_features[i]["sub_words_mapped"]
            for sub_word_list in sub_words_mapped:
                if len(sub_word_list) == 1 and sub_word_list[0] == self.SPECIAL_PIECE:
                    subword_counter += 1
                    continue
                else:
                    predicted_ids.append(slot_ids[subword_counter + 1])
                    predicted_probs.append(slot_scores[subword_counter + 1].numpy())
                    subword_counter += len(sub_word_list)

            predicted_labels = [self.label_map[idx.numpy()] for idx in predicted_ids]
            predicted_probs = tf.nn.softmax(predicted_probs).numpy()

            assert len(dev_features[i]["word_tokens"]) == len(predicted_labels) == len(predicted_probs)
            final_results.append(
                {
                    "sentence": sentences[i],
                    "original_words": dev_features[i]["word_tokens"],
                    "predicted_labels": predicted_labels,
                    "predicted_probs": predicted_probs,
                }
            )
        return final_results

    def __call__(self, sentences):

        dev_features = self.convert_to_features(sentences)
        dev_dataset = self.convert_features_to_dataset(dev_features)
        token_logits_unstacked = self.run(dev_dataset)
        final_result = self.post_process(sentences, dev_features, token_logits_unstacked)
        return final_result
