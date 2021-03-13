import collections

import tensorflow as tf

from tf_transformers.data import TFProcessor
from tf_transformers.data.squad_utils_sp import *
from tf_transformers.data.squad_utils_sp import _compute_softmax, _get_best_indexes
from tf_transformers.utils.tokenization import BasicTokenizer


def extract_from_dict(dict_items, key):
    holder = []
    for item in dict_items:
        holder.append(item[key])
    return holder


class Span_Extraction_Pipeline:
    def __init__(
        self,
        model,
        tokenizer,
        tokenizer_fn,
        SPECIAL_PIECE,
        n_best_size,
        n_best,
        max_answer_length,
        max_seq_length,
        max_query_length,
        doc_stride,
        batch_size=32,
    ):

        self.get_model_fn(model)
        self.tokenizer = tokenizer
        self.tokenizer_fn = tokenizer_fn
        self.SPECIAL_PIECE = SPECIAL_PIECE
        self.n_best_size = n_best_size
        self.n_best = n_best
        self.max_answer_length = max_answer_length

        self.basic_tokenizer = BasicTokenizer(do_lower_case=False)
        self.max_seq_length = max_seq_length
        self.max_query_length = max_query_length
        self.doc_stride = doc_stride
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
        start_logits = []
        end_logits = []
        for batch_inputs in dataset:
            model_outputs = self.model_fn(batch_inputs)
            start_logits.append(model_outputs["start_logits"])
            end_logits.append(model_outputs["end_logits"])

        # Unstack

        start_logits_unstacked = []
        end_logits_unstacked = []

        for batch_logits in start_logits:
            start_logits_unstacked.extend(tf.unstack(batch_logits))
        for batch_logits in end_logits:
            end_logits_unstacked.extend(tf.unstack(batch_logits))

        return start_logits_unstacked, end_logits_unstacked

    def convert_to_features(self, dev_examples):
        """Convert examples to features"""
        qas_id_examples = {ex["qas_id"]: ex for ex in dev_examples}
        dev_examples_cleaned = post_clean_train_squad(dev_examples, self.basic_tokenizer, is_training=False)
        qas_id_info, dev_features = example_to_features_using_fast_sp_alignment_test(
            self.tokenizer,
            dev_examples_cleaned,
            self.max_seq_length,
            self.max_query_length,
            self.doc_stride,
            self.SPECIAL_PIECE,
        )
        return qas_id_info, dev_features, qas_id_examples

    def convert_features_to_dataset(self, dev_features):
        """Feaures to TF dataset"""
        # for TFProcessor
        def local_parser():
            for f in dev_features:
                yield tokenizer_fn(f)

        # Create dataset
        tf_processor = TFProcessor()
        dev_dataset = tf_processor.process(parse_fn=local_parser())
        self.dev_dataset = dev_dataset = tf_processor.auto_batch(dev_dataset, batch_size=self.batch_size)
        return dev_dataset

    def post_process(self, dev_features, qas_id_info, start_logits_unstacked, end_logits_unstacked, qas_id_examples):
        # List of qa_ids per feature
        # List of doc_offset, for shifting when an example gets splitted due to length
        qas_id_list = extract_from_dict(dev_features, "qas_id")
        doc_offset_list = extract_from_dict(dev_features, "doc_offset")

        # Group by qas_id -> predictions , because multiple feature may come from
        # single example :-)

        qas_id_logits = {}
        for i in range(len(qas_id_list)):
            qas_id = qas_id_list[i]
            example = qas_id_info[qas_id]
            feature = dev_features[i]
            assert qas_id == feature["qas_id"]
            if qas_id not in qas_id_logits:
                qas_id_logits[qas_id] = {
                    "tok_to_orig_index": example["tok_to_orig_index"],
                    "aligned_words": example["aligned_words"],
                    "feature_length": [len(feature["input_ids"])],
                    "doc_offset": [doc_offset_list[i]],
                    "passage_start_pos": [feature["input_ids"].index(self.tokenizer.sep_token) + 1],
                    "start_logits": [start_logits_unstacked[i]],
                    "end_logits": [end_logits_unstacked[i]],
                }

            else:
                qas_id_logits[qas_id]["start_logits"].append(start_logits_unstacked[i])
                qas_id_logits[qas_id]["end_logits"].append(end_logits_unstacked[i])
                qas_id_logits[qas_id]["feature_length"].append(len(feature["input_ids"]))
                qas_id_logits[qas_id]["doc_offset"].append(doc_offset_list[i])
                qas_id_logits[qas_id]["passage_start_pos"].append(
                    feature["input_ids"].index(self.tokenizer.sep_token) + 1
                )

        qas_id_answer = {}
        skipped = []
        skipped_null = []
        global_counter = 0
        final_result = {}
        for qas_id in qas_id_logits:

            current_example = qas_id_logits[qas_id]

            _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_log_prob", "end_log_prob"]
            )
            prelim_predictions = []
            example_features = []
            for i in range(len(current_example["start_logits"])):
                f = dev_features[global_counter]
                assert f["qas_id"] == qas_id
                example_features.append(f)
                global_counter += 1
                passage_start_pos = current_example["passage_start_pos"][i]
                feature_length = current_example["feature_length"][i]

                start_log_prob_list = current_example["start_logits"][i].numpy().tolist()[:feature_length]
                end_log_prob_list = current_example["end_logits"][i].numpy().tolist()[:feature_length]
                start_indexes = _get_best_indexes(start_log_prob_list, self.n_best_size)
                end_indexes = _get_best_indexes(end_log_prob_list, self.n_best_size)

                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # We could hypothetically create invalid predictions, e.g., predict
                        # that the start of the span is in the question. We throw out all
                        # invalid predictions.
                        if start_index < passage_start_pos or end_index < passage_start_pos:
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > self.max_answer_length:
                            continue
                        start_log_prob = start_log_prob_list[start_index]
                        end_log_prob = end_log_prob_list[end_index]
                        start_idx = start_index - passage_start_pos
                        end_idx = end_index - passage_start_pos

                        prelim_predictions.append(
                            _PrelimPrediction(
                                feature_index=i,
                                start_index=start_idx,
                                end_index=end_idx,
                                start_log_prob=start_log_prob,
                                end_log_prob=end_log_prob,
                            )
                        )

            prelim_predictions = sorted(
                prelim_predictions, key=lambda x: (x.start_log_prob + x.end_log_prob), reverse=True
            )
            answer_dict = {}
            answer_dict[qas_id] = []
            total_scores = []
            if prelim_predictions:
                for top_n in range(self.n_best):
                    best_index = prelim_predictions[top_n].feature_index
                    aligned_words = current_example["aligned_words"]
                    try:
                        tok_to_orig_index = current_example["tok_to_orig_index"]
                        reverse_start_index_align = tok_to_orig_index[
                            prelim_predictions[top_n].start_index + example_features[best_index]["doc_offset"]
                        ]  # aligned index
                        reverse_end_index_align = tok_to_orig_index[
                            prelim_predictions[top_n].end_index + example_features[best_index]["doc_offset"]
                        ]

                        predicted_words = [
                            w
                            for w in aligned_words[reverse_start_index_align : reverse_end_index_align + 1]
                            if w != self.SPECIAL_PIECE
                        ]
                        predicted_text = " ".join(predicted_words)
                        qas_id_answer[qas_id] = predicted_text
                        total_scores.append(
                            prelim_predictions[top_n].start_log_prob + prelim_predictions[top_n].end_log_prob
                        )
                    except:
                        predicted_text = ""
                        qas_id_answer[qas_id] = ""
                        skipped.append(qas_id)
                        total_scores.append(0.0 + 0.0)
                    answer_dict[qas_id].append({"text": predicted_text})

                _probs = _compute_softmax(total_scores)

                for top_n in range(self.n_best):
                    answer_dict[qas_id][top_n]["probability"] = _probs[top_n]
                final_result[qas_id] = qas_id_examples[qas_id]
            else:
                qas_id_answer[qas_id] = ""
                skipped_null.append(qas_id)
            final_result[qas_id]["answers"] = answer_dict
            return final_result

    def __call__(self, questions, contexts, qas_ids=[]):

        # If qas_id is empty, we assign positions as id
        if qas_ids == []:
            qas_ids = [i for i in range(len(questions))]
        # each question should have a context
        assert len(questions) == len(contexts) == len(qas_ids)

        dev_examples = convert_question_context_to_standard_format(questions, contexts, qas_ids)
        qas_id_info, dev_features, qas_id_examples = self.convert_to_features(dev_examples)
        dev_dataset = self.convert_features_to_dataset(dev_features)
        #self.dev_features = dev_features
        #self.qas_id_info = qas_id_info
        #self.qas_id_examples = qas_id_examples
        
        start_logits_unstacked, end_logits_unstacked = self.run(dev_dataset)
        final_result = self.post_process(dev_features, qas_id_info, start_logits_unstacked, end_logits_unstacked, qas_id_examples)
        return final_result
