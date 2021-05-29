import tensorflow as tf
import collections

from tf_transformers.utils import fast_sp_alignment
from tf_transformers.utils.tokenization import _is_whitespace


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def get_answer_start_end_pos(example):
    # Convert text to tokens
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in example["context"]:
        if _is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

    # Get start and end span

    orig_answer_text = example["answer"]
    answer_offset = example["answer_start_pos"]
    answer_length = len(orig_answer_text)
    start_position = char_to_word_offset[answer_offset]
    end_position = char_to_word_offset[answer_offset + answer_length - 1]

    return doc_tokens, start_position, end_position


def get_offset_subtokens(all_doc_tokens, max_tokens_for_doc, doc_stride):
    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])  # pylint: disable=invalid-name
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)
    return doc_spans


def get_shorter_tokens_with_answer_start_end_pos_from_offset(
    doc_spans, Id, all_doc_tokens, tok_start_position, tok_end_position, query_tokens
):

    all_results = []
    for (doc_span_index, doc_span) in enumerate(doc_spans):
        start_position = None
        end_position = None

        # For training, if our document chunk does not contain an annotation
        # we throw it out, since there is nothing to predict.
        doc_start = doc_span.start
        doc_end = doc_span.start + doc_span.length - 1
        out_of_span = False
        if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
            out_of_span = True
        if out_of_span:
            # continue
            start_position = 0
            end_position = 0
            span_is_impossible = True
            result = {}
            result["query_tokens"] = query_tokens
            result["context_tokens"] = all_doc_tokens[doc_start : doc_end + 1]
            result["start_position"] = start_position
            result["end_position"] = end_position
            result["Id"] = example["Id"]
        else:

            doc_offset = 0
            start_position = tok_start_position - doc_start + doc_offset
            end_position = tok_end_position - doc_start + doc_offset

            result = {}
            result["query_tokens"] = query_tokens
            result["context_tokens"] = all_doc_tokens[doc_start : doc_end + 1]
            result["start_position"] = start_position
            result["end_position"] = end_position
            result["Id"] = example["Id"]

            all_results.append(result)
            return all_results

        all_results.append(result)
    return all_results


class Span_Extraction_Processor_Sp:
    def __init__(
        self,
        tokenizer,
        max_seq_length,
        max_query_length,
        doc_stride,
        IGNORE_MAX_WORD_LEN=4096,
        SPECIAL_PIECE="▁",
        return_feature_instances=False,
    ):

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_query_length = max_query_length
        self.doc_stride = doc_stride
        self.SPECIAL_PIECE = SPECIAL_PIECE
        self.IGNORE_MAX_WORD_LEN = IGNORE_MAX_WORD_LEN

        self.failed_train_examples = []

    def call_train_processor(self, example):

        # Do not process anything beyond this length
        if len(example["context"].split()) > self.IGNORE_MAX_WORD_LEN:
            return []
        # Get answer boundaries
        doc_tokens, start_position, end_position = get_answer_start_end_pos(example)

        # Fast SP Alignment + Assertions
        orig_to_new_index, aligned_words, sub_words_mapped = fast_sp_alignment(
            example["context"], self.tokenizer, self.SPECIAL_PIECE
        )
        # means a SPECIAL PIECE has been added
        if len(aligned_words) > len(doc_tokens):
            new_start_position = orig_to_new_index[start_position]
            new_end_position = orig_to_new_index[end_position]
        else:
            # same as before
            new_start_position = start_position
            new_end_position = end_position

        # Assertion
        temp_tokens = aligned_words[new_start_position : new_end_position + 1]
        temp_tokens = [token for token in temp_tokens if token != self.SPECIAL_PIECE]
        try:
            assert temp_tokens == doc_tokens[start_position : end_position + 1]
        except:
            self.failed_train_examples.append(example)
            return []

        # mapping
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(aligned_words):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = sub_words_mapped[i]
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        # Get better positions
        tok_start_position = orig_to_tok_index[new_start_position]
        if new_end_position < len(aligned_words) - 1:
            tok_end_position = orig_to_tok_index[new_end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        # Better Span
        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens,
            tok_start_position,
            tok_end_position,
            tokenizer,
            example["answer"],
        )

        # If question
        if example["question"]:
            query_tokens = self.tokenizer.tokenize(example["question"])[: self.max_query_length]
            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = self.max_seq_length - len(query_tokens) - 3
        else:
            query_tokens = []
            max_tokens_for_doc = self.max_seq_length - 2  # [CLS and SEP]

        doc_spans = get_offset_subtokens(all_doc_tokens, max_tokens_for_doc, self.doc_stride)
        Id = example["Id"]
        processed_examples = get_shorter_tokens_with_answer_start_end_pos_from_offset(
            doc_spans, Id, all_doc_tokens, tok_start_position, tok_end_position, query_tokens
        )

        return processed_examples

    def __call__(self, example, mode):

        if mode == "train":

            return self.call_train_processor(example)

        else:

            return self.call_test_processor(example)
