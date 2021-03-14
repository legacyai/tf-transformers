from tf_transformers.utils import fast_sp_alignment


def get_tokens_labels(aligned_words, orig_to_new_index, label_tokens, sub_words_mapped, label_pad_token="[PAD]"):
    """
    convert each sub word into labels
    If a word is split into multiple sub words,
    then first sub word is assigned with label and other sub words will be padded
    """
    aligned_labels = [label_pad_token] * len(aligned_words)
    for original_pos, new_pos in enumerate(orig_to_new_index):
        aligned_labels[new_pos] = label_tokens[original_pos]

    flat_tokens = []
    flat_labels = []

    # The first word of the subword token is assigned entity
    # other tokens will be add PAD labels (we will mask it while training)
    assert len(aligned_words) == len(sub_words_mapped) == len(aligned_labels)
    for (_align_word, _align_word, _align_label) in zip(aligned_words, sub_words_mapped, aligned_labels):
        temp_w = []
        for _align_word in _align_word:
            temp_w.append(_align_word)
        temp_l = [label_pad_token] * len(temp_w)
        temp_l[0] = _align_label
        flat_tokens.extend(temp_w)
        flat_labels.extend(temp_l)

    return flat_tokens, flat_labels


def fast_tokenize_and_align_sentence_for_ner(
    tokenizer, sentence, word_tokens, SPECIAL_PIECE, is_training=False, label_tokens=None, label_pad_token=None
):

    """
    align sentence sub words and labels using fast_sp
    """
    subwords = tokenizer.tokenize(sentence)
    orig_to_new_index, aligned_words, sub_words_mapped = fast_sp_alignment(sentence, tokenizer, SPECIAL_PIECE)

    if is_training:
        flat_tokens, flat_labels = get_tokens_labels(
            aligned_words, orig_to_new_index, label_tokens, sub_words_mapped, label_pad_token
        )
        return aligned_words, sub_words_mapped, flat_tokens, flat_labels
    else:
        flat_tokens = [w for sub_words in sub_words_mapped for w in sub_words]
        return aligned_words, sub_words_mapped, flat_tokens, orig_to_new_index
