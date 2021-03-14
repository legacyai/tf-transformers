def fast_sp_split(sentence, tokenizer, SPECIAL_PIECE):
    """Spit and align sub words. It might not be perfect

    Args:
        sentence : The input sentence
        tokenizer : Huggingface tokenizer
        SPECIAL_PIECE : SPECIAL PIECE symbol

    Returns:
        original_words : List of words
        sub_words_mapped : List of list of sub words
    """
    original_words = sentence.split()
    subwords = tokenizer.tokenize(sentence)

    # Convert text into main_words (list of list of subwords per word)
    sub_words_mapped = []
    temp_tokens = []
    for tok in subwords:
        if tok == SPECIAL_PIECE:
            if temp_tokens:
                sub_words_mapped.append(temp_tokens)
                temp_tokens = []
            sub_words_mapped.append([tok])

        else:
            if tok.startswith(SPECIAL_PIECE):
                if temp_tokens:
                    sub_words_mapped.append(temp_tokens)
                    temp_tokens = []
                temp_tokens.append(tok)
            else:
                temp_tokens.append(tok)

    if temp_tokens:
        sub_words_mapped.append(temp_tokens)
    return original_words, sub_words_mapped


def realign_words(word_tokens, sub_words_mapped, SPECIAL_PIECE):
    """Add special tokens to make word_tokens in alignment with extra SPECIAL PIECE in sub_words_mapped

    Args:
        word_tokens : List of words
        sub_words_mapped : List of list of sub words
    Returns:
        orig_to_new_index: old to new index
        aligned_words: List of words after adding SPECIAL PIECE if possible
    """
    # this loop is used to accout for extra SPECIAL_PIECE
    # if any
    special_counter = 0
    aligned_words = []
    orig_to_new_index = []
    for index, _sub_word in enumerate(sub_words_mapped):
        # this is some extra SPECIAL PIECE character
        # add it to original word
        if len(_sub_word) == 1 and _sub_word[0] == SPECIAL_PIECE:
            special_counter += 1
            aligned_words.append(_sub_word[0])
        else:
            pos = index - special_counter
            aligned_words.append(word_tokens[pos])
            orig_to_new_index.append(index)  # whenever original words comes, we need old-new mapping
    return orig_to_new_index, aligned_words


def fast_sp_alignment(sentence, tokenizer, SPECIAL_PIECE):
    """Fast Sentence Piece Alignment

    A sentence will be split into tokens based on whitespace, then tokenize using
    sentence piece tokenizer (GPT2, Albert, etc).

    Args:
        sentence str: The input sentence to be tokenized
        tokenizer HF tokenizer: The tokenizer
        SPECIAL_PIECE str: SPECIAL PIECE symbol

    Returns:
        orig_to_new_index list: Old to new index mapping alignment
        aligned_words list of string: Aligns words (by adding SPECIAL_PIECE) if required
        sub_words_mapped : list of list of subwords
    """

    original_words, sub_words_mapped = fast_sp_split(sentence, tokenizer, SPECIAL_PIECE)
    # If they are of different length mostly due to
    # extra SPECIAL_PIECE or unicode characters
    if len(original_words) != len(sub_words_mapped):
        # Try to re-align if possible
        try:
            orig_to_new_index, aligned_words = realign_words(original_words, sub_words_mapped, SPECIAL_PIECE)
        except:
            # if re-align fails, then tokenize like word-piece tokenizer
            # biut, using sentence piece
            aligned_words = original_words
            sub_words_mapped = [tokenizer.tokenize(word) for word in original_words]
            orig_to_new_index = range(len(original_words))

        assert len(aligned_words) == len(sub_words_mapped)
        return orig_to_new_index, aligned_words, sub_words_mapped
    else:
        # If this mapping fails, logic fails
        orig_to_new_index = range(len(original_words))
        return orig_to_new_index, original_words, sub_words_mapped
