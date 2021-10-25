tokenizer_layer = AlbertTokenizerTFText.from_pretrained(MODEL_NAME, 
                                                        pack_model_inputs=False) # Keras Layer

def map_text_and_label(item):
    """Separate text and labels from delimiter"""
    text = item[0]
    label = item[1]
    label = tf.cast(tf.equal(tf.strings.strip(label),b'1'),tf.int32)
    return {"text": text, "labels": [label]}

def mlm_fn():
    max_seq_len = 128
    max_predictions_per_batch = 20

    cls_token_id = tokenizer_layer.cls_token_id 
    sep_token_id = tokenizer_layer.sep_token_id
    unk_token_id = tokenizer_layer.unk_token_id
    pad_token_id = tokenizer_layer.pad_token_id
    mask_token_id = 4 # tokenizer_layer.mask_token_id
    vocab_size    = tokenizer_layer.vocab_size


    # Random Selector
    random_selector = tf_text.RandomItemSelector(
        max_selections_per_batch=max_predictions_per_batch,
        selection_rate=0.1,
        unselectable_ids=[cls_token_id, sep_token_id, unk_token_id, pad_token_id],
    )

    # Mask Value chooser (Encapsulates the BERT MLM token selection logic)
    mask_values_chooser = tf_text.MaskValuesChooser(vocab_size, mask_token_id, 0.8)
    
    def dynamic_mlm(example):
        # Inputs
        text = example['text']
        if tokenizer_layer._lower_case:
            text = tf_text.case_fold_utf8(text)
            
        inputs = {'text': tf.strings.split(text)}
        segments = tokenizer_layer(inputs)

        # Find the index where max_seq_len is valid
        max_seq_index = tf.where(segments.row_splits < max_seq_len-2)[-1][0]
        segments = segments[:max_seq_index]
        # Flatten and add CLS , SEP
        segments_flattened = segments.merge_dims(-2, 1)
        segments_combined = tf.concat([[cls_token_id], segments_flattened, [sep_token_id]], axis=0)
        # We have to move original row splits to acoomoadate 2 extra tokens added later, CLS and SEP
        row_splits = tf.concat([[0], segments.row_splits + 1, [segments.row_splits[-1]+2]], axis=0)
        segments_combined = tf.RaggedTensor.from_row_splits(segments_combined, row_splits)
        # Apply dynamic masking, with expand_dims on the input batch
        # If expand_dims is not there, whole word masking fails
        masked_token_ids, masked_pos, masked_lm_ids = tf_text.mask_language_model(
            tf.expand_dims(segments_combined, axis=0), item_selector=random_selector, mask_values_chooser=mask_values_chooser
        )
        
        # Prepare and pad combined segment inputs
        input_word_ids, input_mask = tf_text.pad_model_inputs(masked_token_ids, max_seq_length=max_seq_len)
        input_type_ids = tf.zeros_like(input_word_ids)

        # Prepare and pad masking task inputs
        # Masked lm weights will mask the weights
        masked_lm_positions, masked_lm_weights = tf_text.pad_model_inputs(
            masked_pos, max_seq_length=max_predictions_per_batch
        )
        masked_lm_ids, _ = tf_text.pad_model_inputs(masked_lm_ids, max_seq_length=max_predictions_per_batch)

        # Work around broken shape inference.
        output_shape = tf.stack([
            masked_token_ids.nrows(out_type=tf.int32),  # batch_size
            tf.cast(max_seq_len, dtype=tf.int32)])
        output_shape_masked_tokens = tf.stack([
            masked_pos.nrows(out_type=tf.int32),  # batch_size
            tf.cast(max_predictions_per_batch, dtype=tf.int32)])
        def _reshape(t):
            return tf.reshape(t, output_shape)
        def _reshape_masked(t):
            return tf.reshape(t, output_shape_masked_tokens)

        input_word_ids = _reshape(input_word_ids)
        input_type_ids = _reshape(input_type_ids)
        input_mask = _reshape(input_mask)
        
        masked_lm_positions = _reshape_masked(masked_lm_positions)
        masked_lm_ids = _reshape_masked(masked_lm_ids)
        masked_lm_weights = _reshape_masked(masked_lm_weights)
        
        inputs = {}
        inputs['input_ids'] = tf.squeeze(input_word_ids, axis=0)
        inputs['input_type_ids'] = tf.squeeze(input_type_ids, axis=0)
        inputs['input_mask'] = tf.squeeze(input_mask, axis=0)
        inputs['masked_lm_positions'] = tf.squeeze(masked_lm_positions, axis=0)

        labels = {}
        labels['masked_lm_labels'] = tf.squeeze(masked_lm_ids, axis=0)
        labels['masked_lm_weights'] = tf.squeeze(masked_lm_weights, axis=0)  # Mask

        return (inputs, labels)
    
    return dynamic_mlm
    
def split_text(item):
    sentences = tf.strings.split(item['text'], '.')
    item['sentences'] = sentences
    return item

def filter_empty_string(item):
    sentences = item['sentences']
    valid_string_indexes = tf.squeeze(tf.where(tf.not_equal(tf.strings.length(item['sentences']), 0)), axis=1)
    sentences = tf.gather(sentences, valid_string_indexes)
    item['sentences'] = sentences
    return item

def filter_single_sentence(item):
    """If number of sentences after split is 1, ignore"""
    number_of_sentences = tf.shape(item['sentences'])[0]
    return tf.greater(number_of_sentences, tf.constant(1))

# Train Dataset
ds_train = tf.data.TextLineDataset([os.path.join(DATA_DIR, "imdb_train.txt")])

train_dataset = ds_train.map(lambda x: tf.strings.split(x, sep="__||__", maxsplit=2), 
                            num_parallel_calls=tf.data.AUTOTUNE
                            )
train_dataset = train_dataset.map(map_text_and_label, num_parallel_calls=tf.data.AUTOTUNE)

train_dataset = train_dataset.map(split_text, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.map(filter_empty_string, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.filter(filter_single_sentence)

prefix_lm_map = do_prefix_masking(max_seq_len, add_cls_sep=True, return_type_ids=True)
train_dataset2 = train_dataset.map(prefix_lm_map, num_parallel_calls=tf.data.AUTOTUNE)
for item in train_dataset:
    print(item)
    break

for i,item2 in enumerate(train_dataset2):
    print(item2[0]['prefix_mask_index'], tf.reduce_sum(item2[0]['input_mask']))
    if tf.reduce_sum(item2[0]['input_mask']) <= 60:
        break

def do_prefix_masking(max_seq_len, add_cls_sep=False, return_type_ids=False):
    
    
    def prefix_map_fn(item):
        input_ids = tokenizer_layer({'text': item['sentences']})
        # We take random position between 1 and len(sentences)//2
        mid_index = tf.shape(input_ids.flat_values)[0]//2
        prefix_mask_index = tf.random.uniform(minval=1, maxval=mid_index+1, shape=(), dtype=tf.int32)
        print("prefix mask index", prefix_mask_index)
        # We split it to 2 parts left and right
        # left we mask by 1 and right we mask by 0
        input_ids_first_portion  = input_ids[:prefix_mask_index]
        input_ids_second_portion = input_ids[prefix_mask_index:]
        print("input_ids_first_portion", input_ids_first_portion)
        print("input_ids_second_portion", input_ids_second_portion)
        
        # Split and join
        input_mask_first_portion  = tf.ones_like(input_ids_first_portion)
        input_mask_second_portion = tf.zeros_like(input_ids_second_portion)
        input_mask = tf.concat([input_mask_first_portion, input_mask_second_portion], axis=0)
        print("input_mask", input_mask)
        # Pad inputs
        input_ids_ragged  = tf.RaggedTensor.from_tensor(tf.expand_dims(input_ids.merge_dims(-2, 1), 0))
        input_mask_ragged = tf.RaggedTensor.from_tensor(tf.expand_dims(input_mask.merge_dims(-2, 1), 0))
        # Trim inputs (+1 is because for Causal LM we shift inputs and labels)
        if add_cls_sep:
            input_ids_ragged = input_ids_ragged[:, :max_seq_len+1-2]
            input_mask_ragged = input_mask_ragged[:, :max_seq_len+1-2]
            
            input_ids_ragged  = tf.concat([[[cls_token_id]], input_ids_ragged, [[sep_token_id]]], axis=1)
            input_mask_ragged = tf.concat([[[1]], input_mask_ragged, [[1]]], axis=1)

        else:
            input_ids_ragged = input_ids_ragged[:, :max_seq_len+1]
            input_mask_ragged = input_mask_ragged[:, :max_seq_len+1]
            
        input_word_ids, _ = tf_text.pad_model_inputs(input_ids_ragged, max_seq_length=max_seq_len+1)
        input_mask, _ = tf_text.pad_model_inputs(input_mask_ragged, max_seq_length=max_seq_len+1)

        # Squeeze and trim based on max_seq_len
        input_word_ids = tf.squeeze(input_word_ids, axis=0)
        input_mask     = tf.squeeze(input_mask, axis=0)
        
        # Shift positions
        lm_labels = input_word_ids[1:]
        input_word_ids = input_word_ids[:-1]
        input_mask     = input_mask[:-1]
        # Opposite of input_mask
        lm_label_weights = tf.cast(tf.not_equal(input_mask, 1), tf.int32)
        
        inputs = {}
        inputs['input_ids'] = input_word_ids
        inputs['input_mask'] = input_mask
        inputs['prefix_mask_index'] = prefix_mask_index
        if return_type_ids:
            input_type_ids = tf.zeros_like(input_word_ids)
            inputs['input_type_ids'] = input_type_ids
        
        labels = {}
        labels['lm_labels'] = lm_labels
        labels['lm_weights'] = lm_label_weights
        
        return inputs, labels            
        
    return prefix_map_fn

batch_shapes = []
seq_shapes = []
for (batch_inputs, batch_labels) in tqdm.tqdm(train_dataset):
    batch_shapes.append(batch_inputs['input_ids'].shape[0])
    seq_shapes.append(batch_inputs['input_ids'].shape[1])