import tensorflow as tf

from tf_transformers.text.text_decoder_encoder_only import TextDecoderEncoderOnly
from tf_transformers.text.text_decoder_encoder_only_serializable import (
    TextDecoderEncoderOnlySerializable,
)
from tf_transformers.text.text_decoder_seq2seq import TextDecoderSeq2Seq
from tf_transformers.text.text_decoder_seq2seq_serializable import (
    TextDecoderSerializableSeq2Seq,
)


def TextDecoder(
    model,
    decoder_start_token_id=None,
    input_mask_ids=-1,
    input_type_ids=-1,
):

    """
    Function to route the model for text generation tasks
    """

    # Seq2Seq model
    if isinstance(model, tf.keras.Model):
        # Seq2Seq Model (EncoderDecoder Model)
        if "decoder" in model.model_config:
            # If provided use it directly
            # Sometimes if decoder_start_token_id = 0, if decoder_start_token_id fails.
            if isinstance(decoder_start_token_id, int):
                return TextDecoderSeq2Seq(model, decoder_start_token_id, input_mask_ids, input_type_ids)
            if "decoder_start_token_id" in model.model_config['decoder']:
                decoder_start_token_id = model.model_config['decoder']['decoder_start_token_id']
            if decoder_start_token_id is None:
                raise ValueError(
                    "You are passing a Seq2Seq model like T5. \
                    Please pass a value for `decoder_start_token_id`, normall BOS while generation."
                )
            return TextDecoderSeq2Seq(model, decoder_start_token_id, input_mask_ids, input_type_ids)
        else:
            # Encoder only Model
            return TextDecoderEncoderOnly(model, input_mask_ids, input_type_ids)
    else:
        # Saved model
        if "saved_model" in str(type(model)):
            # input_ids means Encoder Only Model
            if 'input_ids' in model.signatures['serving_default'].structured_input_signature[1]:
                return TextDecoderEncoderOnly(model, input_mask_ids, input_type_ids)
            else:
                if isinstance(decoder_start_token_id, int):
                    return TextDecoderSeq2Seq(model, decoder_start_token_id, input_mask_ids, input_type_ids)
                # Seq2Seq (EncoderDecoder Model)
                # Get it from saved_model config
                if "decoder_start_token_id" in model.config:
                    decoder_start_token_id = model.config["decoder_start_token_id"].numpy()
                if decoder_start_token_id is None:
                    raise ValueError(
                        "You are passing a Seq2Seq model like T5. \
                        Please pass a value for `decoder_start_token_id`, normall BOS while generation."
                    )
                return TextDecoderSeq2Seq(model, decoder_start_token_id, input_mask_ids, input_type_ids)


def TextDecoderSerializable(
    model,
    mode,
    decoder_start_token_id=None,
    max_iterations=None,
    batch_size=None,
    max_sequence_length=None,
    num_beams=1,
    eos_id=-100,
    temperature=1.0,
    alpha=0.0,
    do_sample=False,
    top_k=0,
    top_p=0,
    num_return_sequences=1,
    input_type_ids=-1,
    input_mask_ids=1,
):

    """
    Function to route the model for text generation tasks
    """

    # Seq2Seq model
    assert isinstance(model, tf.keras.Model)
    if "decoder" in model.model_config:
        decoder_model = TextDecoderSerializableSeq2Seq(
            model=model,
            mode=mode,
            decoder_start_token_id=decoder_start_token_id,
            max_iterations=max_iterations,
            batch_size=batch_size,
            max_sequence_length=max_sequence_length,
            temperature=temperature,
            alpha=alpha,
            num_beams=num_beams,
            eos_id=eos_id,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            input_type_ids=input_type_ids,
            input_mask_ids=input_mask_ids,
        )
        decoder_model = decoder_model.get_model()
        return decoder_model
    else:
        decoder_model = TextDecoderEncoderOnlySerializable(
            model=model,
            mode=mode,
            max_iterations=max_iterations,
            batch_size=batch_size,
            max_sequence_length=max_sequence_length,
            temperature=temperature,
            alpha=alpha,
            num_beams=num_beams,
            eos_id=eos_id,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            input_type_ids=input_type_ids,
            input_mask_ids=input_mask_ids,
        )
        decoder_model = decoder_model.get_model()
        return decoder_model
