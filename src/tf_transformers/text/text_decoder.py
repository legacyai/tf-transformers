import tensorflow as tf

from tf_transformers.text.text_decoder_encoder_only import TextDecoderEncoderOnly
from tf_transformers.text.text_decoder_seq2seq import TextDecoderSeq2Seq
from tf_transformers.text.text_decoder_seq2seq_serializable import (
    TextDecoderSerializableSeq2Seq,
)
from tf_transformers.text.text_decoder_serializable_encoder_only import (
    TextDecoderEncoderOnlySerializable,
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
        if "decoder" in model.model_config:
            if decoder_start_token_id is None:
                raise ValueError(
                    "You are passing a Seq2Seq model like T5. \
                    Please pass a value for `decoder_start_token_id`, normall BOS while generation."
                )
            return TextDecoderSeq2Seq(model, decoder_start_token_id, input_mask_ids, input_type_ids)
        else:
            return TextDecoderEncoderOnly(model, input_mask_ids, input_type_ids)
    else:
        # Saved model
        if "saved_model" in str(type(model)):
            # input_ids means Encoder Only Model
            if 'input_ids' in model.signatures['serving_default'].structured_input_signature[1]:
                return TextDecoderEncoderOnly(model, input_mask_ids, input_type_ids)
            else:
                # Seq2Seq (EncoderDecoder Model)
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
        if decoder_start_token_id is None:
            raise ValueError(
                "You are passing a Seq2Seq model like T5. Please pass a value for\
                `decoder_start_token_id`, normall BOS while generation."
            )
        return TextDecoderSerializableSeq2Seq(
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
    else:
        return TextDecoderEncoderOnlySerializable(
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
