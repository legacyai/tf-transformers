# coding=utf-8
# Copyright 2021 TF-Transformers Authors.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from tf_transformers.text.decoder_utils import (
    _gather_beams,
    _log_prob_from_logits,
    assign_zeros_to_K_V,
    top_k_logits,
    top_p_logits,
)
from tf_transformers.text.sentencepiece_layer import SentencepieceTokenizer
from tf_transformers.text.text_decoder import TextDecoder, TextDecoderSerializable
from tf_transformers.text.text_decoder_model import TextDecoderModel
from tf_transformers.text.text_decoder_seq2seq import TextDecoderSeq2Seq
from tf_transformers.text.text_decoder_seq2seq_serializable import (
    TextDecoderSerializableSeq2Seq,
)
