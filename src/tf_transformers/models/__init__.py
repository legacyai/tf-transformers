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
from tf_transformers.models.albert import (
    AlbertConfig,
    AlbertEncoder,
    AlbertModel,
    AlbertTokenizerLayer,
    AlbertTokenizerTFText,
)
from tf_transformers.models.bart import BartConfig, BartEncoder, BartModel
from tf_transformers.models.bert import BertConfig, BertEncoder, BertModel
from tf_transformers.models.bigbird import BigBirdRobertaTokenizerTFText
from tf_transformers.models.encoder_decoder import EncoderDecoder
from tf_transformers.models.gpt2 import GPT2Config, GPT2Encoder, GPT2Model
from tf_transformers.models.mt5 import MT5Config, MT5Encoder, MT5Model
from tf_transformers.models.roberta import RobertaConfig, RobertaEncoder, RobertaModel
from tf_transformers.models.t5 import T5Config, T5Encoder, T5Model, T5TokenizerTFText
from tf_transformers.models.tasks import (
    Classification_Model,
    MaskedLMModel,
    Similarity_Model,
    Span_Selection_Model,
    Token_Classification_Model,
)
from tf_transformers.models.vit import ViTEncoder, ViTModel
