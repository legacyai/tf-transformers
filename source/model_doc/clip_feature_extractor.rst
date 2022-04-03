..
    Copyright 2020 The HuggingFace Team and TFT Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

CLIP Feature Extractor
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This page includes information about how to use CLIPFeatureExtractorTF with tensorflow-ops.
This feature extractors works in sync with :class:`~tf.data.Dataset` and so is useful for on the fly preprocessing.

.. code-block::

        >>> from tf_transformers.models import  CLIPFeatureExtractorTF
        >>> image_path_list = # List fo image paths
        >>> CLIP_feature_extractor_tf = CLIPFeatureExtractorTF(img_height=224, img_width=224)
        >>> outputs = CLIP_feature_extractor_tf({'image': tf.constant(image_path_list)})


CLIPFeatureExtractorTF
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tf_transformers.models.CLIPFeatureExtractorTF
    :members:
