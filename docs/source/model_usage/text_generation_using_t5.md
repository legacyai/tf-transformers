---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Text Generation using T5

* This tutorial is intended to provide, a familiarity in how to use ```T5``` for text-generation tasks.
* No training is involved in this.

```{code-cell} ipython3
!pip install tf-transformers

!pip install transformers
```

```{code-cell} ipython3

```

```{code-cell} ipython3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Supper TF warnings

import tensorflow as tf
print("Tensorflow version", tf.__version__)
print("Devices", tf.config.list_physical_devices())

from tf_transformers.models import T5Model, T5TokenizerTFText
from tf_transformers.core import TextGenerationChainer
from tf_transformers.text import TextDecoder, TextDecoderSerializable
```

```{code-cell} ipython3

```

### Load T5 Model 

* 1. Note `use_auto_regressive=True`, argument. This is required for any models to enable text-generation.

```{code-cell} ipython3
model_name = 't5-small'

tokenizer = T5TokenizerTFText.from_pretrained(model_name, dynamic_padding=True, truncate=True, max_length=256)
model = T5Model.from_pretrained(model_name, use_auto_regressive=True)
```

```{code-cell} ipython3

```

### Serialize and load

* The most recommended way of using a Tensorflow model is to load it after serializing.
* The speedup, especially for text generation is up to 50x times.

```{code-cell} ipython3
# Save as serialized
model_dir = 'MODELS/t5'
model.save_transformers_serialized(model_dir)

# Load
loaded = tf.saved_model.load(model_dir)
```

```{code-cell} ipython3

```

### Text-Generation

* . We can pass ```tf.keras.Model``` also to ```TextDecoder```, but ```SavedModel``` this is recommended

```{code-cell} ipython3
decoder = TextDecoder(model=loaded)
```

### Greedy Decoding

```{code-cell} ipython3
texts = ['translate English to German: The house is wonderful and we wish to be here :)', 
         'translate English to French: She is beautiful']

inputs = tokenizer({'text': tf.constant(texts)})

predictions = decoder.decode(inputs, 
                             mode='greedy', 
                             max_iterations=64, 
                             eos_id=tokenizer.eos_token_id)
print(tokenizer._tokenizer.detokenize(tf.squeeze(predictions['predicted_ids'], axis=1)))
```

```{code-cell} ipython3

```

### Beam Decoding

```{code-cell} ipython3
predictions = decoder.decode(inputs, 
                             mode='beam',
                             num_beams=3,
                             max_iterations=64,
                             eos_id=tokenizer.eos_token_id)
print(tokenizer._tokenizer.detokenize(predictions['predicted_ids'][:, 0, :]))
```

### Top K Nucleus Sampling

```{code-cell} ipython3
predictions = decoder.decode(inputs, 
                             mode='top_k_top_p',
                             top_k=50,
                             top_p=0.7,
                             num_return_sequences=3,
                             max_iterations=64,
                             eos_id=tokenizer.eos_token_id)
print(tokenizer._tokenizer.detokenize(predictions['predicted_ids'][:, 0, :]))
```

```{code-cell} ipython3

```

### Advanced Serialization (include preprocessing + Decoding Together)

* What if we can bundle all this into a single model and serialize it ?

```{code-cell} ipython3
model_dir = 'MODELS/t5_serialized/'
# Load Auto Regressive Version
model = T5Model.from_pretrained(model_name=model_name, use_auto_regressive=True)
# Assume we are doing beam decoding
text_generation_kwargs = {'mode': 'beam', 
                         'num_beams': 3,
                          'max_iterations': 32,
                          'eos_id': tokenizer.eos_token_id
                         }
# TextDecoderSerializable - makes decoding serializable
decoder = TextDecoderSerializable(model=model, **text_generation_kwargs)
# TextGenerationChainer - joins tokenizer + TextDecoderSerializable
model_fully_serialized = TextGenerationChainer(tokenizer.get_model(), decoder)
model_fully_serialized = model_fully_serialized.get_model()
# Save as saved_model
model_fully_serialized.save_serialized(model_dir, overwrite=True)
```

```{code-cell} ipython3

```

### Load Advanced Model and Generate text

* How nice is it? All done by model, no overhead of anything (tokenization, decoding, generating)

* 1. TextDecoderSerializable - very advances serializable decoder written in pure tensorflow ops
* 2. TextGenerationChainer -  very simple ```tf.keras.layers.Layer``` wrapper.

```{code-cell} ipython3
loaded = tf.saved_model.load(model_dir)
model = loaded.signatures['serving_default']
```

```{code-cell} ipython3
texts = ['translate English to German: The house is wonderful and we wish to be here :)', 
         'translate English to French: She is beautiful']

predictions = model(**{'text': tf.constant(texts)})
```

```{code-cell} ipython3
print(predictions['decoded_text'])
```

```{code-cell} ipython3

```
