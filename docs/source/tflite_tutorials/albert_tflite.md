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

+++ {"colab": {"base_uri": "https://localhost:8080/"}, "id": "qjUYDME9kgWS", "outputId": "ee5a458d-cc2e-4ed4-bf32-077fecb227d6"}

# Albert TFlite

```{code-cell} ipython3

```

```{code-cell} ipython3
!pip install tf-transformers

!pip install sentencepiece

!pip install tensorflow-text

!pip install transformers
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: cy6shXrXlL_D
outputId: 74fbc1e8-0861-4d1f-9741-f70b8bf7516a
---
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Supper TF warnings

import tensorflow as tf
print("Tensorflow version", tf.__version__)

from tf_transformers.models import AlbertModel
```

```{code-cell} ipython3
:id: 0HEJnnnFlPxR


```

+++ {"id": "P89IVu5JlREX"}

### Convert a Model to TFlite

The most important thing to notice here is that, if we want to convert a model to ```tflite```, we have to ensure that ```inputs``` to the model are **deterministic**, which means inputs should not be dynamic. We have to fix  **batch_size**, **sequence_length** and other related input constraints depends on the model of interest.

### Load Albert Model

1. Fix the inputs
2. We can always check the ```model``` **inputs** and **output** by using ```model.input``` and ```model.output```.
3. We use ```batch_size=1``` and ```sequence_length=64```.)

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: WV-Ygv4Mlnlo
outputId: 06b4df6a-7304-423d-cb88-4d9ed0759c6e
---
model_name = 'albert-base-v2'
batch_size = 1
sequence_length = 64
model = AlbertModel.from_pretrained(model_name, batch_size=batch_size, sequence_length=sequence_length)
```

```{code-cell} ipython3
:id: edB8Qpo2mgZD


```

+++ {"id": "uL7UDdp9m9ty"}

## Verify Models inputs and outputs

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: SL8wv3mjnCaE
outputId: 9fcce560-b612-43ac-dd93-5df464007aaf
---
print("Model inputs", model.input)
print("Model outputs", model.output)
```

```{code-cell} ipython3
:id: JN6QpCnznFNR


```

```{code-cell} ipython3
:id: k97vFtrSnMGd


```

+++ {"id": "SGBPiXy8nMjQ"}

## Save Model as Serialized Version

We have to save the model using ```model.save```. We use the ```SavedModel``` for converting it to ```tflite```.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: HeDAyaXznZiX
outputId: 648e46e2-714d-4c48-aa7c-e22e9ebe4f4e
---
model.save("{}/saved_model".format(model_name), save_format='tf')
```

```{code-cell} ipython3
:id: DGIccdmJnj_5


```

```{code-cell} ipython3
:id: -HOEyoodnvoU


```

+++ {"id": "XTLzBJAGnv2m"}

## Convert SavedModel to TFlite

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: SqC2HJ10nywO
outputId: 6e80b65e-a2fb-478b-dcff-6b66a66e846d
---
converter = tf.lite.TFLiteConverter.from_saved_model("{}/saved_model".format(model_name)) # path to the SavedModel directory
converter.experimental_new_converter = True

tflite_model = converter.convert()

open("{}/saved_model.tflite".format(model_name), "wb").write(tflite_model)
print("TFlite conversion succesful")
```

```{code-cell} ipython3
:id: eBqTsjUToG7I


```

```{code-cell} ipython3
:id: FdIFext9ta6E


```

+++ {"id": "bDnheGa_tmZq"}

## Load TFlite Model 

```{code-cell} ipython3
:id: vHzESzPAtpHd

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="{}/saved_model.tflite".format(model_name))
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
```

```{code-cell} ipython3
:id: GTYdEgPatzVk


```

```{code-cell} ipython3
:id: C5fUX6ZqxefF


```

+++ {"id": "u1I0ZJ-XxfDg"}

## Assert TFlite Model and Keras Model outputs

After conversion we have to assert the model outputs using
```tflite``` and ```Keras``` model, to ensure proper conversion.

1. Create examples using ```tf.random.uniform```. 
2. Check outputs using both models.
3. Note: We need slightly higher ```rtol``` here to assert.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: QnYr9D5Ot6t4
outputId: 8f6c5f60-7c06-4b6d-aa8b-108b57d11fee
---
# Dummy Examples 
input_ids = tf.random.uniform(minval=0, maxval=100, shape=(batch_size, sequence_length), dtype=tf.int32)
input_mask = tf.ones_like(input_ids)
input_type_ids = tf.zeros_like(input_ids)


# input type ids
interpreter.set_tensor(
    input_details[0]['index'],
    input_type_ids,
)
# input_mask
interpreter.set_tensor(input_details[1]['index'], input_mask)

# input ids
interpreter.set_tensor(
    input_details[2]['index'],
    input_ids,
)

# Invoke inputs
interpreter.invoke()
# Take last output
tflite_output = interpreter.get_tensor(output_details[-1]['index'])

# Keras Model outputs .
model_inputs = {'input_ids': input_ids, 'input_mask': input_mask, 'input_type_ids': input_type_ids}
model_outputs = model(model_inputs)

# We need a slightly higher rtol here to assert :-)
tf.debugging.assert_near(tflite_output, model_outputs['token_logits'], rtol=3.0)
print("Outputs asserted and succesful:  ✅")
```

```{code-cell} ipython3
:id: q8n7FuUgw95j


```
