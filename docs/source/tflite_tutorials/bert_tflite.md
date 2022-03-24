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

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: qjUYDME9kgWS
outputId: 89cd7257-9656-4743-f458-09fd70a437c6
---
!pip install tf-transformers

!pip install sentencepiece

!pip install tensorflow-text

!pip install transformers
```

```{code-cell} ipython3
:id: CV0Bh-eFlEot


```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: cy6shXrXlL_D
outputId: da3d5ff0-96f1-4912-f115-9310fc4cc68d
---
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Supper TF warnings

import tensorflow as tf
print("Tensorflow version", tf.__version__)

from tf_transformers.models import BertModel
```

```{code-cell} ipython3
:id: 0HEJnnnFlPxR


```

+++ {"id": "P89IVu5JlREX"}

## Convert a Model to TFlite

The most important thing to notice here is that, if we want to convert a model to ```tflite```, we have to ensure that ```inputs``` to the model are **deterministic**, which means inputs should not be dynamic. We have to fix  **batch_size**, **sequence_length** and other related input constraints depends on the model of interest.

### Load Bert Model

1. Fix the inputs
2. We can always check the ```model``` **inputs** and **output** by using ```model.input``` and ```model.output```.
3. We use ```batch_size=1``` and ```sequence_length=64```.)

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 232
  referenced_widgets: [1cdd9d78ce9043f0ab5226ff3817e40f, 1183ef28125c4355b8384ac0a095a77d,
    8d5acca23dce4761a10beb7d12ea6237, e218568e72914679b1cd3788fde722e8, bf024667e0df4f14bed40f568ee1706d,
    e9991c72ab9b4b7d894eb35cfe9931d5, bd80b419051e4f0bb76d04cead2bf39a, de0d7bda08ae4d4cb52557070798a8a3,
    3406c7bb0b014329ae2028a595f4126b, 3713b7a03a974bb3bbb93a5447fa7756, 3a3a7517314743019c52e50865b09b62,
    3ccd889a14524cffb9ddc48b829ea80c, 482e429c0fd8480d852af1e9072a8327, b594a51ef6824172a54abd4f4bb47ee3,
    ea455218338e44d4bdf2829ac7549f94, b4ef3b1efeee4570b4cc235d0e9cb448, c4814db70d9d4396b02d110e65545564,
    d2ebc13056c64068b748fe3e99863701, ccab4d0e02424cf4a0cf858b43074e81, a3578756cac04bc1aff56fa88691b85d,
    27d903ca66a54e36b93de4cb59aec2b4, 6e2de8476f7c4f42a27c6493b08d8ebe, fe632e8555c04edc817e5cd374b44a49,
    fac901137e764d378a7af42088daf415, eaca6fa5b71a45e59a3903651475a4c7, 305ccd307e4f45678edf855ecba17402,
    ad3364e02d41480c93bfbef9cd87dbb0, 83c128175f514e7c8e9a57ebfff4b65c, 7bbb1a2df6c34f61bc100e6493ed6e12,
    125c421fc9274690b12d18d45ca23b88, b9981523db1d4d2abeb690aed4b4ee22, 0dba2d828ee44b96a581331ec9461037,
    9faeb9428c054b969d7eeaf4e49a9c5a, d73e8fc970104819add8b9a017bbd777, 905a81bcf37c447e888bf8227a8bf6d6,
    46a376bdb9bb4e0384f27da965d8704e, 4c13093c04a34c6cba5c7b61d5008371, f1fc861911ea49029eeddffcacb03ef5,
    333ba5cbe40444d1b249dfc7f4a436dc, 035ce7f12c494cefb4b1ec4c1703e58f, d2a888aa210a4be287fd9508f14cc44b,
    eb759afe634a49b5a045179a3d3f245f, 47023da6416747ccb6a7d47ecadc4bae, 10a5b71899f0429e956f5c9b549def3f,
    3cc63c284163439e84e93c0f481582b9, a8bec3ec0fc049418c0994bcc2d548be, 805f7204ee7b4854af4e2eade2cf1c51,
    1f24e7fc3a6d4a5eb88787dd43fbb98e, d8eeeeeba0814229b3f74eeb7077c520, eddc707e7c6e450c93270113bc4a7f4e,
    0d3fb328555a4e6ba6fc0e25bd701688, 1d37be0c3bb541f28b93022045fc7f3e, e8c1caadb8ee48e989c31b5aee741a32,
    d0d596a9f70e4ec69462aca935756629, 69ccd23f17514af28c93491cafcf1c03]
id: WV-Ygv4Mlnlo
outputId: 453da3db-d8b0-48e6-8f11-8cbff79333ff
---
model_name = 'bert-base-cased'
batch_size = 1
sequence_length = 64
model = BertModel.from_pretrained(model_name, batch_size=batch_size, sequence_length=sequence_length)
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
outputId: 3e6ad61b-f74b-4b61-89bb-80c3f118f6f5
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
outputId: 33d022bc-85fc-48cb-c5ee-8ab0d6d66157
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
outputId: 6cf948fe-df89-4375-9fc8-268570583365
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
outputId: ddb89066-9b37-47db-d13b-246f061b0582
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
tf.debugging.assert_near(tflite_output, model_outputs['token_embeddings'], rtol=3.0)
print("Outputs asserted and succesful:  ✅")
```

```{code-cell} ipython3
:id: mh3bNREFQyk0


```
