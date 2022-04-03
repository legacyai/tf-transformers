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

+++ {"colab": {"base_uri": "https://localhost:8080/"}, "id": "qjUYDME9kgWS", "outputId": "6390e343-f1b1-4ff1-e302-5603f6aab874"}

# Roberta TFLite

```{code-cell} ipython3
:id: CV0Bh-eFlEot

!pip install tf-transformers

!pip install sentencepiece

!pip install tensorflow-text

!pip install transformers
```

```{code-cell} ipython3

```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: cy6shXrXlL_D
outputId: 442bb294-6989-441f-b5b2-873ff7c51bd4
---
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Supper TF warnings

import tensorflow as tf
print("Tensorflow version", tf.__version__)

from tf_transformers.models import RobertaModel
```

```{code-cell} ipython3
:id: 0HEJnnnFlPxR


```

+++ {"id": "P89IVu5JlREX"}

## Convert a Model to TFlite

The most important thing to notice here is that, if we want to convert a model to ```tflite```, we have to ensure that ```inputs``` to the model are **deterministic**, which means inputs should not be dynamic. We have to fix  **batch_size**, **sequence_length** and other related input constraints depends on the model of interest.

### Load Roberta Model

1. Fix the inputs
2. We can always check the ```model``` **inputs** and **output** by using ```model.input``` and ```model.output```.
3. We use ```batch_size=1``` and ```sequence_length=64```.)

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 232
  referenced_widgets: [648c083eda83425893800f7bb7eb1c7d, b225237548cf455f94dd1c8c1859ba5d,
    fb4752ce685c446db45c70f96473cef3, 856a6321af584076bd2ae16b898ca54b, 0cf7e6b78d4e43fa927872c57df84ebe,
    2166ab84c4f346f3b6d61d77257a3a61, 50385731da9e48a4a19311d4ff86c4e1, 5f2b97c848c24463b4d721217e2855dd,
    413aeccc08f64b1fad5139d3f957515d, 0d9f911800884956b62bb64c85b14cff, 6cb7962efc874d97b3755da8dd262c82,
    1c26c357f40f422ebd355af346d5a870, c9a0a5c990a24d678dc0147b8e6fe57c, c91a09cb0a7c47ec8369ae207972aec8,
    3729e233e1d04b609ab9e23b90ec951a, 13b3f57565164f70bbd06cb11b7df1d6, 6159319a493f4c79ba6c0af40e62bd4b,
    39645a72317a447fa277daea323a147d, af0c292f168d4e279fb11a4d03c2b7a8, bb304f3386a54072a30dbc8ee7880ef7,
    bcca57cce3da4b33a4b829e606f9cceb, 37e97cd5e55c4589a5d582095b8a8376, c9d915d484584e7f911af2e84eb33392,
    b1a953560140499ea597b18c88e05158, d2f02fd5d9054e7eaa787f8bef51aaa9, 29b410a328ea4c1ba7c3e4d1f4d45fb1,
    45ff4245d6b74a40afdaa8cb4c90987a, 00c9f6980c9d4ea593a1695f9fe5da2a, ec538a9884334d35a3a4502047ffe161,
    98171768bc8f450787fcfef873d984ed, 32fa85adc44945f1a59b6f9c1a90c34d, 9e3fdb8cca51426795c652ba0a5e7e97,
    7f96d9702a92478990e3a8f4429fab60, f98d141b7cae45f8abeccb76ea408099, c32bb23b9bfa43cc982b5b3856677f13,
    b04321b81ec745dd86c42cc732377669, c740137ecec04e328adec78b82b5a5fd, f85ef686cbe848e481720a61dbcabdbc,
    277d65bc984248cc8c4c0a7255b52dce, bba83e20656b41a38f46177fb55d5a6a, 7762b62e7f314cc1b53673cea6e6cfc9,
    528bb619ed31413da9d4ecb8a0a583f7, b0e09ba2a24842f3892308eeacfe52ab, 7b2e675f975f4914a947f6232dbe01b5,
    6f073c18415940358f72122ac1cbdc23, ab50a4a24b484d9d89bae9de0348937a, 405c34c9ee084e1db0ac71519d7ab6ae,
    3773a8b196484f49990eede142b7f461, f87d174c78f0470da9076523155189e2, 6dcf028eddd24b2eb10963f3cc89dfd8,
    009d29a4ec6a44b5a05b47d231ad427a, 5742d2065e114c1fbe876add5860275c, b9ee7743406b4cbc99275606ba23a483,
    afd59259fdd94666b37983101bd5f465, 607b350ea3694adaa6555572b834f9b4]
id: WV-Ygv4Mlnlo
outputId: b7b16d9e-d45c-479b-8f09-85a619879ec2
---
model_name = 'roberta-base'
batch_size = 1
sequence_length = 64
model = RobertaModel.from_pretrained(model_name, batch_size=batch_size, sequence_length=sequence_length)
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
outputId: c9e0ad85-21bd-4bbc-e78c-f2552bf5c63a
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
outputId: a86d426a-951b-4531-ecaf-86e8fde0cb48
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
outputId: da1f5cc0-37a9-41c0-c988-0852b357bbb9
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
outputId: 5e52236c-db14-4a14-b56b-a7a9c9e8727a
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
