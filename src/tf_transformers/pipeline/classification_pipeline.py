import tensorflow as tf
class Classification_Pipeline:
    def __init__(self, model, tokenizer_fn, label_map, batch_size=32, classification_mode='multiclass'):

        self.get_model_fn(model)
        self.tokenizer_fn = tokenizer_fn
        self.label_map = label_map

        self.batch_size = batch_size
        
        if classification_mode == 'multiclass':
            self.prob_fn = tf.nn.softmax
        if classification_mode == 'multilabel':
            self.prob_fn = tf.nn.sigmoid

    def get_model_fn(self, model):
        self.model_fn = None
        # keras Model
        if isinstance(model, tf.keras.Model):
            self.model_fn = model
        else:
            # saved model
            if "saved_model" in str(type(model)):
                # Extract signature
                self.model_pb = model.signatures["serving_default"]

                def model_fn(x):
                    return self.model_pb(**x)

                self.model_fn = model_fn
        if self.model_fn is None:
            raise ValueError("Please check the type of your model")

    def run(self, dataset):
        token_logits = []
        for batch_inputs in dataset:
            model_outputs = self.model_fn(batch_inputs)
            token_logits.append(model_outputs["class_logits"])

        # Unstack
        token_logits_unstacked = []
        for batch_logits in token_logits:
            token_logits_unstacked.extend(tf.unstack(batch_logits))

        return token_logits_unstacked

    def convert_to_dataset(self, sentences):
        """Feaures to TF dataset"""
        dataset =tf.data.Dataset.from_tensor_slices(self.tokenizer_fn(sentences))
        dataset = dataset.batch(self.batch_size)
        return dataset
    
    def post_process(self, sentences, class_logits_unstacked):

        final_results = []
        for i in range(len(sentences)):
            item = class_logits_unstacked[i]
            probs = self.prob_fn(item)
            predicted_label = tf.argmax(probs).numpy()
            predicted_probs = tf.reduce_max(probs).numpy()
            final_results.append(
                {
                    "sentence": sentences[i],
                    "predicted_labels": self.label_map[predicted_label],
                    "predicted_probs": predicted_probs,
                }
            )
        return final_results

    def __call__(self, sentences):

        dev_dataset = self.convert_to_dataset(sentences)
        class_logits_unstacked = self.run(dev_dataset)
        final_results = self.post_process(sentences, class_logits_unstacked) 
        return final_results
