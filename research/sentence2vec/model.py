import tensorflow as tf
from similarity_model import Similarity_Model_Pretraining

from tf_transformers.models import AlbertModel
from tf_transformers.optimization import create_optimizer


def loss_fn(_batch_labels, model_outputs):  # noqa
    logits_masked = model_outputs['logits']
    logits_mean_masked = model_outputs['logits_mean']

    labels = tf.range(tf.shape(logits_masked)[0])
    loss_masked = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits_masked)
    loss_mean_masked = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits_mean_masked)
    loss = (loss_masked + loss_mean_masked) / 2.0

    return {'loss_cls': loss_masked, 'loss_mean': loss_mean_masked, 'loss': loss}


def get_optimizer(
    learning_rate,
    num_train_steps,
    num_warmup_steps,
    decay_function='linear',
    weight_decay_rate=0.1,
    optimizer_type='adamw',
    use_constant_lr=False,
):
    # As in GPT2 paper, end_learning_rate = 0.1 * learning_rate
    end_learning_rate = 0.1 * learning_rate

    def optimizer_fn():
        optimizer, learning_rate_fn = create_optimizer(
            init_lr=learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            decay_function=decay_function,
            weight_decay_rate=weight_decay_rate,
            end_learning_rate=end_learning_rate,
            optimizer_type=optimizer_type,
            use_constant_lr=use_constant_lr,
        )
        return optimizer

    return optimizer_fn


def get_model(clip_logits, use_random_base):

    if use_random_base is False:

        def model_fn():
            encoder = AlbertModel.from_pretrained("albert-base-v2")
            decoder_config = AlbertModel.get_config("albert-base-v2")
            # decoder_config['num_hidden_layers']= 6
            decoder = AlbertModel.from_config(decoder_config)
            encoder.save_checkpoint("/tmp/albert/", overwrite=True)
            decoder.load_checkpoint("/tmp/albert")

            model = Similarity_Model_Pretraining(
                encoder=encoder, projection_dimension=768, decoder=decoder, clip_logits=clip_logits, siamese=False
            )
            model = model.get_model()

            return model

        return model_fn

    else:

        def model_fn():
            config = AlbertModel.get_config("albert-base-v2")
            encoder = AlbertModel.from_config(config)
            decoder = AlbertModel.from_config(config)
            model = Similarity_Model_Pretraining(
                encoder=encoder, projection_dimension=768, decoder=decoder, clip_logits=clip_logits, siamese=False
            )
            model = model.get_model()

            return model

        return model_fn
