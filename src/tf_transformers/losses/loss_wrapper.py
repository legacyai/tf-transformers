import tensorflow as tf

from tf_transformers.losses import cross_entropy_loss_for_classification


def get_1d_classification_loss(label_column='labels', y_column='class_logits', loss_type: str = None):
    """Get loss based on joint or normal

    Args:
        label_column: the key from data dict.
        y_column: the key of model predictions.
        loss_type: None or "joint"
    """

    if loss_type and loss_type == 'joint':

        def loss_fn(y_true_dict, y_pred_dict):
            """Joint loss over all layers"""
            loss_dict = {}
            loss_holder = []
            for layer_count, per_layer_output in enumerate(y_pred_dict[y_column]):

                loss = cross_entropy_loss_for_classification(
                    labels=tf.squeeze(y_true_dict[label_column], axis=1), logits=per_layer_output
                )
                loss_dict['loss_{}'.format(layer_count + 1)] = loss
                loss_holder.append(loss)

            # Mean over batch, across all layers
            loss_dict['loss'] = tf.reduce_mean(loss_holder, axis=0)
            return loss_dict

    else:

        def loss_fn(y_true_dict, y_pred_dict):
            """last layer loss"""
            loss_dict = {}
            loss = cross_entropy_loss_for_classification(
                labels=tf.squeeze(y_true_dict[label_column], axis=1), logits=y_pred_dict[y_column]
            )
            loss_dict['loss'] = loss
            return loss_dict

    return loss_fn
