import tensorflow as tf
from absl import logging

from tf_transformers.core.legacy_compile import (LossesContainer,
                                                 MetricsContainer)
from tf_transformers.core.legacy_module import LegacyModuleCustom

logging.set_verbosity("INFO")


class LegacyModel(tf.keras.Model):
    def compile2(
        self,
        optimizer="rmsprop",
        loss=None,
        custom_loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
        **kwargs,
    ):
        """Configures the model for training.

        Arguments:
            optimizer: String (name of optimizer) or optimizer instance. See
              `tf.keras.optimizers`.
            loss: String (name of objective function), objective function or
              `tf.keras.losses.Loss` instance. See `tf.keras.losses`. An objective
              function is any callable with the signature `loss = fn(y_true,
              y_pred)`, where y_true = ground truth values with shape =
              `[batch_size, d0, .. dN]`, except sparse loss functions such as sparse
              categorical crossentropy where shape = `[batch_size, d0, .. dN-1]`.
              y_pred = predicted values with shape = `[batch_size, d0, .. dN]`. It
              returns a weighted loss float tensor. If a custom `Loss` instance is
              used and reduction is set to NONE, return value has the shape
              [batch_size, d0, .. dN-1] ie. per-sample or per-timestep loss values;
              otherwise, it is a scalar. If the model has multiple outputs, you can
              use a different loss on each output by passing a dictionary or a list
              of losses. The loss value that will be minimized by the model will
              then be the sum of all individual losses.
            custom_loss: dict of loss fn
            metrics: List of metrics to be evaluated by the model during training
              and testing. Each of this can be a string (name of a built-in
              function), function or a `tf.keras.metrics.Metric` instance. See
              `tf.keras.metrics`. Typically you will use `metrics=['accuracy']`. A
              function is any callable with the signature `result = fn(y_true,
              y_pred)`. To specify different metrics for different outputs of a
              multi-output model, you could also pass a dictionary, such as
                `metrics={'output_a': 'accuracy', 'output_b': ['accuracy', 'mse']}`.
                  You can also pass a list (len = len(outputs)) of lists of metrics
                  such as `metrics=[['accuracy'], ['accuracy', 'mse']]` or
                  `metrics=['accuracy', ['accuracy', 'mse']]`. When you pass the
                  strings 'accuracy' or 'acc', we convert this to one of
                  `tf.keras.metrics.BinaryAccuracy`,
                  `tf.keras.metrics.CategoricalAccuracy`,
                  `tf.keras.metrics.SparseCategoricalAccuracy` based on the loss
                  function used and the model output shape. We do a similar
                  conversion for the strings 'crossentropy' and 'ce' as well.
            loss_weights: Optional list or dictionary specifying scalar coefficients
              (Python floats) to weight the loss contributions of different model
              outputs. The loss value that will be minimized by the model will then
              be the *weighted sum* of all individual losses, weighted by the
              `loss_weights` coefficients.
                If a list, it is expected to have a 1:1 mapping to the model's
                  outputs. If a dict, it is expected to map output names (strings)
                  to scalar coefficients.
            weighted_metrics: List of metrics to be evaluated and weighted by
              sample_weight or class_weight during training and testing.
            run_eagerly: Bool. Defaults to `False`. If `True`, this `Model`'s
              logic will not be wrapped in a `tf.function`. Recommended to leave
              this as `None` unless your `Model` cannot be run inside a
              `tf.function`.
            **kwargs: Any additional arguments. Supported arguments:
                - `experimental_steps_per_execution`: Int. The number of batches to
                  run during each `tf.function` call. Running multiple batches
                  inside a single `tf.function` call can greatly improve performance
                  on TPUs or small models with a large Python overhead. Note that if
                  this value is set to `N`, `Callback.on_batch` methods will only be
                  called every `N` batches. This currently defaults to `1`. At most,
                  one full epoch will be run each execution. If a number larger than
                  the size of the epoch is passed, the execution will be truncated
                  to the size of the epoch.
                - `sample_weight_mode` for backward compatibility.

        Raises:
            ValueError: In case of invalid arguments for
                `optimizer`, `loss` or `metrics`.
        """
        # _keras_api_gauge.get_cell('compile2').set(True)
        with self.distribute_strategy.scope():
            self._validate_compile(optimizer, metrics, **kwargs)
            self._run_eagerly = run_eagerly

            self.optimizer = self._get_optimizer(optimizer)

            custom_loss_copy = custom_loss.copy()
            if custom_loss is not None:
                if not isinstance(custom_loss, dict):
                    raise ValueError(
                        "Custom loss should be ideally dict, \
                    where each model output key has corresponding loss_fn "
                    )
                if loss is not None:
                    raise ValueError(
                        "When `loss` is set `custom_loss` shouldn't be set \
                and vice versa"
                    )
                # set loss to custom_loss
                loss = custom_loss_copy.copy()

            # self.compiled_loss = legacy_compile.LossesContainer(
            #     loss, loss_weights, output_names=self.output_names)
            self.compiled_loss = LossesContainer(loss, loss_weights, output_names=self.output_names)
            # self.compiled_metrics = compile_utils.MetricsContainer(
            #     metrics, weighted_metrics, output_names=self.output_names)
            self.compiled_metrics = MetricsContainer(metrics, weighted_metrics, output_names=self.output_names)

            experimental_steps_per_execution = kwargs.pop("experimental_steps_per_execution", 1)
            self._configure_steps_per_execution(experimental_steps_per_execution)

            # Initializes attrs that are reset each time `compile` is called.
            self._reset_compile_cache()
            self._is_compiled = True

            self.loss = loss or {}  # Backwards compat.

    def load_checkpoint(self, checkpoint_dir):
        """[summary]

        Args:
            checkpoint_dir ([str]): [Location of the model]
        """
        checkpoint = tf.train.Checkpoint(model=self)
        manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=1)
        status = checkpoint.restore(manager.latest_checkpoint)
        # Important
        if status.assert_existing_objects_matched():
            logging.info("Succesful: Model checkpoints matched")
        else:
            logging.info("Failed to load the checkpoint. Status Assertion Failed.")

    def save_checkpoint(self, checkpoint_dir, overwrite=False):
        """Save checkpoint

        Args:
            checkpoint_dir ([str]): [Location of the model]

        Returns:
            None
        """
        if not overwrite:
            import os

            if os.path.exists(checkpoint_dir):
                raise FileExistsError()

        # If you want to save the model as checkpoints
        checkpoint = tf.train.Checkpoint(model=self)
        manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=1)
        manager.save()
        logging.info("Saved model at {}".format(manager.latest_checkpoint))

    def save_as_serialize_module(self, directory, overwrite=False):
        """Save as tf.saved_model.save (.pb)

        Args:
            directory ([str]): [Location of the model]

        Returns:
            None
        """
        if not overwrite:
            import os

            if os.path.exists(directory):
                raise FileExistsError()

        module = LegacyModuleCustom(self)
        module.save(directory)
