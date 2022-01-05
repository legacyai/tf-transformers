import tensorflow as tf


class WarmUp_Linear(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applies a warmup schedule on a given learning rate decay schedule."""

    def __init__(
        self,
        initial_learning_rate,
        num_training_steps,
        warmup_steps,
        power=1.0,
        name=None,
    ):
        super(WarmUp_Linear, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.num_training_steps = num_training_steps
        self.power = power
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmUp") as name:
            # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
            # learning rate will be `global_step/num_warmup_steps * init_lr`.
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            initial_learning_rate = tf.cast(self.initial_learning_rate, tf.float32)
            num_training_steps = tf.cast(self.num_training_steps, tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = initial_learning_rate * tf.math.pow(warmup_percent_done, self.power)

            def decay_fn():
                denominator = tf.maximum(1.0, num_training_steps - warmup_steps_float)
                numerator = num_training_steps - global_step_float
                return initial_learning_rate * tf.maximum(0.0, tf.divide(numerator, denominator))

            return tf.cond(
                tf.less(global_step_float, warmup_steps_float),
                lambda: warmup_learning_rate,
                lambda: decay_fn(),
                name=name,
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "warmup_steps": self.warmup_steps,
            "num_training_steps": self.num_training_steps,
            "power": self.power,
            "name": self.name,
        }


class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applies a warmup schedule on a given learning rate decay schedule."""

    def __init__(
        self,
        initial_learning_rate,
        decay_schedule_fn,
        warmup_steps,
        power=1.0,
        name=None,
    ):
        super(WarmUp, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.decay_schedule_fn = decay_schedule_fn
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmUp") as name:
            # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
            # learning rate will be `global_step/num_warmup_steps * init_lr`.
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * tf.math.pow(warmup_percent_done, self.power)
            return tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: self.decay_schedule_fn(step),
                name=name,
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule_fn": self.decay_schedule_fn,
            "warmup_steps": self.warmup_steps,
            "power": self.power,
            "name": self.name,
        }
