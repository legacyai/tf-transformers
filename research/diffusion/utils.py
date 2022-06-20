import tensorflow as tf


# As per original diffusion code
def get_initializer(scale=0):
    if scale == 0:
        scale = scale = 1e-10
    initializer = tf.keras.initializers.VarianceScaling(scale=scale, mode='fan_avg', distribution='uniform')
    return initializer
