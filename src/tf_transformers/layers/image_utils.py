import tensorflow as tf


def read_and_process(img, img_height, img_width, num_channels, rescale, normalize, read):
    """Read and Process image.
    If read is True, tf.io will read and parse the image.
    If read is False, we expect a numpy array (PIL open image) or TF image (tf.uint8)

    Args:
        img (): tf.string Filepath or tf.uint8 Image array
        img_height (int): Image Height for resizing
        img_width (int): Image Width for resizing
        num_channels (int): 3 for RGB and 1 for GrayScale
        rescale (bool): rescale image to (0 to 1)
        normalize (bool): normalize image by 0 mean and 1 stddev
        read (bool): to read and decode the image or to skip it

    Returns:
        tf.float32 (image array (3D))
    """
    if read:
        # Read image
        img = tf.io.read_file(img)
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=num_channels)

    # resize the image to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    # Rescale between (0 and 1)
    if rescale:
        img = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)(img)
    # TODO tf.keras.layers.experimental.preprocessing.Normalization
    if normalize:
        img = tf.image.per_image_standardization(img)
    return img
