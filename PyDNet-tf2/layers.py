import tensorflow as tf


# Hand-made leaky relu
def leaky_relu(x, alpha=0.2):
    return tf.maximum(x, alpha * x)


# 2D convolution wrapper
def conv2d_leaky(x, kernel_shape, bias_shape, strides=1, relu=True, padding="SAME"):
    initializer = tf.initializers.GlorotUniform()
    weights = tf.Variable(initializer(shape=kernel_shape), dtype=tf.float32)
    biases = tf.Variable(tf.random.truncated_normal(bias_shape), dtype=tf.float32)

    output = tf.nn.conv2d(x, weights, strides=[1, strides, strides, 1], padding=padding)
    output = tf.nn.bias_add(output, biases)

    if relu:
        output = leaky_relu(output, 0.2)
    return output


# 2D deconvolution wrapper
def deconv2d_leaky(x, kernel_shape, bias_shape, strides=1, relu=True, padding="SAME"):
    initializer = tf.initializers.GlorotUniform()
    weights = tf.Variable(initializer(shape=kernel_shape), dtype=tf.float32)
    biases = tf.Variable(tf.random.truncated_normal(bias_shape), dtype=tf.float32)

    x_shape = tf.shape(x)
    output_shape = [
        x_shape[0],
        x_shape[1] * strides,
        x_shape[2] * strides,
        kernel_shape[2],
    ]

    output = tf.nn.conv2d_transpose(
        x,
        weights,
        output_shape=output_shape,
        strides=[1, strides, strides, 1],
        padding=padding,
    )
    output = tf.nn.bias_add(output, biases)

    if relu:
        output = leaky_relu(output, 0.2)
    return output
