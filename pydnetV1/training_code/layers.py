#
# MIT License
#
# Copyright (c) 2018 Matteo Poggi m.poggi@unibo.it
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
import keras


# Hand-made leaky relu
def leaky_relu(x, alpha=0.2):
    return tf.maximum(x, alpha * x)  # Just a handmade Leaky ReLU


# 2D convolution wrapper
def conv2d_leaky(x, kernel_shape, bias_shape, strides=1, relu=True, padding='SAME'):
    # Conv2D
    weights = tf.compat.v1.get_variable(
        "weights",  # The name of the variable we are referring to (NOTE: it has to be unique in the context where it is
        # defined, otherwise an error will occur).
        kernel_shape,  # This determines the shape of the weight matrix
        initializer=keras.initializers.GlorotUniform(),  # This specifies the way the weights will be initialized
        # the Xavier initializer it's one among various choices (suggested to use in this case)
        dtype=tf.float32  # The type used in memory to store the weights
    )
    biases = tf.compat.v1.get_variable("biases", bias_shape, initializer=keras.initializers.truncated_normal(), dtype=tf.float32)
    # Same thing described in the line above

    output = tf.nn.conv2d(
        x,  # Here we are passing the input data through the convolutional layer
        # It should have a size of: [batch_size, input_height, input_width, input_channels]
        weights,  # This is the filter that will be applied
        # It should have a size of: [filter_height, filter_width, input_channels, output_channels]
        strides=[1, strides, strides, 1],  # It specifies the stride of the convolution along each dimension
        # of the input tensor. It respects the [batch_size, input_height, input_width, input_channels] format.
        # Saying 1 in the first and last element specifies that we are considering every element in the training set,
        # and we are considering every channel. The two in between can be set for example to 2, meaning that one pixel
        # every two is skipped.
        padding=padding  # Determines how the input tensor is padded. "SAME" means that the input is padded in such a
        # way that the output feature map has the same spatial dimensions as the input. "VALID" means that the filter
        # is applied only where it fully overlaps with the tensor.
    )
    
    output = tf.nn.elu(output)  # Application of the activation function.
    # This line was added by me to handle conflicts inherited by the migration from TF1 to TF2.

    output = tf.nn.bias_add(output, biases)  # Here we are just adding biases to the output tensor

    # ReLU (if required)
    if relu:
        output = leaky_relu(output, 0.2)  # Here it's applying the leaky ReLU function previously built to the
        # output tensor
    return output


# 2D deconvolution wrapper
def deconv2d_leaky(x, kernel_shape, bias_shape, strides=1, relu=True, padding='SAME'):
    # Conv2D
    weights = tf.compat.v1.get_variable("weights", kernel_shape, initializer=keras.initializers.GlorotUniform(),
                              dtype=tf.float32)
    # Same as in the above function

    biases = tf.compat.v1.get_variable("biases", bias_shape, initializer=keras.initializers.truncated_normal(), dtype=tf.float32)
    # Same as in the above function

    x_shape = tf.shape(x)  # Here it's extracting the shape size of the input tensor
    outputShape = [x_shape[0], x_shape[1] * strides, x_shape[2] * strides, kernel_shape[2]]
    # Here it's computing the shape of the output layer, using the shape previously extracted, but doubling the height
    # and the width of each image (we are doubling the dimensionality), moving in the opposite direction of the above
    # function.

    output = tf.nn.conv2d_transpose(x,
                                    weights,
                                    output_shape=outputShape,  # Here we are using the shape calculated before.
                                    strides=[1, strides, strides, 1],  # Note that in transposed convolutions strides
                                    # increase the dimensionality of the output.
                                    padding=padding
                                    )

    output = tf.nn.elu(output)  # Application of the activation function.
    # This line was added by me to handle conflicts inherited by the migration from TF1 to TF2.

    output = tf.nn.bias_add(output, biases)
    # ReLU (if required)
    if relu:
        output = leaky_relu(output, 0.2)
    return output
