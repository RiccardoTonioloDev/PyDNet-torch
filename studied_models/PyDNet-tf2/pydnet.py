from layers import *
import tensorflow as tf


class pydnet(object):
    def __init__(self, placeholders=None):
        self.model_collection = ["PyDnet"]
        self.placeholders = placeholders
        self.build_model()
        self.build_outputs()

    def build_model(self):
        pyramid = self.build_pyramid(self.placeholders["im0"])

        # SCALE 6
        with tf.name_scope("L6"):
            with tf.name_scope("estimator"):
                conv6 = self.build_estimator(pyramid[6])
                self.disp7 = self.get_disp(conv6)
            with tf.name_scope("upsampler"):
                upconv6 = self.bilinear_upsampling_by_deconvolution(conv6)

        # SCALE 5
        with tf.name_scope("L5"):
            with tf.name_scope("estimator"):
                conv5 = self.build_estimator(pyramid[5], upconv6)
                self.disp6 = self.get_disp(conv5)
            with tf.name_scope("upsampler"):
                upconv5 = self.bilinear_upsampling_by_deconvolution(conv5)

        # SCALE 4
        with tf.name_scope("L4"):
            with tf.name_scope("estimator"):
                conv4 = self.build_estimator(pyramid[4], upconv5)
                self.disp5 = self.get_disp(conv4)
            with tf.name_scope("upsampler"):
                upconv4 = self.bilinear_upsampling_by_deconvolution(conv4)

        # SCALE 3
        with tf.name_scope("L3"):
            with tf.name_scope("estimator"):
                conv3 = self.build_estimator(pyramid[3], upconv4)
                self.disp4 = self.get_disp(conv3)
            with tf.name_scope("upsampler"):
                upconv3 = self.bilinear_upsampling_by_deconvolution(conv3)

        # SCALE 2
        with tf.name_scope("L2"):
            with tf.name_scope("estimator"):
                conv2 = self.build_estimator(pyramid[2], upconv3)
                self.disp3 = self.get_disp(conv2)
            with tf.name_scope("upsampler"):
                upconv2 = self.bilinear_upsampling_by_deconvolution(conv2)

        # SCALE 1
        with tf.name_scope("L1"):
            with tf.name_scope("estimator"):
                conv1 = self.build_estimator(pyramid[1], upconv2)
                self.disp2 = self.get_disp(conv1)

    # Pyramidal features extraction
    def build_pyramid(self, input_batch):
        features = [input_batch]
        with tf.name_scope("conv1a"):
            conv1a = conv2d_leaky(input_batch, [3, 3, 3, 16], [16], 2, True)
        with tf.name_scope("conv1b"):
            conv1b = conv2d_leaky(conv1a, [3, 3, 16, 16], [16], 1, True)
        features.append(conv1b)

        with tf.name_scope("conv2a"):
            conv2a = conv2d_leaky(conv1b, [3, 3, 16, 32], [32], 2, True)
        with tf.name_scope("conv2b"):
            conv2b = conv2d_leaky(conv2a, [3, 3, 32, 32], [32], 1, True)
        features.append(conv2b)

        with tf.name_scope("conv3a"):
            conv3a = conv2d_leaky(conv2b, [3, 3, 32, 64], [64], 2, True)
        with tf.name_scope("conv3b"):
            conv3b = conv2d_leaky(conv3a, [3, 3, 64, 64], [64], 1, True)
        features.append(conv3b)

        with tf.name_scope("conv4a"):
            conv4a = conv2d_leaky(conv3b, [3, 3, 64, 96], [96], 2, True)
        with tf.name_scope("conv4b"):
            conv4b = conv2d_leaky(conv4a, [3, 3, 96, 96], [96], 1, True)
        features.append(conv4b)

        with tf.name_scope("conv5a"):
            conv5a = conv2d_leaky(conv4b, [3, 3, 96, 128], [128], 2, True)
        with tf.name_scope("conv5b"):
            conv5b = conv2d_leaky(conv5a, [3, 3, 128, 128], [128], 1, True)
        features.append(conv5b)

        with tf.name_scope("conv6a"):
            conv6a = conv2d_leaky(conv5b, [3, 3, 128, 192], [192], 2, True)
        with tf.name_scope("conv6b"):
            conv6b = conv2d_leaky(conv6a, [3, 3, 192, 192], [192], 1, True)
        features.append(conv6b)

        return features

    # Single scale estimator
    def build_estimator(self, features, upsampled_disp=None):
        if upsampled_disp is not None:
            disp2 = tf.concat([features, upsampled_disp], -1)
        else:
            disp2 = features

        with tf.name_scope("disp-3"):
            disp3 = conv2d_leaky(disp2, [3, 3, disp2.shape[3], 96], [96], 1, True)
        with tf.name_scope("disp-4"):
            disp4 = conv2d_leaky(disp3, [3, 3, disp3.shape[3], 64], [64], 1, True)
        with tf.name_scope("disp-5"):
            disp5 = conv2d_leaky(disp4, [3, 3, disp4.shape[3], 32], [32], 1, True)
        with tf.name_scope("disp-6"):
            disp6 = conv2d_leaky(
                disp5, [3, 3, disp5.shape[3], 8], [8], 1, False
            )  # 8 channels for compatibility with other devices
        return disp6

    # Upsampling layer
    def bilinear_upsampling_by_deconvolution(self, x):
        f = x.get_shape().as_list()[-1]
        return deconv2d_leaky(x, [2, 2, f, f], f, 2, True)

    # Disparity prediction layer
    def get_disp(self, x):
        disp = 0.3 * tf.nn.sigmoid(tf.slice(x, [0, 0, 0, 0], [-1, -1, -1, 2]))
        return disp

    # Build multi-scale outputs
    def build_outputs(self):
        shape = tf.shape(self.placeholders["im0"])
        size = [shape[1], shape[2]]
        self.results = (
            tf.image.resize(self.disp2, size),
            tf.image.resize(self.disp3, size),
            tf.image.resize(self.disp4, size),
        )
