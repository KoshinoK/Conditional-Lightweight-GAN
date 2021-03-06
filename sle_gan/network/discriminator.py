import tensorflow as tf

import sle_gan
from sle_gan import center_crop_images
from sle_gan.network.common_layers import GLU


class InputBlock(tf.keras.layers.Layer):
    def __init__(self, downsampling_factor: int, filters, **kwargs):
        super().__init__(**kwargs)
        assert downsampling_factor in [1, 2, 4]

        conv_1_strides = 2
        conv_2_strides = 2

        if downsampling_factor <= 2:
            conv_2_strides = 1

        if downsampling_factor == 1:
            conv_1_strides = 1

        self.conv_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=4, strides=conv_1_strides, padding="same")
        self.activation_1 = tf.keras.layers.LeakyReLU(0.1)
        self.conv_2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=4, strides=conv_2_strides, padding="same")
        self.normalization = tf.keras.layers.BatchNormalization()
        self.activation_2 = tf.keras.layers.LeakyReLU(0.1)

    def call(self, inputs, **kwargs):
        x = self.conv_1(inputs)
        x = self.activation_1(x)
        x = self.conv_2(x)
        x = self.normalization(x)
        x = self.activation_2(x)
        return x


class DownSamplingBlock1(tf.keras.layers.Layer):
    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)

        self.conv_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=4, strides=2, padding="same")
        self.normalization_1 = tf.keras.layers.BatchNormalization()
        self.activation_1 = tf.keras.layers.LeakyReLU(0.1)

        self.conv_2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding="same")
        self.normalization_2 = tf.keras.layers.BatchNormalization()
        self.activation_2 = tf.keras.layers.LeakyReLU(0.1)

    def call(self, inputs, **kwargs):
        x = self.conv_1(inputs)
        x = self.normalization_1(x)
        x = self.activation_1(x)
        x = self.conv_2(x)
        x = self.normalization_2(x)
        x = self.activation_2(x)
        return x


class DownSamplingBlock2(tf.keras.layers.Layer):
    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)

        self.pooling = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding="valid")
        self.normalization = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.LeakyReLU(0.1)

    def call(self, inputs, **kwargs):
        x = self.pooling(inputs)
        x = self.conv(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x


class DownSamplingBlock(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)

        self.down_1 = DownSamplingBlock1(filters)
        self.down_2 = DownSamplingBlock2(filters)

    def call(self, inputs, **kwargs):
        x_1 = self.down_1(inputs)
        x_2 = self.down_2(inputs)
        return x_1 + x_2


class SimpleDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, output_filters, **kwargs):
        super().__init__(**kwargs)
        self.upsampling = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest")
        self.conv = tf.keras.layers.Conv2D(filters=output_filters * 2, kernel_size=3, padding="same")
        self.normalization = tf.keras.layers.BatchNormalization()
        self.glu = GLU()

    def call(self, inputs, **kwargs):
        x = self.upsampling(inputs)
        x = self.conv(x)
        x = self.normalization(x)
        x = self.glu(x)
        return x


class SimpleDecoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.decoder_block_filter_sizes = [256, 128, 128, 64]
        self.decoder_blocks = [SimpleDecoderBlock(output_filters=x) for x in self.decoder_block_filter_sizes]
        self.conv_output = tf.keras.layers.Conv2D(3, 1, 1, padding="same")

    def call(self, inputs, **kwargs):
        x = inputs
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x)
        x = self.conv_output(x)
        x = tf.nn.tanh(x)
        return x


class RealFakeOutputBlock(tf.keras.layers.Layer):
    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)

        self.conv_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1)
        self.normalization = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.LeakyReLU(0.1)
        self.conv_2 = tf.keras.layers.Conv2D(filters=1, kernel_size=4)

    def call(self, inputs, **kwargs):
        x = self.conv_1(inputs)
        x = self.normalization(x)
        x = self.activation(x)
        x = self.conv_2(x)
        return x


class Discriminator(tf.keras.models.Model):
    def __init__(self, input_resolution: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert input_resolution in [256, 512, 1024], "Resolution should be 256 or 512 or 1024"
        self.input_resolution = input_resolution

        downsampling_factor_dict = {256: 1, 512: 2, 1024: 4}
        input_block_filters_dict = {256: 8, 512: 16, 1024: 32}
        self.input_block = InputBlock(filters=input_block_filters_dict[input_resolution],
                                      downsampling_factor=downsampling_factor_dict[input_resolution])

        self.downsample_128 = DownSamplingBlock(filters=64)
        self.downsample_64 = DownSamplingBlock(filters=128)
        self.downsample_32 = DownSamplingBlock(filters=128)
        self.downsample_16 = DownSamplingBlock(filters=256)
        self.downsample_8 = DownSamplingBlock(filters=512)

        self.decoder_image_part = SimpleDecoder()
        self.decoder_image = SimpleDecoder()

        self.real_fake_output = RealFakeOutputBlock(filters=256)

    """
    条件ベクトルを付加するように修正
        修正済み
    """
    def initialize(self, batch_size, conditional_vectors):
        sample_input = tf.random.uniform(shape=(batch_size, self.input_resolution, self.input_resolution, 3), minval=0,
                                         maxval=1, dtype=tf.float32)
        sample_output = self.call(sle_gan.data.create_discriminator_inputs(sample_input, conditional_vectors))
        return sample_output

    @tf.function
    def call(self, inputs, training=None, mask=None):
        """
        inputs: real or fake。条件ベクトルがチャンネルとして付加されている。
        """
        x = self.input_block(inputs)  # --> (B, 256, 256, F)

        x = self.downsample_128(x)  # --> (B, 128, 128, 64)
        x = self.downsample_64(x)  # --> (B, 64, 64, 128)
        x = self.downsample_32(x)  # --> (B, 32, 32, 128)
        x_16 = self.downsample_16(x)  # --> (B, 16, 16, 256)
        x_8 = self.downsample_8(x_16)  # --> (B, 8, 8, 512)

        # TODO: instead of just center cropping implement random cropping
        center_cropped_x_16 = center_crop_images(x_16, 8)  # --> (B, 8, 8, 64)
        x_image_decoded_128_center_part = self.decoder_image_part(center_cropped_x_16)  # --> (B, 128, 128, 3)
        x_image_decoded_128 = self.decoder_image(x_8)  # --> (B, 128, 128, 3)

        x_real_fake_logits = self.real_fake_output(x_8)  # --> (B, 5, 5, 1)

        return x_real_fake_logits, x_image_decoded_128, x_image_decoded_128_center_part
