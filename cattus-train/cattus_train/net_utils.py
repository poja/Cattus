import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filter_size: int,
        output_channels: int,
        bn_scale=False,
    ):
        super().__init__()
        # def conv_block(inputs, filter_size, output_channels, name, l2reg, cpu, bn_scale=False):
        #     conv_data_fmt = "channels_last" if cpu else "channels_first"

        #     # convolution
        #     flow = tf.keras.layers.Conv2D(
        #         output_channels,
        #         filter_size,
        #         use_bias=False,
        #         padding="same",
        #         kernel_initializer="glorot_normal",
        #         kernel_regularizer=l2reg,
        #         data_format=conv_data_fmt,
        #         name=name + "_conv2d",
        #     )(inputs)
        #     # batch normalization
        #     flow = batch_norm(flow, name=name + "_bn", cpu=cpu, scale=bn_scale)
        #     # a rectifier nonlinearity
        #     return tf.keras.layers.Activation("relu")(flow)
        self._conv = nn.Conv2d(
            in_channels, output_channels, filter_size, bias=False, padding="same"
        )
        self._bn = nn.BatchNorm2d(output_channels, affine=bn_scale)
        self._relu = nn.ReLU()

    def forward(self, input):
        flow = self._conv(input)
        flow = self._bn(flow)
        return self._relu(flow)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
    ):
        super().__init__()
        # def residual_block(inputs, channels, name, l2reg, cpu):
        #     conv_data_fmt = "channels_last" if cpu else "channels_first"

        #     # convolution
        #     flow = tf.keras.layers.Conv2D(
        #         channels,
        #         3,
        #         use_bias=False,
        #         padding="same",
        #         kernel_initializer="glorot_normal",
        #         kernel_regularizer=l2reg,
        #         data_format=conv_data_fmt,
        #         name=name + "_1_conv2d",
        #     )(inputs)
        #     # batch normalization
        #     flow = batch_norm(flow, name + "_1_bn", cpu=cpu, scale=False)
        #     # a rectifier nonlinearity
        #     flow = tf.keras.layers.Activation("relu")(flow)

        #     # convolution
        #     flow = tf.keras.layers.Conv2D(
        #         channels,
        #         3,
        #         use_bias=False,
        #         padding="same",
        #         kernel_initializer="glorot_normal",
        #         kernel_regularizer=l2reg,
        #         data_format=conv_data_fmt,
        #         name=name + "_2_conv2d",
        #     )(flow)
        #     # batch normalization
        #     flow = batch_norm(flow, name + "_2_bn", cpu=cpu, scale=True)
        #     # ... (squeeze_excitation)
        #     #  skip connection adding input to the block
        #     flow = tf.keras.layers.add([inputs, flow])

        #     # a rectifier nonlinearity
        #     return tf.keras.layers.Activation("relu")(flow)
        self._conv1 = nn.Conv2d(channels, channels, 3, bias=False, padding="same")
        self._bn1 = nn.BatchNorm2d(channels, affine=False)
        self._relu1 = nn.ReLU()
        self._conv2 = nn.Conv2d(channels, channels, 3, bias=False, padding="same")
        self._bn2 = nn.BatchNorm2d(channels, affine=True)
        self._relu2 = nn.ReLU()

    def forward(self, input):
        flow = self._conv1(input)
        flow = self._bn1(flow)
        flow = self._relu1(flow)
        flow = self._conv2(flow)
        flow = self._bn2(flow)
        return self._relu2(input + flow)


class ConvNetV1(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int, int],
        residual_block_num: int,
        residual_filter_num: int,
        value_head_conv_output_channels_num: int,
        policy_head_conv_output_channels_num: int,
        moves_num: int,
    ):
        super().__init__()

        # def create_convnetv1(
        #     inputs,
        #     residual_block_num,
        #     residual_filter_num,
        #     value_head_conv_output_channels_num,
        #     policy_head_conv_output_channels_num,
        #     moves_num,
        #     l2reg,
        #     cpu,
        # ):
        #     l2reg = tf.keras.regularizers.l2(l=l2reg) if l2reg else None

        #     # single conv block
        #     flow = conv_block(
        #         inputs,
        #         filter_size=3,
        #         output_channels=residual_filter_num,
        #         name="input_planes",
        #         l2reg=l2reg,
        #         cpu=cpu,
        #         bn_scale=True,
        #     )

        #     # multiple residual blocks
        #     for block_idx in range(residual_block_num):
        #         flow = residual_block(
        #             flow,
        #             residual_filter_num,
        #             name="residual_{}".format(block_idx + 1),
        #             l2reg=l2reg,
        #             cpu=cpu,
        #         )

        #     # Value head
        #     flow_val = conv_block(
        #         flow,
        #         filter_size=1,
        #         output_channels=value_head_conv_output_channels_num,
        #         name="value",
        #         l2reg=l2reg,
        #         cpu=cpu,
        #     )
        #     flow_val = tf.keras.layers.Flatten()(flow_val)
        #     flow_val = tf.keras.layers.Dense(
        #         128,
        #         kernel_initializer="glorot_normal",
        #         kernel_regularizer=l2reg,
        #         activation="relu",
        #         name="value_dense1",
        #     )(flow_val)
        #     head_val = tf.keras.layers.Dense(
        #         1,
        #         kernel_initializer="glorot_normal",
        #         kernel_regularizer=l2reg,
        #         activation="tanh",
        #         name="value_head",
        #     )(flow_val)

        #     # Policy head
        #     flow_pol = conv_block(
        #         flow,
        #         filter_size=1,
        #         output_channels=policy_head_conv_output_channels_num,
        #         name="policy",
        #         l2reg=l2reg,
        #         cpu=cpu,
        #     )
        #     flow_pol = tf.keras.layers.Flatten()(flow_pol)
        #     head_pol = tf.keras.layers.Dense(
        #         moves_num,
        #         kernel_initializer="glorot_normal",
        #         kernel_regularizer=l2reg,
        #         bias_regularizer=l2reg,
        #         name="policy_head",
        #     )(flow_pol)

        B, C, H, W = input_shape
        assert B == 1

        # single conv block
        self._conv1 = ConvBlock(C, 3, residual_filter_num, bn_scale=True)  # B, RC, H, W

        # multiple residual blocks
        self._residual_blocks = nn.Sequential(
            *[ResidualBlock(residual_filter_num) for _ in range(residual_block_num)]
        )  # B, RC, H, W

        # Value head
        self._value_head = nn.Sequential(
            ConvBlock(
                residual_filter_num, 1, value_head_conv_output_channels_num
            ),  # B, VHC, H, W
            nn.Flatten(),  # B, VHC * H * W
            nn.Linear(value_head_conv_output_channels_num * H * W, 128),  # B, 128
            nn.ReLU(),
            nn.Linear(128, 1),  # B, 1
            nn.Tanh(),
        )

        # Policy head
        self._policy_head = nn.Sequential(
            ConvBlock(
                residual_filter_num, 1, policy_head_conv_output_channels_num
            ),  # B, PHC, H, W
            nn.Flatten(),  # B, PHC * H * W
            nn.Linear(policy_head_conv_output_channels_num * H * W, moves_num),
        )

    def forward(self, input):
        flow = self._conv1(input)
        flow = self._residual_blocks(flow)
        value = self._value_head(flow)
        policy = self._policy_head(flow)
        return policy, value


class SimpleTwoHeadedModel(nn.Module):
    def __init__(self, input_shape, moves_num):
        super().__init__()

        B, C, H, W = input_shape
        assert B == 1
        features_num = C * H * W

        self._flatten = nn.Flatten()
        self._dense1 = nn.Linear(features_num, features_num)
        self._relu1 = nn.ReLU()
        self._dense2 = nn.Linear(features_num, features_num)
        self._relu2 = nn.ReLU()

        self._value_head = nn.Linear(features_num, 1)
        self._value_head_activation = nn.Tanh()

        self._policy_head = nn.Linear(features_num, moves_num)

    def forward(self, input):
        flow = self._flatten(input)
        flow = self._relu1(self._dense1(flow))
        flow = self._relu2(self._dense2(flow))

        policy_flow = self._policy_head(flow)

        value_flow = self._value_head(flow)
        value_flow = self._value_head_activation(value_flow)

        return policy_flow, value_flow
