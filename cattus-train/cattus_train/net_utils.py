import keras
import tensorflow as tf

# N: number of images in the batch
# H: height of the image
# W: width of the image
# C: number of channels of the image
#
# CPU - NHWC ('channels_last')
# GPU - NCHW ('channels_first')


def mask_illegal_moves(target, output):
    output = tf.cast(output, tf.float32)
    legal_moves = tf.greater_equal(target, 0)
    output = tf.where(legal_moves, output, tf.zeros_like(output) - 1.0e10)
    target = tf.nn.relu(target)
    return target, output


@keras.saving.register_keras_serializable()
def loss_cross_entropy(target, output):
    target, output = mask_illegal_moves(target, output)
    policy_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.stop_gradient(target), logits=output
    )
    return tf.reduce_mean(input_tensor=policy_cross_entropy)


def loss_const_0(target, output):
    return tf.constant(0.0)


def policy_head_accuracy(target, output):
    target, output = mask_illegal_moves(target, output)
    return tf.reduce_mean(
        tf.cast(
            tf.equal(tf.argmax(input=target, axis=1), tf.argmax(input=output, axis=1)),
            tf.float32,
        )
    )


@keras.saving.register_keras_serializable()
def value_head_accuracy(target, output):
    # Both the target and output should be in range [-1,1]
    return 1 - tf.abs(target - output) / 2


def batch_norm(input, name, cpu, scale=False):
    axis = 3 if cpu else 1
    return tf.keras.layers.BatchNormalization(
        epsilon=1e-5, axis=axis, center=True, scale=scale, name=name
    )(input)


def conv_block(inputs, filter_size, output_channels, name, l2reg, cpu, bn_scale=False):
    conv_data_fmt = "channels_last" if cpu else "channels_first"

    # convolution
    flow = tf.keras.layers.Conv2D(
        output_channels,
        filter_size,
        use_bias=False,
        padding="same",
        kernel_initializer="glorot_normal",
        kernel_regularizer=l2reg,
        data_format=conv_data_fmt,
        name=name + "_conv2d",
    )(inputs)
    # batch normalization
    flow = batch_norm(flow, name=name + "_bn", cpu=cpu, scale=bn_scale)
    # a rectifier nonlinearity
    return tf.keras.layers.Activation("relu")(flow)


def residual_block(inputs, channels, name, l2reg, cpu):
    conv_data_fmt = "channels_last" if cpu else "channels_first"

    # convolution
    flow = tf.keras.layers.Conv2D(
        channels,
        3,
        use_bias=False,
        padding="same",
        kernel_initializer="glorot_normal",
        kernel_regularizer=l2reg,
        data_format=conv_data_fmt,
        name=name + "_1_conv2d",
    )(inputs)
    # batch normalization
    flow = batch_norm(flow, name + "_1_bn", cpu=cpu, scale=False)
    # a rectifier nonlinearity
    flow = tf.keras.layers.Activation("relu")(flow)

    # convolution
    flow = tf.keras.layers.Conv2D(
        channels,
        3,
        use_bias=False,
        padding="same",
        kernel_initializer="glorot_normal",
        kernel_regularizer=l2reg,
        data_format=conv_data_fmt,
        name=name + "_2_conv2d",
    )(flow)
    # batch normalization
    flow = batch_norm(flow, name + "_2_bn", cpu=cpu, scale=True)
    # ... (squeeze_excitation)
    #  skip connection adding input to the block
    flow = tf.keras.layers.add([inputs, flow])

    # a rectifier nonlinearity
    return tf.keras.layers.Activation("relu")(flow)


def create_convnetv1(
    inputs,
    residual_block_num,
    residual_filter_num,
    value_head_conv_output_channels_num,
    policy_head_conv_output_channels_num,
    moves_num,
    l2reg,
    cpu,
):
    l2reg = tf.keras.regularizers.l2(l=l2reg) if l2reg else None

    # single conv block
    flow = conv_block(
        inputs,
        filter_size=3,
        output_channels=residual_filter_num,
        name="input_planes",
        l2reg=l2reg,
        cpu=cpu,
        bn_scale=True,
    )

    # multiple residual blocks
    for block_idx in range(residual_block_num):
        flow = residual_block(
            flow,
            residual_filter_num,
            name="residual_{}".format(block_idx + 1),
            l2reg=l2reg,
            cpu=cpu,
        )

    # Value head
    flow_val = conv_block(
        flow,
        filter_size=1,
        output_channels=value_head_conv_output_channels_num,
        name="value",
        l2reg=l2reg,
        cpu=cpu,
    )
    flow_val = tf.keras.layers.Flatten()(flow_val)
    flow_val = tf.keras.layers.Dense(
        128,
        kernel_initializer="glorot_normal",
        kernel_regularizer=l2reg,
        activation="relu",
        name="value_dense1",
    )(flow_val)
    head_val = tf.keras.layers.Dense(
        1,
        kernel_initializer="glorot_normal",
        kernel_regularizer=l2reg,
        activation="tanh",
        name="value_head",
    )(flow_val)

    # Policy head
    flow_pol = conv_block(
        flow,
        filter_size=1,
        output_channels=policy_head_conv_output_channels_num,
        name="policy",
        l2reg=l2reg,
        cpu=cpu,
    )
    flow_pol = tf.keras.layers.Flatten()(flow_pol)
    head_pol = tf.keras.layers.Dense(
        moves_num,
        kernel_initializer="glorot_normal",
        kernel_regularizer=l2reg,
        bias_regularizer=l2reg,
        name="policy_head",
    )(flow_pol)

    return [head_val, head_pol]
