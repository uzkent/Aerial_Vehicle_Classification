import tensorflow as tf

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def weight_variable(shape):
    """ Define the Weights and Initialize Them and Attach to the Summary """
    initial = tf.truncated_normal(shape, stddev=0.1)
    weights = tf.Variable(initial)
    variable_summaries(weights)
    return weights

def bias_variable(shape):
    """ Define the Biases and Initialize Them and Attach to the Summary """
    initial = tf.constant(0.1, shape=shape)
    bias = tf.Variable(initial)
    variable_summaries(bias)
    return bias

def resnet50_block(input_feature_map, number_bottleneck_channels,
                    number_input_channels, number_output_channels, stride=[1, 1, 1, 1]):
    """ Run a ResNet Block """
    W_conv1_res = weight_variable([1, 1, number_input_channels, number_bottleneck_channels])
    b_conv1_res = bias_variable([number_bottleneck_channels])

    # Apply 1x1 convolution
    out_conv1_res = tf.nn.conv2d(input_feature_map, W_conv1_res, strides=[1, 1, 1, 1], padding='SAME')
    out_relu1_res = tf.nn.relu(out_conv1_res + b_conv1_res)

    # Perform a ResNet Block
    W_conv2_res = weight_variable([3, 3, number_bottleneck_channels, number_bottleneck_channels])
    b_conv2_res = bias_variable([number_bottleneck_channels])

    # Apply 3x3 convolution - Stride of 2 applied in some cases
    out_conv2_res = tf.nn.conv2d(out_relu1_res, W_conv2_res, strides=stride, padding='SAME')
    out_relu2_res = tf.nn.relu(out_conv2_res + b_conv2_res)

    # Perform a ResNet Block
    W_conv3_res = weight_variable([1, 1, number_bottleneck_channels, number_output_channels])
    b_conv3_res = bias_variable([number_output_channels])

    # Apply 1x1 convolution
    out_conv3_res = tf.nn.conv2d(out_relu2_res, W_conv3_res, strides=[1, 1, 1, 1], padding='SAME')
    out_relu3_res = tf.nn.relu(out_conv3_res + b_conv3_res)

    # Project the Previous Feature Map to the Current Feature Map Size
    project_weights = weight_variable([1, 1, number_input_channels, number_output_channels])
    project_bias = bias_variable([number_output_channels])

    # Apply 1x1 convolution
    project_out = tf.nn.conv2d(input_feature_map, project_weights, strides=stride, padding='SAME')
    project_out_act = tf.nn.relu(project_out + project_bias)

    # Perform Residual Connection
    return tf.add(out_relu3_res, project_out_act)
