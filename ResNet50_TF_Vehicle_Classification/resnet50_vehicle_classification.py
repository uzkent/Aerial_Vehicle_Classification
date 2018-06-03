import tensorflow as tf
import numpy as np

def _parse_function(filename, label):
    """ Reads an image from a file, decodes it into a dense tensor, and resizes it
    to a fixed shape """
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, [56, 56])
    return image_resized, label

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

# Read the filenames in the DIRSIG Vehicle Classification Dataset
filenames = np.genfromtxt('/home/burak/Downloads/vehicle_dataset/train_dirsig/train.txt', delimiter=' ', dtype=None)

# Save all the file names and labels
file_names = []
file_labels = []
for file_information in filenames:
    file_names.append('./{}/{}/{}'.format('vehicle_dataset', 'train_dirsig', file_information[0]))
    if file_information[1] == 0:
        file_labels.append([1, 0])
    else:
        file_labels.append([0, 1])

file_names = tf.constant(file_names)
file_labels = tf.constant(file_labels)
dataset = tf.data.Dataset.from_tensor_slices((file_names, file_labels))
dataset = dataset.map(_parse_function)

# Shuffle the dataset and define an iterator to the next batch of the dataset
dataset = dataset.shuffle(buffer_size=10000)
batched_dataset = dataset.batch(32)
iterator = batched_dataset.make_one_shot_iterator()

# Define Placeholders for the input data and labels and also save the images into the summary
x = tf.placeholder(tf.float32, [32, 56, 56, 3])
y = tf.placeholder(tf.int32, [None, 2])
tf.summary.image("input_image", x, max_outputs=20)

with tf.variable_scope("FirstStageFeatureExtractor"):
    # Create the first layer parameters
    W_conv1 = weight_variable([7, 7, 3, 64])
    b_conv1 = bias_variable([64])

    # Perform first convolutional layer and apply max pooling
    out_conv1 = tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
    out_relu1 = tf.nn.relu(out_conv1 + b_conv1)

with tf.variable_scope("ResNetBlock1"):
    # Apply ResNet Blocks
    out_res1 = resnet50_block(out_relu1, 64, 64, 256, stride=[1, 2, 2, 1])

    # Apply ResNet Blocks
    out_res2 = resnet50_block(out_res1, 64, 256, 256)

    # Apply ResNet Blocks
    out_block1 = resnet50_block(out_res2, 64, 256, 256)

with tf.variable_scope("ResNetBlock2"):
    # Apply ResNet Blocks
    out_res1 = resnet50_block(out_block1, 128, 256, 512, stride=[1, 2, 2, 1])

    # Apply ResNet Blocks
    out_res2 = resnet50_block(out_res1, 128, 512, 512)

    # Apply ResNet Blocks
    out_res3 = resnet50_block(out_res2, 128, 512, 512)

    # Apply ResNet Blocks
    out_block2 = resnet50_block(out_res3, 128, 512, 512)

with tf.variable_scope("ResNetBlock3"):
    # Apply ResNet Blocks
    out_res1 = resnet50_block(out_block2, 256, 512, 1024, stride=[1, 2, 2, 1])

    # Apply ResNet Blocks
    out_res2 = resnet50_block(out_res1, 256, 1024, 1024)

    # Apply ResNet Blocks
    out_res3 = resnet50_block(out_res2, 256, 1024, 1024)

    # Apply ResNet Blocks
    out_res4 = resnet50_block(out_res3, 256, 1024, 1024)

    # Apply ResNet Blocks
    out_res5 = resnet50_block(out_res4, 256, 1024, 1024)

    # Apply ResNet Blocks
    out_block3 = resnet50_block(out_res5, 256, 1024, 1024)

with tf.variable_scope("ResNetBlock4"):
    # Apply ResNet Blocks
    out_res1 = resnet50_block(out_block3, 512, 1024, 2048)

    # Apply ResNet Blocks
    out_res2 = resnet50_block(out_res1, 512, 2048, 2048)

    # Apply ResNet Blocks
    out_block4 = resnet50_block(out_res2, 512, 2048, 2048)

with tf.variable_scope("PredictionBlock"):
    # Apply Average Pooling to Reduce to 1x1 Feature Maps
    out_block4_pooled = tf.nn.pool(out_block4, window_shape=[7, 7], pooling_type='AVG',
                                    padding="VALID")
    print "!!!!!!!!", out_block4_pooled

    # Fully connected layer 1
    W_fc1 = weight_variable([2048, 1000])
    b_fc1 = bias_variable([1000])

    out_fc_layer = tf.reshape(out_block4_pooled, [-1, 1*1*2048])
    out_fc_layer_act = tf.nn.relu(tf.matmul(out_fc_layer, W_fc1) + b_fc1)

    # Fully connected layer 2
    W_fc2 = weight_variable([1000, 2])
    b_fc2 = bias_variable([2])

    # Perform the final fully connected layer
    y_pred = tf.matmul(out_fc_layer_act, W_fc2) + b_fc2

# Define the loss function and optimizer
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
tf.summary.scalar('cross_entropy', cross_entropy)

# Define the Classification Accuracy
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# Merge all visualized parameters
merged = tf.summary.merge_all()

# Create the session to perform training and validation
number_epochs = 10000
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('./train', sess.graph)
    test_writer = tf.summary.FileWriter('./test', sess.graph)
    sess.run(tf.global_variables_initializer())
    for epoch_number in range(number_epochs):
        batch = iterator.get_next()
        summary, _, loss_value = sess.run([merged, optimizer, cross_entropy], feed_dict={x: batch[0].eval(session=sess),
        y: batch[1].eval(session=sess)})
        train_writer.add_summary(summary, epoch_number)

        # Run the model on the test data for validation
        if epoch_number % 10 == 0:
            print "Loss at iteration {} : {}".format(epoch_number, loss_value)
            summary, acc = sess.run([merged, accuracy], feed_dict={
            x:batch[0].eval(session=sess), y: batch[1].eval(session=sess)})
            print "Accuracy at iteration {} : {}".format(epoch_number, acc)
            test_writer.add_summary(summary, epoch_number)
