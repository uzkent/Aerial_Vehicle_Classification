import tensorflow as tf
import numpy as np
import argparse

from resnet_block import weight_variable, bias_variable, variable_summaries, resnet50_block
from prepare_dataset import dataset_iterator, parse_function

def get_parser():
    """ This function returns a parser object """
    aparser = argparse.ArgumentParser()
    aparser.add_argument('--batch_size', type=int, default=32,
                         help='Batch size in the training stage')
    aparser.add_argument('--number_epochs', type=int, default=10,
                         help='Number of epochs to train the network')
    aparser.add_argument('--learning_rate', type=float, default=1e-4,
                         help='Learning rate')
    aparser.add_argument('--test_frequency', type=int, default=10,
                         help='After every provided number of iterations the model will be test')
    aparser.add_argument('--train_dir', type=str,
                         help='Provide the training directory to the text file with file names and labels in it')
    aparser.add_argument('--test_dir', type=str,
                     help='Provide the test directory to the text file with file names and labels in it')

    return aparser

def _conv2d(input_data, shape, bias_shape, stride, filter_id):
    weights = weight_variable(shape, 'weights' + filter_id)
    bias = bias_variable(bias_shape, 'bias' + filter_id)

    # Perform first convolutional layer and apply max pooling
    output_conv = tf.nn.conv2d(input_data, weights, strides=stride, padding='SAME')
    output_relu = tf.nn.relu(output_conv + bias)

    return output_relu

def main():
    # Parse the Command Line Options
    args = get_parser().parse_args()

    # Read the filenames in the DIRSIG Vehicle Classification Dataset
    train_filenames = np.genfromtxt('{}/{}'.format(args.train_dir, 'train.txt'), delimiter=' ', dtype=None)

    # Prepare the training dataset
    file_names, file_labels = dataset_iterator(args.train_dir, train_filenames)
    train_dataset = tf.data.Dataset.from_tensor_slices((file_names, file_labels))
    train_dataset = train_dataset.map(parse_function)
    train_dataset = train_dataset.repeat(2000)
    train_dataset = train_dataset.shuffle(buffer_size=10000)
    train_batched_dataset = train_dataset.batch(args.batch_size)
    train_iterator = train_batched_dataset.make_initializable_iterator()
    train_batch = train_iterator.get_next()

    # Define Placeholders for the input data and labels and also save the images into the summary
    x_train = tf.placeholder(tf.float32, [args.batch_size, 227, 227, 3])
    y_train = tf.placeholder(tf.int32, [None, 2])
    tf.summary.image("training_input_image", x_train, max_outputs=20)

    # Build the Graph
    with tf.variable_scope("ConvolutionalLayers"):
        out_1 = _conv2d(x_train, [11, 11, 3, 96], [96], [1, 4, 4, 1], '_1')
        out_2 = tf.nn.max_pool(out_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
        out_3 = _conv2d(out_2, [5, 5, 96, 256], [256], [1, 1, 1, 1], '_2')
        out_4 = tf.nn.max_pool(out_3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
        out_5 = _conv2d(out_4, [3, 3, 256, 384], [384], [1, 1, 1, 1], '_3')
        out_6 = _conv2d(out_5, [3, 3, 384, 384], [384], [1, 1, 1, 1], '_4')
        out_7 = _conv2d(out_6, [3, 3, 384, 256], [256], [1, 1, 1, 1], '_5')
        out_8 = tf.nn.max_pool(out_7, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
        print out_8

    with tf.variable_scope("FullyConnectedLayers"):
        W_fc1 = weight_variable([6*6*256, 4096], 'weights_fc_1')
        b_fc1 = bias_variable([4096], 'bias_fc_1')
        print out_8
        out_fc_layer = tf.reshape(out_8, [-1, 6*6*256])
        out_fc_layer1_act = tf.nn.relu(tf.matmul(out_fc_layer, W_fc1) + b_fc1)

        W_fc2 = weight_variable([4096, 4096], 'weights_fc_2')
        b_fc2 = bias_variable([4096], 'bias_fc_2')
        out_fc_layer2_act = tf.nn.relu(tf.matmul(out_fc_layer1_act, W_fc2) + b_fc2)

        W_fc3 = weight_variable([4096, 2], 'weights_fc_3')
        b_fc3 = bias_variable([2], 'bias_fc_3')
        y_pred = tf.matmul(out_fc_layer2_act, W_fc3) + b_fc3

    # Define the loss function and optimizer
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_train, logits=y_pred))
    optimizer = tf.train.GradientDescentOptimizer(args.learning_rate).minimize(cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy)

    # Define the Classification Accuracy
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_train,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Merge all visualized parameters
    merged = tf.summary.merge_all()

    # Execute the graph
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter('./train', sess.graph)
        test_writer = tf.summary.FileWriter('./test', sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(train_iterator.initializer)
        for epoch_number in range(args.number_epochs):
            summary, _, loss_value = sess.run([merged, optimizer, cross_entropy], feed_dict={x_train: train_batch[0].eval(session=sess),
            y_train: train_batch[1].eval(session=sess)})
            train_writer.add_summary(summary, epoch_number)
            print("Loss at iteration {} : {}".format(epoch_number, loss_value))

            # Run the model on the test data for validation
            if epoch_number % args.test_frequency == 0:
                summary, acc = sess.run([merged, accuracy], feed_dict={
                x_train:train_batch[0].eval(session=sess), y_train: train_batch[1].eval(session=sess)})
                print("Accuracy at iteration {} : {}".format(epoch_number, acc))
                test_writer.add_summary(summary, epoch_number)

if __name__ == '__main__':
    main()
