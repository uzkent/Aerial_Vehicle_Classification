import tensorflow as tf
import numpy as np
import argparse
from tqdm import tqdm

from prepare_dataset import dataset_iterator, parse_function

def get_parser():
    """ This function returns a parser object """
    aparser = argparse.ArgumentParser()
    aparser.add_argument('--batch_size', type=int, default=32,
                         help='Batch size in the training stage')
    aparser.add_argument('--number_epochs', type=int, default=100,
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

class ResNet50():
    """ This class contains the components of the ResNet50 Architecture """
    def variable_summaries(self, var):
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

    def weight_variable(self, shape, filter_name):
        """ Define the Weights and Initialize Them and Attach to the Summary """
        weights = tf.get_variable(filter_name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        self.variable_summaries(weights)
        return weights

    def bias_variable(self, shape, bias_name):
        """ Define the Biases and Initialize Them and Attach to the Summary """
        bias = tf.get_variable(bias_name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        self.variable_summaries(bias)
        return bias

    def _conv2d(self, input_data, shape, bias_shape, stride, filter_id, padding='SAME'):
        """ Perform 2D convolution on the input data and apply RELU """
        weights = self.weight_variable(shape, 'weights' + filter_id)
        bias = self.bias_variable(bias_shape, 'bias' + filter_id)
        output_conv = tf.nn.conv2d(input_data, weights, strides=stride, padding='SAME')
        return tf.nn.relu(output_conv + bias)

    def _fcl(self, input_data, shape, bias_shape, filter_id, classification_layer=False):
        """ Run a Fully Connected Layer and ReLU if necessary """
        weights = self.weight_variable(shape, 'weights'+  filter_id)
        bias = self.bias_variable(bias_shape, 'bias' + filter_id)

        if classification_layer:
            return tf.matmul(input_data, weights) + bias
        else:
            out_fc_layer = tf.reshape(input_data, [-1, shape[0]])
            return tf.nn.relu(tf.matmul(out_fc_layer, weights) + bias)

    def resnet50_block(self, input_feature_map, number_bottleneck_channels,
    number_input_channels, number_output_channels, stride=[1, 1, 1, 1]):
        """ Run a ResNet block """
        out_1 = self._conv2d(input_feature_map, [1, 1, number_input_channels, number_bottleneck_channels],
        [number_bottleneck_channels], [1, 1, 1, 1], 'bottleneck_down')
        out_2 = self._conv2d(out_1, [3, 3, number_bottleneck_channels, number_bottleneck_channels],
        [number_bottleneck_channels], stride, 'conv3x3')
        out_3 = self._conv2d(out_2, [1, 1, number_bottleneck_channels, number_output_channels],
        [number_output_channels], [1, 1, 1, 1], 'bottleneck_up')
        identity_mapping = self._conv2d(input_feature_map, [1, 1, number_input_channels, number_output_channels],
        [number_output_channels], stride, 'identity_mapping')
        return tf.add(identity_mapping, out_3)

    def resnet50_module(self, input_data, number_blocks, number_bottleneck_channels, number_input_channels,
    number_output_channels, stride=[1, 2, 2, 1]):
        """ Run a ResNet module consisting of residual blocks """
        for index, block in enumerate(range(number_blocks)):
            if index == 0:
                with tf.variable_scope('module' + str(index)):
                    out = self.resnet50_block(input_data, number_bottleneck_channels, number_input_channels,
                    number_output_channels, stride=stride)
            else:
                with tf.variable_scope('module' + str(index)):
                    out = self.resnet50_block(out, number_bottleneck_channels, number_output_channels,
                    number_output_channels, stride=[1, 1, 1, 1])

        return out

def get_data(file_names, file_labels, batch_size):
    """ This function returns a pointer to iterate over the batches of data """
    train_dataset = tf.data.Dataset.from_tensor_slices((file_names, file_labels))
    train_dataset = train_dataset.map(parse_function)
    train_dataset = train_dataset.repeat(200)
    train_dataset = train_dataset.shuffle(buffer_size=1000)
    train_batched_dataset = train_dataset.batch(batch_size)
    train_iterator = train_batched_dataset.make_initializable_iterator()

    return train_iterator.get_next(), train_iterator

def main():
    args = get_parser().parse_args()

    # Read the filenames in the DIRSIG Vehicle Classification Dataset
    train_filenames = np.genfromtxt('{}/{}'.format(args.train_dir, 'train.txt'), delimiter=' ', dtype=None)

    # Prepare the training dataset
    file_names, file_labels = dataset_iterator(args.train_dir, train_filenames)
    train_batch, train_iterator = get_data(file_names, file_labels, args.batch_size)
    x_train = tf.placeholder(tf.float32, [args.batch_size, 56, 56, 3])
    y_train = tf.placeholder(tf.int32, [None, 2])
    tf.summary.image("training_input_image", x_train, max_outputs=20)

    # Build the Graph
    net = ResNet50()
    with tf.variable_scope("FirstStageFeatureExtractor") as scope:
        out_1 = net._conv2d(x_train, [3, 3, 3, 64], [64], [1, 1, 1, 1], 'conv3x3')

    with tf.variable_scope("ResNetBlock1"):
        out_2 = net.resnet50_module(out_1, 3, 64, 64, 256, [1, 1, 1, 1])

    with tf.variable_scope("ResNetBlock2"):
        out_3 = net.resnet50_module(out_2, 4, 128, 256, 512)

    with tf.variable_scope("ResNetBlock3"):
        out_4 = net.resnet50_module(out_3, 6, 256, 512, 1024)

    with tf.variable_scope("ResNetBlock4"):
        out_5 = net.resnet50_module(out_4, 3, 512, 1024, 2048)

    with tf.variable_scope("PredictionBlock"):
        out_6 = tf.nn.pool(out_5, window_shape=[7, 7], pooling_type='AVG', padding="VALID")
        out_7 = net._fcl(out_6, [2048, 1024], [1024], 'fc_1')
        y_pred = net._fcl(out_7, [1024, 2], [2], 'fc_2', classification_layer=True)

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
        for epoch_number in tqdm(range(args.number_epochs)):
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
