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

def main():
    # Parse the Command Line Options
    args = get_parser().parse_args()

    # Read the filenames in the DIRSIG Vehicle Classification Dataset
    train_filenames = np.genfromtxt('{}/{}'.format(args.train_dir, 'train.txt'), delimiter=' ', dtype=None)

    # Prepare the training dataset
    file_names, file_labels = dataset_iterator(args.train_dir, train_filenames)
    train_dataset = tf.data.Dataset.from_tensor_slices((file_names, file_labels))
    train_dataset = train_dataset.map(parse_function)
    train_dataset = train_dataset.shuffle(buffer_size=10000)
    train_batched_dataset = train_dataset.batch(args.batch_size)
    train_iterator = train_batched_dataset.make_one_shot_iterator()

    # Define Placeholders for the input data and labels and also save the images into the summary
    x_train = tf.placeholder(tf.float32, [args.batch_size, 56, 56, 3])
    y_train = tf.placeholder(tf.int32, [None, 2])
    tf.summary.image("training_input_image", x_train, max_outputs=20)

    # Build the Graph
    with tf.variable_scope("FirstStageFeatureExtractor"):
        # Create the first layer parameters
        W_conv1 = weight_variable([7, 7, 3, 64])
        b_conv1 = bias_variable([64])

        # Perform first convolutional layer and apply max pooling
        out_conv1 = tf.nn.conv2d(x_train, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
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
        tf.nn.softmax_cross_entropy_with_logits(labels=y_train, logits=y_pred))
    optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(cross_entropy)
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
        for epoch_number in range(args.number_epochs):
            train_batch = train_iterator.get_next()
            summary, _, loss_value = sess.run([merged, optimizer, cross_entropy], feed_dict={x_train: train_batch[0].eval(session=sess),
            y_train: train_batch[1].eval(session=sess)})
            train_writer.add_summary(summary, epoch_number)

            # Run the model on the test data for validation
            if epoch_number % args.test_frequency == 0:
                print("Loss at iteration {} : {}".format(epoch_number, loss_value))
                summary, acc = sess.run([merged, accuracy], feed_dict={
                x_train:train_batch[0].eval(session=sess), y_train: train_batch[1].eval(session=sess)})
                print("Accuracy at iteration {} : {}".format(epoch_number, acc))
                test_writer.add_summary(summary, epoch_number)

if __name__ == '__main__':
    main()
