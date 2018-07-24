import tensorflow as tf
import numpy as np
import pdb
from tensorflow.contrib import rnn

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_png(image_string, channels=1) 
  image_normalized = tf.image.per_image_standardization(image_decoded)
  return image_normalized, label

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

def weight_variable(shape, filter_name):
    """ Define the Weights and Initialize Them and Attach to the Summary """
    weights = tf.get_variable(filter_name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    variable_summaries(weights)
    return weights

def bias_variable(shape, bias_name):
    """ Define the Biases and Initialize Them and Attach to the Summary """
    bias = tf.get_variable(bias_name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    variable_summaries(bias)
    return bias

# Read the filenames in the DIRSIG Vehicle Classification Dataset
filenames = np.genfromtxt('./vehicle_dataset/train_dirsig/train.txt', delimiter=' ', dtype=None)
arr = np.arange(len(filenames))
np.random.shuffle(arr)

# Save all the file names and labels
file_names = []
file_labels = []
for index in range(len(filenames)):
    file_names.append('./{}/{}/{}'.format('vehicle_dataset', 'train_dirsig', filenames[arr[index]][0].decode('UTF-8')))
    if filenames[arr[index]][1] == 0:
        file_labels.append([1, 0])
    else:
        file_labels.append([0, 1])

file_names = tf.constant(file_names)
file_labels = tf.constant(file_labels, dtype=tf.float32)
dataset = tf.data.Dataset.from_tensor_slices((file_names, file_labels))
dataset = dataset.map(_parse_function)

# Shuffle the dataset and define an iterator to the next batch of the dataset
batch_size = 64
dataset = dataset.repeat(10000)
# dataset = dataset.shuffle(buffer_size=100)
batched_dataset = dataset.batch(batch_size)
iterator = batched_dataset.make_one_shot_iterator()

# Define Placeholders for the input data and labels
time_steps = 64
n_units = 256
n_inputs = 64
n_classes = 2
x = tf.placeholder(tf.float32, [None, n_inputs, time_steps])
input = tf.unstack(x, time_steps, 1)
y = tf.placeholder(tf.float32, [None, n_classes])

# Convert the RNN final output state to classification scores
out_weights = weight_variable([n_units, n_classes], 'prediction_weights')
out_bias = bias_variable([n_classes], 'prediction_bias')

#defining the network
lstm_layer = rnn.BasicLSTMCell(n_units, forget_bias=1)
outputs, _ = rnn.static_rnn(lstm_layer, input, dtype="float32")

# Prediction Layer
prediction = tf.nn.softmax(tf.matmul(outputs[-1], out_weights) + out_bias)

# Define the loss function and optimizer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
tf.summary.scalar('cross_entropy', cross_entropy)

# Define the Classification Accuracy
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# Merge all visualized parameters
merged = tf.summary.merge_all()

# Create the session to perform training and validation
number_epochs = 100000
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('./train', sess.graph)
    test_writer = tf.summary.FileWriter('./test', sess.graph)
    sess.run(tf.global_variables_initializer())
    batch = iterator.get_next()
    for epoch_number in range(number_epochs):

        # Update the weights
        train_batch = batch[0].eval(session=sess).reshape((batch_size, time_steps, n_inputs))
        train_labels = batch[1].eval(session=sess)
        summary, _, loss, _ = sess.run([merged, optimizer, cross_entropy, prediction], feed_dict={x: train_batch, y: train_labels})
        train_writer.add_summary(summary, epoch_number)

        # Perform validation on the test data
        if epoch_number % 100 == 0:
            # test_batch = batch[0].eval(session=sess).reshape((batch_size, time_steps, n_inputs))
            summary, acc, loss, preds = sess.run([merged, accuracy, cross_entropy, prediction], feed_dict={x: train_batch, y: train_labels})
            # print(preds)
            # print(train_labels)
            test_writer.add_summary(summary, epoch_number)
            print('Loss at iteration:{} - {} - and accuracy - {}'.format(epoch_number, loss, acc))
