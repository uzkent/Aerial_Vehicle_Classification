import tensorflow as tf
import numpy as np
import pandas as pd
import utils
import pdb
import math

from tqdm import tqdm

class SSDResNet50():
    """ This class contains the components of the ResNet50 Architecture """
    feature_layers = ['block2', 'block3', 'block4']

    def __init__(self):
        """ Constructor for the SSD-ResNet50 Model """
        self.number_classes = 1
        self.number_iterations = 1000
        self.anchor_sizes = [(21., 45.),
                      (45., 99.),
                      (99., 153.)]
        self.anchor_ratios = [[2, .5],
                        [2, .5],
                        [2, .5]]
        self.feat_shapes = [[200, 200],[100, 100],[50, 50]]
        self.anchor_steps = [8, 16, 32]
        self.img_shape = [400, 400]
        self.batch_size = 1
        self.positive_threshold = 0.7
        self.negative_threshold = 0.3

    def weight_variable(self, shape, filter_name):
        """ Define the Weights and Initialize Them and Attach to the Summary """
        weights = tf.get_variable(filter_name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        return weights

    def bias_variable(self, shape, bias_name):
        """ Define the Biases and Initialize Them and Attach to the Summary """
        bias = tf.get_variable(bias_name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
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

    '''
    def jaccard_with_anchors(self, anchors_layer, bbox):
        """ Compute jaccard score between a ground truth box and the anchors.
            Source : https://github.com/balancap/SSD-Tensorflow/blob/e0e3104d3a2cc5d830fad041d4e56ebcf84caac3/nets/ssd_common.py#L64
        """
        # [TODO]@BurakUzkent : Anchor boxes need to be projected to the image domain to
        # compute IoU's w.r.t ground truth
        yref, xref, href, wref = anchors_layer
        ymin = yref - href / 2.
        xmin = xref - wref / 2.
        ymax = yref + href / 2.
        xmax = xref + wref / 2.
        vol_anchors = (xmax - xmin) * (ymax - ymin)

        int_ymin = tf.maximum(ymin, bbox[0])
        int_xmin = tf.maximum(xmin, bbox[1])
        int_ymax = tf.minimum(ymax, bbox[2])
        int_xmax = tf.minimum(xmax, bbox[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)

        inter_vol = h * w
        union_vol = vol_anchors - inter_vol \
            + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        return tf.div(inter_vol, union_vol)
    '''

    def ssd_anchor_box_encoder(self, index, dtype=np.float32, offset=0.5):
        """ Compute SSD anchor boxes in the domain of feature maps of interest to perform
            detections.
        """
        # Compute the position grid: simple way.
        y, x = np.mgrid[0:self.feat_shapes[index][0], 0:self.feat_shapes[index][1]]
        y = (y.astype(dtype) + offset) * self.anchor_steps[index] / self.img_shape[0]
        x = (x.astype(dtype) + offset) * self.anchor_steps[index] / self.img_shape[1]

        # Expand dims to support easy broadcasting.
        y = np.expand_dims(y, axis=-1)
        x = np.expand_dims(x, axis=-1)

        # Compute relative height and width.
        # Tries to follow the original implementation of SSD for the order.
        num_anchors = len(self.anchor_sizes[index]) + len(self.anchor_ratios[index])
        h = np.zeros((num_anchors, ), dtype=dtype)
        w = np.zeros((num_anchors, ), dtype=dtype)
        # Add first anchor boxes with ratio=1.
        h[0] = self.anchor_sizes[index][0] / self.img_shape[0]
        w[0] = self.anchor_sizes[index][0] / self.img_shape[1]
        di = 1
        for i, r in enumerate(self.anchor_ratios[index]):
            h[i+di] = self.anchor_sizes[index][0] / self.img_shape[0] / math.sqrt(float(r))
            w[i+di] = self.anchor_sizes[index][0] / self.img_shape[1] * math.sqrt(float(r))

        return y, x, h, w

    def detection_layer(self, inputs, index):
        """ Predict bounding box locations and classes in each head """
        net = inputs
        num_anchors = len(self.anchor_sizes[index]) + len(self.anchor_ratios[index])

        # Location prediction - Returns nxnx(4xnum_anchors) tensor
        num_loc_pred = num_anchors * 4
        filter_loc = tf.get_variable('conv_loc', [3, 3, net.get_shape()[3].value, num_loc_pred],
                            initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
        loc_predictions = tf.nn.conv2d(net, filter_loc,
                                padding="SAME", strides=[1, 1, 1, 1])
        # Class prediction - Return nxnx(number_classes) tensor
        num_class_pred = num_anchors * self.number_classes
        filter_class = tf.get_variable('conv_class', [3, 3, net.get_shape()[3].value, num_class_pred],
                            initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
        class_predictions = tf.nn.conv2d(net, filter_class,
                                padding="SAME", strides=[1, 1, 1, 1])

        return class_predictions, loc_predictions

    def loss_function(self, gt_localizations, gt_classes, overall_predictions, overall_anchors):
        """ Define the loss function for SSD - Classification + Localization """
        overall_loss = 0
        for index, (predictions, anchors) in enumerate(zip(overall_predictions, overall_anchors)):
            target_labels, target_localizations, target_scores = utils.ssd_bboxes_encode_layer(gt_classes, gt_localizations, anchors, 1)
            pos_samples = tf.cast(target_scores > self.positive_threshold, tf.uint16)
            num_pos_samples = tf.reduce_sum(pos_samples)
            neg_samples = tf.cast(target_scores < self.negative_threshold, tf.uint16)
            num_neg_samples = tf.reduce_sum(neg_samples)

            # [TODO]@BurakUzkent : Here we should limit the number of negative samples to 3*num_pos_samples
            target_labels_flattened = tf.reshape(target_labels, [-1])
            predictions_flattened = tf.reshape(predictions[0], [-1])
            pos_samples_flattened = tf.to_float(tf.reshape(pos_samples, [-1]))
            neg_samples_flattened = tf.to_float(tf.reshape(neg_samples, [-1]))
            with tf.name_scope('cross_entropy_pos{}'.format(index)):
                loss = tf.nn.softmax_cross_entropy_with_logits(logits=predictions_flattened, labels=target_labels_flattened)
                loss_classification_pos = tf.div(tf.reduce_sum(loss * pos_samples_flattened), self.batch_size, name='value')

            with tf.name_scope('cross_entropy_neg{}'.format(index)):
                loss = tf.nn.softmax_cross_entropy_with_logits(logits=predictions_flattened, labels=target_labels_flattened)
                loss_classification_neg = tf.div(tf.reduce_sum(loss * neg_samples_flattened), self.batch_size, name='value')

            with tf.name_scope('localization{}'.format(index)):
                pdb.set_trace()
                weights = tf.expand_dims(1.0 * pos_samples_flattened, axis=-1)
                loss = tf.subtract(tf.abs(predictions[1]), tf.abs(target_localizations))
                loss_localizaiton = tf.div(tf.reduce_sum(loss * weights), self.batch_size, name='value')

            overall_loss += loss_classification_pos + loss_classification_neg + loss_localization

        return overall_loss

def ssd_resnet50():

    # Construct the Graph
    net = SSDResNet50()
    endpoints = {}
    x_train = tf.placeholder(tf.float32, [net.batch_size, net.img_shape[0], net.img_shape[1], 1])
    with tf.variable_scope("FirstStageFeatureExtractor") as scope:
        out_1 = net._conv2d(x_train, [3, 3, 1, 64], [64], [1, 1, 1, 1], 'conv3x3')
    with tf.variable_scope("ResNetBlock1"):
        out_2 = net.resnet50_module(out_1, 3, 64, 64, 256, [1, 1, 1, 1])
        endpoints['block1'] = out_2
    with tf.variable_scope("ResNetBlock2"):
        out_3 = net.resnet50_module(out_2, 4, 128, 256, 512)
        endpoints['block2'] = out_3
    with tf.variable_scope("ResNetBlock3"):
        out_4 = net.resnet50_module(out_3, 6, 256, 512, 1024)
        endpoints['block3'] = out_4
    with tf.variable_scope("ResNetBlock4"):
        out_5 = net.resnet50_module(out_4, 3, 512, 1024, 2048)
        endpoints['block4'] = out_4
    with tf.variable_scope("PredictionBlock"):
        out_6 = tf.nn.pool(out_5, window_shape=[8, 8], pooling_type='AVG', padding="VALID")
        out_7 = net._fcl(out_6, [2048, 1024], [1024], 'fc_1')
        y_pred = net._fcl(out_7, [1024, 2], [2], 'fc_2', classification_layer=True)

    # Perform Detections on the Desired Blocks
    overall_predictions = []
    overall_anchors = []
    for index, layer in enumerate(net.feature_layers):
        with tf.variable_scope("PredictionModule{}".format(index)):
            overall_anchors.append(net.ssd_anchor_box_encoder(index))
            overall_predictions.append(net.detection_layer(endpoints[layer], index))

    # [TODO]@BurakUzkent : Add a module to read the ground truth data for the given batch
    file_names = ['~/Downloads/profile_picture.jpeg'] # An example image
    gt_bboxes = tf.constant(np.reshape(np.asarray([0.1, 0.1, 0.2, 0.2], np.float32), (1,4)))
    gt_classes = tf.constant([1], tf.int64)
    # gt_bboxes, file_names = utils.annotation_finder(images_csv, annotations_csv)
    train_batch, train_iterator = utils.create_tf_dataset(file_names, net.batch_size)

    """
    # Find the overlaps between the anchors and ground truth bounding boxes
    overall_overlaps = []
    for gt_box in gt_bboxes:
        for anchor_layer in overall_anchors:
            overall_overlaps.append(net.jaccard_with_anchors(gt_box, anchor_layer))
    """

    total_loss = net.loss_function(gt_bboxes, gt_classes, overall_predictions, overall_anchors)
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(total_loss)

    # Execute the graph
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter('./train', sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(train_iterator.initializer)
        for iteration_id in tqdm(range(net.number_iterations)):
            _, loss_value = sess.run([optimizer, total_loss], feed_dict={x_train: train_batch[0].eval(session=sess)})
            print("Loss at iteration {} : {}".format(iteration_id, loss_value))