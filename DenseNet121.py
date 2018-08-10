import os
import tensorflow as tf
import numpy as np
import time
import math
import argparse
import random
import pandas as pd
from io import StringIO
import io

from tqdm import tqdm
import prepare_dataset


CATEGORY = ['park', 'forest','camp', 'hospital', 'road', 'university', 'bay','river', 'company', 'airport', 'stadium','power station', 'populated place', 'waterfall', 'lake', 'school', 'college', 'valley', 'hotel', 'building']


def build_index():
    dictionary = {}
    for index in range(len(CATEGORY)):
        dictionary[CATEGORY[index]]=index
    return dictionary


def get_parser():
    """ This function returns a parser object """
    aparser = argparse.ArgumentParser()
    aparser.add_argument('--batch_size', type=int, default=1,
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
    aparser.add_argument('--image_path', type=str,
                        help='Path to image folder')
    aparser.add_argument('--csv_path', type=str,
                        help='Path to csv file')
    aparser.add_argument('--train_file_name', type=str,
                           help='training set file name')  
    aparser.add_argument('--val_file_name', type=str,
                           help='validation set file name') 
    return aparser


class DenseNet121():
    
    def __init__(self, dense_block_num=4, growth_rate=32, filter_num=64, layers_num = [6,12,24,16], reduction=0.0, dropout_rate=0.0, weight_decay=1e-4, classes=40, weights_path=None):  
        self.compression = 1.0 - reduction
        self.dense_block_num = dense_block_num
        self.growth_rate = growth_rate
        self.filter_num = filter_num
        self.layers_num = layers_num
        self.reduction = reduction
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.classes = classes
        self.concat_axis = 3
        
    """ This class contains the components of the DenseNet121 Architecture """
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
    
    def _batch_norm(self, input, name, is_training):
        """ Apply Batch Normalization After Convolution and Before Activation """
        input_norm = tf.contrib.layers.batch_norm(input, center=True, scale=True, is_training=is_training)
        return input_norm
    
    def _conv2d(self, input_data, shape, bias_shape, stride, filter_id, is_training, padding='SAME'):
            """ Perform 2D convolution on the input data and apply RELU """
            weights = self.weight_variable(shape, 'weights' + filter_id)
            bias = self.bias_variable(bias_shape, 'bias' + filter_id)
            output_conv = tf.nn.conv2d(input_data, weights, strides=stride, padding='SAME')
            output_conv_norm = self._batch_norm(output_conv + bias, filter_id, is_training)
            return tf.nn.relu(output_conv_norm)
        
    def _fcl(self, input_data, shape, bias_shape, filter_id, need_relu=False):
        """ Run a Fully Connected Layer and ReLU if necessary """
        weights = self.weight_variable(shape, 'weights'+  filter_id)
        bias = self.bias_variable(bias_shape, 'bias' + filter_id)
        out_put = tf.matmul(input_data, weights) + bias
        
        if need_relu:
            out_put = tf.nn.relu(out_put)
        return tf.matmul(input_data, weights) + bias
    
    
    def conv_block(self, input_data, is_training, stage, branch, filter_num, dropout_rate=None, weight_decay=1e-4):
        """ Build convolutional layers for dense blocks"""
        conv_name_base = 'convolution' + str(stage) + '_' + str(branch)
        relu_name_base = 'relu' + str(stage) + '_' + str(branch)      
        if filter_num == None:
            filter_num = self.filter_num      
        # 1x1 Convolution (Bottleneck layer)
        out = self._batch_norm(input_data, conv_name_base+'_X1_bn', is_training)
        out = tf.nn.relu(out, name = relu_name_base+'_X1')
        # Produce 4k feature-maps
        out = self._conv2d(out, [1,1, out.get_shape().as_list()[3], filter_num*4], [filter_num*4], [1,1,1,1], conv_name_base+'x1', is_training, padding='SAME')
        if dropout_rate:
            out = tf.nn.dropout(out, dropout_rate)
        # 3x3 Convolution
        out = self._batch_norm(out, conv_name_base+'_X2_bn', is_training)
        out = tf.nn.relu(out, name = relu_name_base+'_X2')
        out = self._conv2d(out, [3,3, out.get_shape().as_list()[3], filter_num], [filter_num], [1,1,1,1], conv_name_base+'X2', is_training, padding='SAME')
        if dropout_rate:
            out = tf.nn.dropout(out, dropout_rate)
        return out
      
    
    def dense_block(self, input_data, stage, layer_num, filter_num, growth_rate, is_training, dropout_rate=None, weight_decay=1e-4, grow_filter_num=True):
        """ Build dense blocks"""
        concat_feat = input_data
        for i in range(layer_num):
            branch = i + 1
            input_data = self.conv_block(concat_feat,is_training, stage, branch, growth_rate, dropout_rate, weight_decay)
            concat_axis = self.concat_axis
            concat_feat = tf.concat([concat_feat, input_data], axis=concat_axis, name='concat_'+str(stage)+'_'+str(branch))
            if grow_filter_num:
                filter_num += growth_rate
        return concat_feat, filter_num
     
    
    def transition_block(self, input_data, stage, filter_num, is_training, compression=1.0, dropout_rate=None, weight_decay=1e-4):
        """ Build transition block"""
        conv_name_base = 'convolution' + str(stage) + '_blk'
        relu_name_base = 'relu' + str(stage) + '_blk'
        pool_name_base = 'pool' + str(stage)     
        out = self._batch_norm(input_data, conv_name_base+'_X1_bn', is_training)        
        out = tf.nn.relu(out, name = relu_name_base+'_X1')      
        out = self._conv2d(out, [1,1,out.get_shape().as_list()[3], int(filter_num * compression)], [int(filter_num * compression)], [1,1,1,1], conv_name_base, is_training, padding='SAME')
        
        if dropout_rate:
            out = tf.nn.dropout(out, dropout_rate)
        out = tf.nn.pool(out, window_shape=[2, 2], strides=[2,2],pooling_type='AVG', padding='VALID')    
        return out
    
    
    # Run a ResNet module consisting of residual blocks 
    def densenet121_module(self, input_data, is_training, layers_num=None, growth_rate=None):
        
        if not layers_num:
            layers_num = self.layers_num
        if not growth_rate:
            growth_rate=self.growth_rate
        filter_num =  self.filter_num
            
        for index, block in enumerate(range(self.dense_block_num)):
            if index == 0:
                with tf.variable_scope('module' + str(index+1)):
                    out, filter_num = self.dense_block(input_data, index + 1, layers_num[index], filter_num, growth_rate, is_training)
            else:
                with tf.variable_scope('module' + str(index+1)):
                    out, filter_num = self.dense_block(out, index + 1, layers_num[index], filter_num, growth_rate, is_training) 
               
            if index < self.dense_block_num-1:  
                out = self.transition_block(out, index+1, filter_num, is_training, compression=self.compression, dropout_rate=self.dropout_rate, weight_decay=self.weight_decay)
                filter_num = int(filter_num * self.compression)
            else:
                out = net._batch_norm(out, 'convolution'+str(self.dense_block_num)+'_blk_bn', is_training)
                out = tf.nn.relu(out, name = 'relu'+str(self.dense_block_num)+'_blk')
            
        return out
    

    
if __name__=='__main__':    
    parser = get_parser()
    args = parser.parse_args()
    
    # Prepare the training dataset
    dictionary = build_index()
    file_names, file_labels = prepare_dataset.dataset_iterator(args.image_path, args.csv_path, args.train_file_name, dictionary=dictionary)
    train_batch, train_iterator = prepare_dataset.get_data(file_names, file_labels, args.batch_size)   

    classes = len(dictionary)
    x_train = tf.placeholder(tf.float32, [None, 224, 224, 3])
    y_train = tf.placeholder(tf.int32, [None, classes])
    is_training = tf.placeholder(tf.bool)
    tf.summary.image("training_input_image", x_train, max_outputs=20)

    
    # Prepare the validation dataset
    file_names, file_labels = prepare_dataset.dataset_iterator(args.image_path, args.csv_path, args.val_file_name, dictionary=dictionary)  
    val_batch, val_iterator = prepare_dataset.get_data(file_names, file_labels, len(file_names))

    # Build the Graph
    net = DenseNet121(classes=classes)
    with tf.variable_scope("FirstStageFeature"):    
        out_1 = net._conv2d(x_train, [7,7,3,16],[16], [1, 2, 2, 1], 'convolution7x7', is_training, padding='SAME')
        #the paper uses 3*3 kernel
        out_2 = tf.nn.max_pool(value=out_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        #print(x_train,out_1,out_2)
    with tf.variable_scope("DenseNetBlocks"):
        out_3 = net.densenet121_module(out_2, is_training)
        #print(out_3)       
    with tf.variable_scope("Pooling"):
        out_4 = tf.nn.pool(out_3, window_shape=[7, 7], pooling_type='AVG', padding="VALID")
        #print(out_4)
    with tf.variable_scope("FullingConnected"):
        #dim = tf.reduce_prod(tf.shape(out_4)[1:])
        out_5 = tf.reshape(out_4, [-1, out_4.get_shape()[1]*out_4.get_shape()[2]*out_4.get_shape()[3]])
        #print(out_5)
        y_pred = net._fcl(out_5, [out_5.get_shape().as_list()[-1], classes], classes, 'fc5', need_relu=False)


        # Define the loss function and optimizer
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_train, logits=y_pred))
        optimizer = tf.train.AdamOptimizer(args.learning_rate)  
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
             train_op = optimizer.minimize(cross_entropy)
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
        print('begin session')
        train_writer = tf.summary.FileWriter('./train', sess.graph)
        test_writer = tf.summary.FileWriter('./test', sess.graph)
              
        sess.run(tf.global_variables_initializer())
        sess.run(train_iterator.initializer)
        sess.run(val_iterator.initializer)
        for epoch_number in tqdm(range(args.number_epochs)):
            summary, _, loss_value = sess.run([merged, train_op, cross_entropy], feed_dict={is_training: True, x_train: train_batch[0].eval(session=sess), y_train: tf.reshape(train_batch[1], [-1, train_batch[1].get_shape().as_list()[-1]]).eval(session=sess)})              
            train_writer.add_summary(summary, epoch_number)
            print("Loss at iteration {} : {}".format(epoch_number, loss_value))

            # Run the model on the test data for validation
            if epoch_number % args.test_frequency == 0:                
                summary, acc = sess.run([merged, accuracy], feed_dict={is_training : False,
                x_train: val_batch[0].eval(session=sess), y_train: tf.reshape(val_batch[1], [-1, val_batch[1].get_shape().as_list()[-1]]).eval(session=sess)})
                print("Accuracy at iteration {} : {}".format(epoch_number, acc))
                test_writer.add_summary(summary, epoch_number)   
    
    
    
    
    
    
  
