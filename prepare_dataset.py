import tensorflow as tf
import pandas as pd
import numpy as np

def parse_function(filename, label):
    """ Reads an image from a file, decodes it into a dense tensor, and resizes it
    to a fixed shape """
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, [224, 224])
    image_scaled = tf.image.per_image_standardization(image_resized)
    return image_scaled, label


def dataset_iterator(image_path, csv_path, filename, image_prefix ='dg_wiki_uganda_1000x1000_',dictionary=None):
    """ This function returns an iterator to the dataset """
    # Save all the file names and labels
    file_names = []
    file_labels = []
    train_filenames = pd.read_csv('{}/{}'.format(csv_path, filename), delimiter=',', engine='python')
    for index, file_information in train_filenames.iterrows():
        image_name = image_prefix + str(file_information[0]) + '.jpeg'
        file_names.append('{}/{}'.format(image_path, image_name))
        if dictionary != None:
            file_labels.append(tf.convert_to_tensor(dictionary[file_information[2]]))
        else:
            file_labels.append(file_information[2])
    return file_names, file_labels
