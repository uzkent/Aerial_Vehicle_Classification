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
    if not dictionary:
        print('Please input Category dictionary')
        return
    file_names = []
    file_labels = []
    train_filenames = pd.read_csv('{}/{}'.format(csv_path, filename), delimiter=',', engine='python')
    for index, file_information in train_filenames.iterrows():      
        if file_information[2] not in dictionary.keys():
            print('Warning: {} not in CATEGORY set!'.format(file_information[2]))
            continue
        image_name = image_prefix + str(file_information[0]) + '.jpeg'
        one_hot_vector = np.zeros((1, len(dictionary)))
        one_hot_vector[0, dictionary[file_information[2]]]=1 
        one_hot_vector=tf.convert_to_tensor(one_hot_vector)           
        file_labels.append(one_hot_vector)
        file_names.append('{}/{}'.format(image_path, image_name))
        if not file_names:
            print('Warning: empty dataset')
    return file_names, file_labels

def get_data(file_names, file_labels, batch_size):
    """ This function returns a pointer to iterate over the batches of data """
    train_dataset = tf.data.Dataset.from_tensor_slices((file_names, file_labels))
    train_dataset = train_dataset.map(parse_function)
    train_dataset = train_dataset.repeat(200)
    train_dataset = train_dataset.shuffle(buffer_size=1000)
    train_batched_dataset = train_dataset.batch(batch_size)
    train_iterator = train_batched_dataset.make_initializable_iterator()
    return train_iterator.get_next(), train_iterator
