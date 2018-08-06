import tensorflow as tf
import pdb

def get_data(file_names, file_labels, batch_size):
    """ This function returns a pointer to iterate over the batches of data """
    train_dataset = tf.data.Dataset.from_tensor_slices((file_names, file_labels))
    train_dataset = train_dataset.map(parse_function)
    train_dataset = train_dataset.repeat(2000)
    train_batched_dataset = train_dataset.batch(batch_size)
    train_iterator = train_batched_dataset.make_initializable_iterator()

    return train_iterator.get_next(), train_iterator

def parse_function(filename, label):
    """ Reads an image from a file, decodes it into a dense tensor, and resizes it
    to a fixed shape """
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, [56, 56])
    image_scaled = tf.image.per_image_standardization(image_resized)
    return image_scaled, label

def dataset_iterator(full_path, filenames, arr):
    """ This function returns an iterator to the dataset """
    # Save all the file names and labels
    file_names = []
    file_labels = []
    for index in range(len(filenames)):
        file_names.append('./{}/{}'.format(full_path, filenames[arr[index]][0].decode('UTF-8')))
        file_labels.append(filenames[arr[index]][1])

    file_names = tf.constant(file_names)
    file_labels = tf.one_hot(file_labels, 2)

    return file_names, file_labels
