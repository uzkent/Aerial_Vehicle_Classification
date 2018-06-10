import tensorflow as tf

def parse_function(filename, label):
    """ Reads an image from a file, decodes it into a dense tensor, and resizes it
    to a fixed shape """
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, [56, 56])
    image_scaled = tf.image.per_image_standardization(image_resized)
    return image_scaled, label

def dataset_iterator(full_path, filenames):
    """ This function returns an iterator to the dataset """
    # Save all the file names and labels
    file_names = []
    file_labels = []
    for file_information in filenames:
        file_names.append('{}/{}'.format(full_path, file_information[0]))
        file_labels.append(file_information[1])

    file_names = tf.constant(file_names)
    file_labels = tf.one_hot(file_labels, 2)

    return file_names, file_labels
