import tensorflow as tf

def parse_function(filename, label):
    """ Reads an image from a file, decodes it into a dense tensor, and resizes it
    to a fixed shape """
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, [56, 56])
    return image_resized, label

def dataset_iterator(full_path, filenames):
    """ This function returns an iterator to the dataset """
    # Save all the file names and labels
    file_names = []
    file_labels = []
    for file_information in filenames:
        file_names.append('{}/{}'.format(full_path, file_information[0]))
        # Convert them to hot labels
        if file_information[1] == 0:
            file_labels.append([1, 0])
        else:
            file_labels.append([0, 1])

    file_names = tf.constant(file_names)
    file_labels = tf.constant(file_labels)

    return file_names, file_labels
