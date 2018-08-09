import tensorflow as tf
import os
import pdb

def top_k_label_finder(original_label, similarity_df):
    """ This function find the top k labels that is mutually-inclusive to the label of interest """
    search_df = similarity_df[similarity_df['anchor'] == original_label]
    search_df['score_ranked'] = search_df['score'].rank(ascending=0)
    inclusive_labels_original = []
    for ind, row in search_df.iterrows():
        if row['score_ranked'] == 2 or row['score_ranked'] == 3 or row['score_ranked'] == 4:
           inclusive_labels_original.append(row['query'])
    return inclusive_labels_original
   
def label_mapper(labels_df, similarity_df, original_label):
    """ Maps the semantic label to a one-hot encoded labels """
    label_index = labels_df[labels_df['Tag'] == original_label].index[0] + 1
    inclusive_labels_original = top_k_label_finder(original_label, similarity_df)
    inclusive_labels_ind = []
    for inc_label in inclusive_labels_original:
        try:
            inclusive_labels_ind.append(labels_df[labels_df['Tag'] == inc_label].index[0] + 1)              
        except Exception as error:
            pdb.set_trace()
            print('Label Mapping Error', error)
    return label_index, inclusive_labels_ind

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
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize_images(image_decoded, [224, 224])
    image_scaled = tf.image.per_image_standardization(image_resized)
    return image_scaled, label

def dataset_iterator(full_path, filenames, arr, labels_df, similarity_df, threshold=500):
    """ This function returns an iterator to the dataset """
    file_names = []
    file_labels = []
    for index in range(len(filenames)):
        if not os.path.isfile('{}/{}{}.{}'.format(full_path, 'dg_wiki_uganda_1000x1000_', filenames['id'][arr[index]], 'jpeg')):
            continue
        file_names.append('{}/{}{}.{}'.format(full_path, 'dg_wiki_uganda_1000x1000_', filenames['id'][arr[index]], 'jpeg'))
        file_labels.append(label_mapper(labels_df, similarity_df, filenames['category'][arr[index]]))

    file_names = tf.constant(file_names)
    pdb.set_trace()
    np_labels = np.zeros((len(labels_df), np.int32))
    np_labels[np.array(inclusive_labels_ind)] = 1
    file_labels = tf.convert_to_tensor(np_labels)
    return file_names, file_labels
