import random
import re

import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_20newsgroups

classes = 20

embedding_size = 75  # Words are represented by vectors of length 75, initially with random values, but eventually learned
filter_sizes = (1, 2, 3)
num_filters_per_size = 150
stride = (1, 1, 1, 1)
max_document_length_cap = 500  # To speed training, enforce a max document size

learning_rate = 0.12
keep_rate = 0.5  # Percentage to keep when doing dropout
l2_lambda = 1e-4

training_epochs = 10
batch_size = 50

num_fc_layers = 1  # Includes the final output layer

keep_alphanumeric = re.compile('[^ \w]+', re.UNICODE)  # Used to remove non-alphanumeric characters from the input data


train_data_raw = fetch_20newsgroups(subset='train', shuffle=True, remove=('headers', 'footers', 'quotes'))
test_data_raw = fetch_20newsgroups(subset='test', shuffle=True, remove=('headers', 'footers', 'quotes'))


def get_network(input_tensor, total_words, max_document_length):
    num_convolutions = num_filters_per_size * len(filter_sizes)

    # Input/embedding layer (pull word vector representation by its id)
    embedding_weights = tf.get_variable('ew', [total_words, embedding_size], initializer=tf.random_uniform_initializer,
                                        regularizer=tf.nn.l2_loss)
    layer = tf.expand_dims(tf.nn.embedding_lookup(embedding_weights, input_tensor), -1)

    # Single convolution + max pooling layer
    layers = []
    for filter_size in filter_sizes:
        layers.append(get_pooling_layer(get_convolution_layer(layer, filter_size), max_document_length, filter_size))
    # Concatenate convolution layers along the 4th dimension
    layer = tf.concat(layers, 3)

    # Reshape for fully connected format and perform dropout
    layer = tf.reshape(layer, (-1, num_convolutions))
    layer = tf.nn.dropout(layer, keep_rate)

    # Intermediate fully connected layers
    for i in range(num_fc_layers - 1):
        w = tf.get_variable('fcw{}'.format(i), [num_convolutions, num_convolutions], initializer=tf.random_normal_initializer,
                            regularizer=tf.nn.l2_loss)
        b = tf.get_variable('fcb{}'.format(i), [num_convolutions], initializer=tf.random_normal_initializer)
        layer = tf.nn.xw_plus_b(layer, w, b)

    # Output layer
    w = tf.get_variable('ow', [num_convolutions, classes], initializer=tf.random_normal_initializer, regularizer=tf.nn.l2_loss)
    b = tf.get_variable('ob', [classes], initializer=tf.random_normal_initializer)
    return tf.nn.xw_plus_b(layer, w, b)


def get_convolution_layer(input_layer, filter_size):
    w = tf.get_variable('cw{}'.format(filter_size), [filter_size, embedding_size, 1, num_filters_per_size],
                        initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss)
    b = tf.get_variable('cb{}'.format(filter_size), [num_filters_per_size], initializer=tf.random_normal_initializer)
    layer = tf.nn.conv2d(input_layer, w, stride, 'VALID')
    return tf.nn.leaky_relu(tf.nn.bias_add(layer, b))


def get_pooling_layer(input_layer, max_document_length, filter_size):
    return tf.nn.max_pool(input_layer, [1, max_document_length - filter_size + 1, 1, 1], stride, 'VALID')


def process_data(train_data, test_data):
    max_document_length = 0
    total_words = 1  # Start at 1 b/c pad word index is 0
    train_skips = []
    test_skips = []
    words = {}

    # Default train/test split is 60/40, make that 80/20 instead
    more_train_data, test_data.data = np.array_split(test_data.data, 2)
    more_train_target, test_data.target = np.array_split(test_data.target, 2)
    np.concatenate((train_data.data, more_train_data))
    np.concatenate((train_data.target, more_train_target))

    # Assign unique integers to each word in the train + test data
    for i, text in enumerate(train_data.data[:]):
        # Split the document into non-empty alphanumeric words
        words_in_text = [word for word in keep_alphanumeric.sub('', text).split(' ') if word]

        # Discard empty data
        if not words_in_text:
            train_skips.append(i)
            continue

        # Truncate lengthy documents
        if len(words_in_text) > max_document_length_cap:
            words_in_text = words_in_text[:max_document_length_cap]

        # Continuously track the largest document
        max_document_length = max(max_document_length, len(words_in_text))

        # Assign a unique integer id to any new words
        for word in words_in_text:
            if words.setdefault(word.lower(), total_words) == total_words:
                total_words += 1

    # Same as above, except with test data
    for i, text in enumerate(test_data.data[:]):
        words_in_text = [word for word in keep_alphanumeric.sub('', text).split(' ') if word]

        if not words_in_text:
            test_skips.append(i)
            continue

        if len(words_in_text) > max_document_length_cap:
            words_in_text = words_in_text[:max_document_length_cap]

        max_document_length = max(max_document_length, len(words_in_text))

        for word in words_in_text:
            if words.setdefault(word.lower(), total_words) == total_words:
                total_words += 1

    train_input = []
    train_output = []
    test_input = []
    test_output = []

    # Now, format data for input + output from the network
    # Uses bag-of-words model for input and one hot encoding for output
    for i, text in enumerate(train_data.data):
        if i in train_skips:
            continue

        # Split the document into non-empty alphanumeric words
        words_in_text = [word for word in keep_alphanumeric.sub('', text).split(' ') if word]

        # Truncate lengthy documents
        if len(words_in_text) > max_document_length_cap:
            words_in_text = words_in_text[:max_document_length_cap]

        # Create network inputs using the unique integer ids
        input_layer = np.zeros(max_document_length, dtype=int)
        for j, word in enumerate(words_in_text):
            input_layer[j] = words[word.lower()]
        train_input.append(input_layer)

    for i, category in enumerate(train_data.target):
        # Create network outputs using the target class ids
        if i in train_skips:
            continue
        output_layer = np.zeros((classes,), dtype=float)
        output_layer[category] = 1.0
        train_output.append(output_layer)

    # Same as above, except with test data
    for i, text in enumerate(test_data.data):
        if i in test_skips:
            continue

        words_in_text = [word for word in keep_alphanumeric.sub('', text).split(' ') if word]

        if len(words_in_text) > max_document_length_cap:
            words_in_text = words_in_text[:max_document_length_cap]

        input_layer = np.zeros(max_document_length, dtype=int)
        for j, word in enumerate(words_in_text):
            input_layer[j] = words[word.lower()]
        test_input.append(input_layer)

    for i, category in enumerate(test_data.target):
        if i in test_skips:
            continue
        output_layer = np.zeros((classes,), dtype=float)
        output_layer[category] = 1.0
        test_output.append(output_layer)

    return total_words, max_document_length, train_input, train_output, test_input, test_output


def get_batch(input_data, output_data):
    # Shuffle the data
    combined = list(zip(input_data, output_data))
    random.shuffle(combined)
    input_data, output_data = zip(*combined)

    # Yield data in mini-batches
    for i in range(len(input_data) // batch_size + 1):
        yield input_data[i * batch_size:i * batch_size + batch_size]
        yield output_data[i * batch_size:i * batch_size + batch_size]


def main():
    total_words, max_document_length, train_input, train_output, test_input, test_output = process_data(train_data_raw, test_data_raw)

    num_train_batches = len(train_input) // batch_size
    num_test_batches = len(test_input) // batch_size

    # Create input and output of the network
    input_tensor = tf.placeholder(tf.int32, [None, max_document_length], name="input")
    output_tensor = tf.placeholder(tf.float32, [None, classes], name="output")

    # Construct the network
    prediction = get_network(input_tensor, total_words, max_document_length)

    # Define a loss measurement
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=output_tensor))
    loss += l2_lambda * sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))  # Add l2 regularization loss

    # Create an optimizer to minimize loss
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # Create an accuracy tester
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for epoch in range(training_epochs):
            # Do training over all mini-batches
            train_batch_generator = get_batch(train_input, train_output)
            for i in range(num_train_batches):
                loss_amount, _ = session.run(fetches=[loss, optimizer], feed_dict={input_tensor: next(train_batch_generator),
                                                                                   output_tensor: next(train_batch_generator)})

            # Test the test accuracy after each epoch
            test_batch_generator = get_batch(test_input, test_output)
            test_accuracy = 0
            for i in range(num_test_batches):
                test_accuracy += accuracy.eval({input_tensor: next(test_batch_generator), output_tensor: next(test_batch_generator)})
            print('Epoch:', epoch, 'Test Accuracy:', test_accuracy / num_test_batches)

        # Test the final training accuracy
        train_batch_generator = get_batch(train_input, train_output)
        train_accuracy = 0
        for i in range(num_train_batches):
            train_accuracy += accuracy.eval({input_tensor: next(train_batch_generator), output_tensor: next(train_batch_generator)})
        print('Train Accuracy:', train_accuracy / num_train_batches)


if __name__ == '__main__':
    main()
