import random
import re

import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_20newsgroups

classes = 5

embedding_size = 150
filter_sizes = (2, 3, 4, 5, 6)
num_filters_per_size = 100
stride = (1, 1, 1, 1)

learning_rate = 0.1
keep_rate = 0.7
beta = 1e-4
relux_max = 1e15

training_epochs = 100
batch_size = 50  # Set to maximum size that will run on my Macbook Pro GPU, you can make it higher if you have more GPU memory
nodes_per_layer = 100
layers_scalar = 1  # Scales the number of nodes in each layer down
layers = 3  # Number of hidden layers

keep_alphanumeric = re.compile('[^ \w\']+', re.UNICODE)  # Used to remove all non-alphanumeric characters from the inputs

config_limit_gpu_memory = 0.49  # Limits how much GPU memory is used so that the program doesn't crash

train_data_raw = fetch_20newsgroups(subset='train', shuffle=True, remove=('headers', 'footers', 'quotes'),#)
                                    categories=('comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
                                                'comp.sys.mac.hardware', 'comp.windows.x',
                                                ))
                                    # categories=('alt.atheism', 'rec.autos', 'sci.crypt'))
test_data_raw = fetch_20newsgroups(subset='test', shuffle=True, remove=('headers', 'footers', 'quotes'),#)
                                   categories=('comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
                                               'comp.sys.mac.hardware', 'comp.windows.x',
                                               ))
                                   # categories=('alt.atheism', 'rec.autos', 'sci.crypt'))


def relux(x):
    return tf.minimum(tf.nn.leaky_relu(x), relux_max)


def get_network(input_tensor, total_words, max_document_length):
    # Input/embedding layer
    embedding_weights = tf.Variable(tf.random_uniform([total_words, embedding_size], -1, 1))
    layer = tf.expand_dims(tf.nn.embedding_lookup(embedding_weights, input_tensor), -1)

    # Hidden layers
    conv_pool_layers = []
    for i in range(1, layers + 1):
        for filter_size in filter_sizes:
            conv = get_convolution_layer(layer, filter_size)
            conv_pool_layers.append(get_pooling_layer(conv, max_document_length, filter_size))

    # Concatenate all conv_pool layers to perform dropout
    layer_before_dropout = tf.reshape(tf.concat(conv_pool_layers, 3), (-1, num_filters_per_size * len(filter_sizes)))
    layer = tf.nn.dropout(layer_before_dropout, keep_rate)

    # Output layer
    return tf.nn.xw_plus_b(layer, tf.Variable(tf.random_normal((num_filters_per_size * len(filter_sizes), classes))),
                           tf.Variable(tf.constant(0.1, shape=(classes,))))


def get_convolution_layer(input_layer, filter_size):
    conv = tf.nn.conv2d(input_layer, tf.Variable(tf.truncated_normal((filter_size, embedding_size, 1, num_filters_per_size))),
                        stride, 'VALID')
    return tf.nn.relu(tf.nn.bias_add(conv, tf.Variable(tf.constant(0.1, shape=(num_filters_per_size,)))))


def get_pooling_layer(input_layer, max_document_length, filter_size):
    return tf.nn.max_pool(input_layer, (1, max_document_length - filter_size + 1, 1, 1), stride, 'VALID')


def process_data(train_data, test_data):
    max_document_length = 0
    total_words = 0
    words = {}

    # Assign unique integers to each word
    for text in train_data.data:
        words_in_text = [word for word in keep_alphanumeric.sub(' ', text).split(' ') if word]
        max_document_length = max(max_document_length, len(words_in_text))
        for word in words_in_text:
            if words.setdefault(word.lower(), total_words) == total_words:
                total_words += 1

    for text in test_data.data:
        words_in_text = [word for word in keep_alphanumeric.sub(' ', text).split(' ') if word]
        max_document_length = max(max_document_length, len(words_in_text))
        for word in words_in_text:
            if words.setdefault(word.lower(), total_words) == total_words:
                total_words += 1

    train_input = []
    train_output = []
    test_input = []
    test_output = []

    # Morph data into usable inputs and outputs
    for text in train_data.data:
        input_layer = np.zeros(max_document_length, dtype=int)
        for i, word in enumerate([word for word in keep_alphanumeric.sub(' ', text).split(' ') if word]):
            input_layer[i] = words[word.lower()]
        train_input.append(input_layer)

    for category in train_data.target:
        output_layer = np.zeros((classes,), dtype=float)
        output_layer[category] = 1.0
        train_output.append(output_layer)

    for text in test_data.data:
        input_layer = np.zeros(max_document_length, dtype=int)
        for i, word in enumerate([word for word in keep_alphanumeric.sub(' ', text).split(' ') if word]):
            input_layer[i] = words[word.lower()]
        test_input.append(input_layer)

    for category in test_data.target:
        output_layer = np.zeros((classes,), dtype=float)
        output_layer[category] = 1.0
        test_output.append(output_layer)

    return total_words, max_document_length, train_input, train_output, test_input, test_output


def get_batch(input_data, output_data):
    # Shuffle the data for SGD and better training performance
    combined = list(zip(input_data, output_data))
    random.shuffle(combined)
    input_data, output_data = zip(*combined)

    for i in range(len(input_data) // batch_size + 1):
        yield input_data[i * batch_size:i * batch_size + batch_size]
        yield output_data[i * batch_size:i * batch_size + batch_size]


def main():
    total_words, max_document_length, train_input, train_output, test_input, test_output = process_data(train_data_raw, test_data_raw)

    print(max_document_length)

    input_tensor = tf.placeholder(tf.int32, [None, max_document_length], name="input")
    output_tensor = tf.placeholder(tf.float32, [None, classes], name="output")

    # Construct model
    prediction = get_network(input_tensor, total_words, max_document_length)

    # Define regularizer
    # regularization_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # regularizer = tf.nn.l2_loss(regularization_variables)

    # Define loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=output_tensor))
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=output_tensor) + beta * regularizer)
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=output_tensor))

    # Define the optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # Initializing the variables
    init = tf.global_variables_initializer()
    # init = tf.initialize_all_variables()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = config_limit_gpu_memory

    # Launch the graph
    with tf.Session(config=config) as session:
        session.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            train_batch_generator = get_batch(train_input, train_output)
            for i in range(len(train_input) // batch_size):
                loss_amount, _ = session.run(fetches=[loss, optimizer], feed_dict={input_tensor: next(train_batch_generator),
                                                                                   output_tensor: next(train_batch_generator)})
                # print('Epoch: {} batch: {} loss: {}'.format(epoch, i, loss_amount))

            if epoch % 10 == 0:
                correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

                test_batch_generator = get_batch(test_input, test_output)
                test_accuracy = 0
                for i in range(len(test_input) // batch_size):
                    test_accuracy += accuracy.eval({input_tensor: next(test_batch_generator), output_tensor: next(test_batch_generator)})
                print('Test Accuracy:', test_accuracy / (len(test_input) // batch_size))

        # Test model
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))

        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        # Calculate the average accuracy from mini batches, which are necessary because of limited GPU memory
        test_batch_generator = get_batch(test_input, test_output)
        test_accuracy = 0
        for i in range(len(test_input) // batch_size):
            test_accuracy += accuracy.eval({input_tensor: next(test_batch_generator), output_tensor: next(test_batch_generator)})

        train_batch_generator = get_batch(train_input, train_output)
        train_accuracy = 0
        for i in range(len(train_input) // batch_size):
            train_accuracy += accuracy.eval({input_tensor: next(train_batch_generator), output_tensor: next(train_batch_generator)})

        print('Test Accuracy:', test_accuracy / (len(test_input) // batch_size))
        print('Train Accuracy:', train_accuracy / (len(train_input) // batch_size))


if __name__ == '__main__':
    main()
