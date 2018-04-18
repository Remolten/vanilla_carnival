import random
import re

import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_20newsgroups

classes = 5

learning_rate = 0.1
keep_rate = 0.7
beta = 1e-4
relux_max = 1e15

training_epochs = 100
batch_size = 50  # Set to maximum size that will run on my Macbook Pro GPU, you can make it higher if you have more GPU memory
nodes_per_layer = 100
layers_scalar = 1  # Scales the number of nodes in each layer down
layers = 3  # Number of hidden layers

keep_alphanumeric = re.compile('[\W_]+', re.UNICODE)  # Used to remove all non-alphanumeric characters from the inputs

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


def get_network(input_tensor, weights, biases):
    # Input layer
    next_input = tf.matmul(input_tensor, weights[0])
    layer = relux(next_input)

    # Hidden layers
    for i in range(1, layers + 1):
        next_input = tf.add(tf.matmul(layer, weights[i]), biases[i])
        layer_before_dropout = relux(next_input)
        layer = tf.nn.dropout(layer_before_dropout, keep_rate)

    # Output layer
    return tf.matmul(layer, weights[len(weights) - 1]) + biases[len(biases)]


def generate_weights(total_words):
    npl = nodes_per_layer

    # Add the input layer
    weights = {0: tf.Variable(tf.random_normal([total_words, npl]))}

    # Add the hidden layers
    i = 0
    for i in range(1, layers + 1):
        weights[i] = tf.Variable(tf.random_normal([max(npl, classes), max(int(npl * layers_scalar), classes)]))
        npl = int(npl * layers_scalar)

    # Add the output layer
    weights[i + 1] = tf.Variable(tf.random_normal([max(npl, classes), classes]))

    # Add all weights to the regularization collection, to do regularization later
    for weight in list(weights.values())[1:-1]:
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight)

    return weights


def generate_biases():
    npl = int(nodes_per_layer * layers_scalar)
    biases = {}

    i = 0
    for i in range(1, layers + 1):
        biases[i] = tf.Variable(tf.random_normal([max(npl, classes)]))
        npl = int(npl * layers_scalar)

    biases[i + 1] = tf.Variable(tf.random_normal([classes]))

    return biases


def process_data(train_data, test_data):
    # word_indexes = list(range(18232))  # Used to randomly assign the integer indexes to each word
    word_indexes = list(range(4734))  # Used to randomly assign the integer indexes to each word
    total_words = 0
    words = {}

    # Shuffle the indexes
    random.shuffle(word_indexes)

    # Assign indices to words
    for text in train_data.data:
        for word in keep_alphanumeric.sub('', text).split(' '):
            if words.setdefault(word.lower(), word_indexes[0]) == word_indexes[0]:
                word_indexes.pop(0)
                total_words += 1

    for text in test_data.data:
        for word in keep_alphanumeric.sub('', text).split(' '):
            if words.setdefault(word.lower(), word_indexes[0]) == word_indexes[0]:
                word_indexes.pop(0)
                total_words += 1

    print(total_words)
    train_input = []
    train_output = []
    test_input = []
    test_output = []

    # Morph data into usable inputs and outputs
    for text in train_data.data:
        input_layer = np.zeros(total_words, dtype=float)
        for word in keep_alphanumeric.sub('', text).split(' '):
            input_layer[words[word.lower()]] += 1
        # input_layer -= len(keep_alphanumeric.sub('', text).split(' ')) / total_words  # Subtract the mean
        train_input.append(input_layer)

    for category in train_data.target:
        output_layer = np.zeros((classes,), dtype=float)
        output_layer[category] = 1.0
        train_output.append(output_layer)

    for text in test_data.data:
        input_layer = np.zeros(total_words, dtype=float)
        for word in keep_alphanumeric.sub('', text).split(' '):
            input_layer[words[word.lower()]] += 1
        # input_layer -= len(keep_alphanumeric.sub('', text).split(' ')) / total_words  # Subtract the mean
        test_input.append(input_layer)

    for category in test_data.target:
        output_layer = np.zeros((classes,), dtype=float)
        output_layer[category] = 1.0
        test_output.append(output_layer)

    return total_words, train_input, train_output, test_input, test_output


def get_batch(input_data, output_data):
    # Shuffle the data for SGD and better training performance
    combined = list(zip(input_data, output_data))
    random.shuffle(combined)
    input_data, output_data = zip(*combined)

    for i in range(len(input_data) // batch_size + 1):
        yield input_data[i * batch_size:i * batch_size + batch_size]
        yield output_data[i * batch_size:i * batch_size + batch_size]


def main():
    total_words, train_input, train_output, test_input, test_output = process_data(train_data_raw, test_data_raw)

    weights = generate_weights(total_words)
    biases = generate_biases()

    input_tensor = tf.placeholder(tf.float32, [None, total_words], name="input")
    output_tensor = tf.placeholder(tf.float32, [None, classes], name="output")

    # Construct model
    prediction = get_network(input_tensor, weights, biases)

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
