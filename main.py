import random

import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_20newsgroups

classes = 20

learning_rate = 0.15
keep_rate = 0.8
beta = 1e-4
relux_max = 1e6

training_epochs = 10
batch_size = 100  # Set to maximum size that will run on my Macbook Pro GPU, you can make it higher if you have more GPU memory
nodes_per_layer = 100
layers = 10  # Number of hidden layers

config_limit_gpu_memory = 0.49  # Limits how much GPU memory is used so that the program doesn't crash

train_data_raw = fetch_20newsgroups(subset='train', shuffle=True, remove=('headers', 'footers', 'quotes'))
                                    # categories=('alt.atheism', 'rec.autos', 'sci.crypt'))
test_data_raw = fetch_20newsgroups(subset='test', shuffle=True, remove=('headers', 'footers', 'quotes'))
                                   # categories=('alt.atheism', 'rec.autos', 'sci.crypt'))


def relux(x):
    return tf.minimum(tf.nn.relu(x), relux_max)


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
    # Add the input layer
    weights = {0: tf.Variable(tf.random_normal([total_words, nodes_per_layer]))}

    # Add the hidden layers
    i = 0
    for i in range(1, layers + 1):
        weights[i] = tf.Variable(tf.random_normal([nodes_per_layer, nodes_per_layer]))

    # Add the output layer
    weights[i + 1] = tf.Variable(tf.random_normal([nodes_per_layer, classes]))

    # Add all weights to the regularization collection, to do regularization later
    for weight in list(weights.values())[1:-1]:
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight)

    return weights


def generate_biases():
    biases = {}

    i = 0
    for i in range(1, layers + 1):
        biases[i] = tf.Variable(tf.random_normal([nodes_per_layer]))

    biases[i + 1] = tf.Variable(tf.random_normal([classes]))

    return biases


def process_data(train_data, test_data):
    total_words = 0
    words = {}

    # Assign indices to words
    for text in train_data.data:
        for word in text.split(' '):
            if words.setdefault(word.lower(), total_words) == total_words:
                total_words += 1  # We added a new word, so increment the index

    for text in test_data.data:
        for word in text.split(' '):
            if words.setdefault(word.lower(), total_words) == total_words:
                total_words += 1  # We added a new word, so increment the index

    train_input = []
    train_output = []
    test_input = []
    test_output = []

    # Morph data into usable inputs and outputs
    for text in train_data.data:
        input_layer = np.zeros(total_words, dtype=float)
        for word in text.split(' '):
            input_layer[words[word.lower()]] += 1
        train_input.append(input_layer)

    for category in train_data.target:
        output_layer = np.zeros((classes,), dtype=float)
        output_layer[category] = 1.0
        train_output.append(output_layer)

    for text in test_data.data:
        input_layer = np.zeros(total_words, dtype=float)
        for word in text.split(' '):
            input_layer[words[word.lower()]] += 1
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
    rinput_data, routput_data = zip(*combined)

    for i in range(len(rinput_data) // batch_size + 1):
        yield rinput_data[i * batch_size:i * batch_size + batch_size]
        yield routput_data[i * batch_size:i * batch_size + batch_size]


def main():
    total_words, train_input, train_output, test_input, test_output = process_data(train_data_raw, test_data_raw)

    weights = generate_weights(total_words)
    biases = generate_biases()

    input_tensor = tf.placeholder(tf.float32, [None, total_words], name="input")
    output_tensor = tf.placeholder(tf.float32, [None, classes], name="output")

    # Construct model
    prediction = get_network(input_tensor, weights, biases)

    # Define regularizer
    regularization_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    regularizer = tf.nn.l2_loss(regularization_variables)

    # Define loss
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=output_tensor))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=output_tensor) + beta * regularizer)
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
                print('Epoch: {} batch: {} loss: {}'.format(epoch, i, loss_amount))

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
