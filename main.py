import pickle

import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_20newsgroups

classes = 20

learning_rate = 0.1
training_epochs = 3
batch_size = 290  # Set to maximum size that will run on my Macbook Pro GPU, you can make it higher if you have more GPU memory
nodes_per_layer = 10
layers = 3  # Including the input layer

config_limit_gpu_memory = 0.49  # Limits how much GPU memory is used so that the program doesn't crash

train_data_raw = fetch_20newsgroups(subset='train', shuffle=True, remove=('headers', 'footers', 'quotes'))
test_data_raw = fetch_20newsgroups(subset='test', shuffle=True, remove=('headers', 'footers', 'quotes'))


def get_network(input_tensor, weights, biases):
    # h1
    next_input = tf.add(tf.matmul(input_tensor, weights[0]), biases[0])
    layer_1 = tf.nn.relu(next_input)

    # h2
    next_input = tf.add(tf.matmul(layer_1, weights[1]), biases[1])
    layer_2 = tf.nn.relu(next_input)

    # Output
    return tf.matmul(layer_2, weights[2]) + biases[2]


def generate_weights(total_words):
    # Add the input layer
    weights = {0: tf.Variable(tf.random_normal([total_words, nodes_per_layer]))}

    # Add the hidden layers
    i = 0
    for i in range(layers - 1):
        weights[i + 1] = tf.Variable(tf.random_normal([nodes_per_layer, nodes_per_layer]))

    # Add the output layer
    weights[i + 1] = tf.Variable(tf.random_normal([nodes_per_layer, classes]))

    return weights


def generate_biases():
    biases = {}

    i = 0
    for i in range(layers - 1):
        biases[i] = tf.Variable(tf.random_normal([nodes_per_layer]))

    biases[i + 1] = tf.Variable(tf.random_normal([classes]))

    return biases


def process_data(train_data, test_data):
    # Try to deserialize processed data from file

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

    # Serialize data so we don't have to process next time
    # pickle.dump(total_words, open('total_words.pkl', 'wb'))
    # pickle.dump(train_input, open('train_input.pkl', 'wb'))
    # pickle.dump(train_output, open('train_output.pkl', 'wb'))
    # pickle.dump(test_input, open('test_input.pkl', 'wb'))
    # pickle.dump(test_output, open('test_output.pkl', 'wb'))

    return total_words, train_input, train_output, test_input, test_output


def get_batch(input_data, output_data):
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

    # Define loss and optimizer
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=output_tensor))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=output_tensor))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # Initializing the variables
    # init = tf.global_variables_initializer()
    init = tf.initialize_all_variables()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = config_limit_gpu_memory

    # Launch the graph
    with tf.Session(config=config) as session:
        session.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            for i in range(len(train_input) // batch_size):
                train_batch_generator = get_batch(train_input, train_output)
                loss_amount, _ = session.run(fetches=[loss, optimizer], feed_dict={input_tensor: next(train_batch_generator),
                                                                                   output_tensor: next(train_batch_generator)})
                print('Epoch: {} batch: {} loss: {}'.format(epoch, i, loss_amount))

        # Test model
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))

        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        print('Accuracy:', accuracy.eval({input_tensor: test_input, output_tensor: test_output}))

    # Serialize weight and bias values
    # pickle.dump(weights, open('weights.pkl', 'wb'))
    # pickle.dump(biases, open('biases.pkl', 'wb'))


if __name__ == '__main__':
    main()
