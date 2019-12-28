import tensorflow as tf


def make_rnn_cell(rnn_layer_sizes,
                  dropout_keep_prob=1.0,
                  attn_length=0,
                  base_cell=tf.contrib.rnn.BasicLSTMCell):
    """
    Makes a RNN cell from the given hyperparameters.
    :param rnn_layer_sizes: A list of integer sizes (in units) for each layer of the RNN.
    :param dropout_keep_prob: The float probability to keep the output of any given sub-cell.
    :param attn_length: The size of the attention vector.
    :param base_cell: The base tf.contrib.rnn.RNNCell to use for sub-cells.
    :return: A tf.contrib.rnn.MultiRNNCell based on the given hyperparameters.
    """
    cells = []
    for num_units in rnn_layer_sizes:
        cell = base_cell(num_units)
        if attn_length and not cells:
            # Add attention wrapper to first layer.
            cell = tf.contrib.rnn.AttentionCellWrapper(cell, attn_length, state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
        cells.append(cell)

    cell = tf.contrib.rnn.MultiRNNCell(cells)

    return cell


def weight_bias(W_shape, b_shape, bias_init=0.1):
    W = tf.get_variable(name="weight1", shape=W_shape, dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.1))
    b = tf.get_variable(name="bias1", shape=b_shape, dtype=tf.float32,
                        initializer=tf.constant_initializer(bias_init))
    return W, b


def highway_network(x, size, carry_bias=-1, scope="enc"):
    W, b = weight_bias([size, size], [size])

    with tf.variable_scope("transform_gate{}".format(scope)):
        W_T, b_T = weight_bias([size, size], [size], bias_init=carry_bias)

    T = tf.sigmoid(tf.matmul(x, W_T) + b_T, name="transform_gate")
    H = tf.nn.sigmoid(tf.matmul(x, W) + b, name="activation")
    C = tf.subtract(1.0, T, name="carry_gate")      # 1-T

    y = tf.add(tf.multiply(H, T), tf.multiply(x, C), "y")

    return y
