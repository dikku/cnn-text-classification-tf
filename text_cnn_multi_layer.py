import tensorflow as tf
import numpy as np


class TextCNNMultiLayer(object):
    """
    A Deep CNN for text classification.
    Uses an embedding layer, followed by n convolutional layers, a flat max-pooling layer, a FC layer
    and a softmax layer.
    Xavier initialization done on weights in all layers

        * Extendable to other deep convolutional architectures for intent/sentiment classification
        * TODO: Use static embeddings
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes,
            num_filters, num_hidden, use_non_linearity=True, l2_reg_lambda_embed=0.0, l2_reg_lambda_fc=0.0):

        assert len(num_filters) == len(filter_sizes), 'Filter sizes and number of filters must match'

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W_embed = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size],
                                  -np.sqrt(0.3/embedding_size), np.sqrt(0.3/embedding_size)),
                name="W_embed")
            # Look up embedding from the 1-hot representation
            self.embedded_chars = tf.nn.embedding_lookup(W_embed, self.input_x)
            # Make the tensor 4D
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, 2)

        # Create a convolution + maxpool layer for each filter size
        input = self.embedded_chars_expanded
        for i, filter_size_num_filter in enumerate(zip(filter_sizes, num_filters)):
            with tf.name_scope("layer-%s-conv" % (i+1)):
                # Convolution Layer i
                filter_size, num_filter = filter_size_num_filter
                input_shape = input.get_shape().as_list()
                filter_shape = [filter_size, input_shape[2], input_shape[3], num_filter]
                if i == 0:
                    std_dev = 2. / np.sqrt(filter_size)
                else:
                    std_dev = 2. / np.sqrt(np.prod(filter_shape[:-1]))
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=std_dev), name="W_%s" % (i+1))
                b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b_%s" % (i+1))
                conv = tf.nn.bias_add(tf.nn.conv2d(
                    input,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv-%s" % (i + 1)), b)
                if use_non_linearity:
                    # Apply nonlinearity if requested
                    input = tf.nn.relu(conv, name="relu-%s" % (i + 1))
                else:
                    input = conv

        input_shape = input.get_shape().as_list()
        self.h_pool = tf.nn.max_pool(
            input,
            ksize=[1, input_shape[1], 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")

        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters[-1]])
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        W_fc = tf.get_variable(
                "W_fc",
                shape=[num_filters[-1], num_hidden],
                initializer=tf.contrib.layers.xavier_initializer())
        b_fc = tf.Variable(tf.constant(0.1, shape=[num_hidden]), name="b_fc")
        hidden = tf.matmul(self.h_drop, W_fc) + b_fc
        # Add dropout
        with tf.name_scope("dropout"):
            hidden_drop = tf.nn.dropout(hidden, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W_soft_max = tf.get_variable(
                "W_soft_max",
                shape=[num_hidden, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b_soft_max = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b_soft_max")
            if l2_reg_lambda_fc:
                l2_loss += l2_reg_lambda_fc * tf.nn.l2_loss(W_soft_max) + l2_reg_lambda_fc * tf.nn.l2_loss(W_fc)
            if l2_reg_lambda_embed:
                l2_loss += l2_reg_lambda_embed * tf.nn.l2_loss(W_embed)
            self.scores = tf.nn.xw_plus_b(hidden_drop, W_soft_max, b_soft_max, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
