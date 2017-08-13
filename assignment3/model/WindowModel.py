import tensorflow as tf
from model.NERModel import NERModel


class WindowModel(NERModel):
    """
    Implements a feedforward neural network with an embedding layer and
    single hidden layer.
    This network will predict what label (e.g. PER) should be given to a
    given token (e.g. Manning) by  using a featurized window around the token.
    """
    def __init__(self, helper, config, pretrained_embeddings, report=None):
        super(WindowModel, self).__init__(helper, config, report)
        self.pretrained_embeddings = pretrained_embeddings

        # Defining placeholders.
        self.input_placeholder = None
        self.labels_placeholder = None
        self.dropout_placeholder = None

        self.build()

    def make_windowed_data(self, data, start, end, window_size=1):
        """Uses the input sequences in @data to construct new windowed data points.

        TODO: In the code below, construct a window from each word in the
        input sentence by concatenating the words @window_size to the left
        and @window_size to the right to the word. Finally, add this new
        window data point and its label. to windowed_data.

        Args:
            data: is a list of (sentence, labels) tuples. @sentence is a list
                containing the words in the sentence and @label is a list of
                output labels. Each word is itself a list of
                @n_features features. For example, the sentence "Chris
                Manning is amazing" and labels "PER PER O O" would become
                ([[1,9], [2,9], [3,8], [4,8]], [1, 1, 4, 4]). Here "Chris"
                the word has been featurized as "[1, 9]", and "[1, 1, 4, 4]"
                is the list of labels.
            start: the featurized `start' token to be used for windows at the very
                beginning of the sentence.
            end: the featurized `end' token to be used for windows at the very
                end of the sentence.
            window_size: the length of the window to construct.
        Returns:
            a new list of data points, corresponding to each window in the
            sentence. Each data point consists of a list of
            @n_window_features features (corresponding to words from the
            window) to be used in the sentence and its NER label.
            If start=[5,8] and end=[6,8], the above example should return
            the list
            [([5, 8, 1, 9, 2, 9], 1),
             ([1, 9, 2, 9, 3, 8], 1),
             ...
             ([3, 8, 4, 8, 6, 8],1)
             ]
        """
        windowed_data = []
        start_padding = [start for _ in range(window_size)]
        end_padding = [end for _ in range(window_size)]
        for (sentence, labels) in data:
            ### YOUR CODE HERE (5-20 lines)
            T = len(labels)
            sentence = start_padding + sentence + end_padding
            for t in range(window_size, window_size + T):
                window = []
                for sent in sentence[t - window_size:t + window_size + 1]:
                    window.extend(sent)
                windowed_data.append((window, labels[t - window_size]))
        ### END YOUR CODE
        return windowed_data

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of  shape (None, n_window_features), type tf.int32
        labels_placeholder: Labels placeholder tensor of shape (None,), type tf.int32
        dropout_placeholder: Dropout value placeholder (scalar), type tf.float32

        Add these placeholders to self as the instance variables
            self.input_placeholder
            self.labels_placeholder
            self.dropout_placeholder

        (Don't change the variable names)
        """
        ### YOUR CODE HERE (~3-5 lines)
        self.input_placeholder = tf.placeholder(dtype=tf.int32, shape=(None, self.config.n_window_features))
        self.labels_placeholder = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.dropout_placeholder = tf.placeholder(dtype=tf.float32)
        ### END YOUR CODE

    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=1):
        """Creates the feed_dict for the model.
        A feed_dict takes the form of:
        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.
        Hint: When an argument is None, don't add it to the feed_dict.

        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        ### YOUR CODE HERE (~5-10 lines)
        feed_dict = {
            self.input_placeholder: inputs_batch,
            self.dropout_placeholder: dropout
        }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch

        ### END YOUR CODE
        return feed_dict

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:
            - Creates an embedding tensor and initializes it with self.pretrained_embeddings.
            - Uses the input_placeholder to index into the embeddings tensor, resulting in a
              tensor of shape (None, n_window_features, embedding_size).
            - Concatenates the embeddings by reshaping the embeddings tensor to shape
              (None, n_window_features * embedding_size).

        Hint: You might find tf.nn.embedding_lookup useful.
        Hint: You can use tf.reshape to concatenate the vectors. See following link to understand
            what -1 in a shape means.
            https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#reshape.
        Returns:
            embeddings: tf.Tensor of shape (None, n_window_features*embed_size)
        """
        ### YOUR CODE HERE (!3-5 lines)
        embeddings = tf.Variable(initial_value= self.pretrained_embeddings)
        embeddings = tf.nn.embedding_lookup(embeddings, ids=self.input_placeholder)
        embeddings = tf.reshape(embeddings, shape=(-1, self.config.n_window_features*self.config.embed_size))
        ### END YOUR CODE
        return embeddings


    def add_prediction_op(self):
        """Adds the 1-hidden-layer NN:
            h = Relu(xW + b1)
            h_drop = Dropout(h, dropout_rate)
            pred = h_drop U + b2

        Recall that we are not applying a softmax to pred. The softmax will instead be done in
        the add_loss_op function, which improves efficiency because we can use
        tf.nn.softmax_cross_entropy_with_logits

        When creating a new variable, use the tf.get_variable function
        because it lets us specify an initializer.

        Use tf.contrib.layers.xavier_initializer to initialize matrices.
        This is TensorFlow's implementation of the Xavier initialization
        trick we used in last assignment.

        Note: tf.nn.dropout takes the keep probability (1 - p_drop) as an argument.
            The keep probability should be set to the value of dropout_rate.

        Returns:
            pred: tf.Tensor of shape (batch_size, n_classes)
        """
        n_window_features = self.config.n_window_features
        embed_size = self.config.embed_size
        hidden_size = self.config.hidden_size
        n_classes = self.config.n_classes

        x = self.add_embedding()
        dropout_rate = self.dropout_placeholder
        ### YOUR CODE HERE (~10-20 lines)
        W = tf.get_variable(name="W", shape=(n_window_features*embed_size, hidden_size), dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        U = tf.get_variable(name="U", shape=(hidden_size, n_classes), dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable(name="b1", initializer=tf.zeros(shape=(hidden_size,), dtype=tf.float32))
        b2 = tf.get_variable(name="b2", initializer=tf.zeros(shape=(n_classes,), dtype=tf.float32))


        h = tf.nn.relu(tf.matmul(x, W) + b1)
        h_drop = tf.nn.dropout(h, dropout_rate)
        pred = tf.matmul(h_drop, U) + b2
        ### END YOUR CODE
        return pred

    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.
        In this case we are using cross entropy loss.
        The loss should be averaged over all examples in the current minibatch.

        Remember that you can use tf.nn.sparse_softmax_cross_entropy_with_logits to simplify your
        implementation. You might find tf.reduce_mean useful.
        Args:
            pred: A tensor of shape (batch_size, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        ### YOUR CODE HERE (~2-5 lines)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred, self.labels_placeholder))
        ### END YOUR CODE
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Use tf.train.AdamOptimizer for this model.
        Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        ### YOUR CODE HERE (~1-2 lines)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        train_op = optimizer.minimize(loss)
        ### END YOUR CODE
        return train_op

    def preprocess_sequence_data(self, examples):
        return self.make_windowed_data(examples, start=self.helper.START, end=self.helper.END, window_size=self.config.window_size)

    def consolidate_predictions(self, examples_raw, examples, preds):
        """Batch the predictions into groups of sentence length.
        """
        ret = []
        #pdb.set_trace()
        i = 0
        for sentence, labels in examples_raw:
            labels_ = preds[i:i+len(sentence)]
            i += len(sentence)
            ret.append([sentence, labels, labels_])
        return ret

    def predict_on_batch(self, sess, inputs_batch):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        feed = self.create_feed_dict(inputs_batch)
        predictions = sess.run(tf.argmax(self.pred, axis=1), feed_dict=feed)
        return predictions

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch,
                                     dropout=self.config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss


