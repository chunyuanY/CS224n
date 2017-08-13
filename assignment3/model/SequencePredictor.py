import tensorflow as tf

from model import Model
from q2_rnn_cell import RNNCell
from q3_gru_cell import GRUCell
from util import Progbar, minibatches


class SequencePredictor(Model):

    def __init__(self, config):
        self.config = config
        self.inputs_placeholder = None
        self.labels_placeholder = None
        self.grad_norm = None
        self.build()


    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors
        NOTE: You do not have to do anything here.
        """
        self.inputs_placeholder = tf.placeholder(tf.float32, shape=(None, self.config.max_length, 1), name="x")
        self.labels_placeholder = tf.placeholder(tf.float32, shape=(None, 1), name="y")

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        """Creates the feed_dict for the model.
        NOTE: You do not have to do anything here.
        """
        feed_dict = {
            self.inputs_placeholder: inputs_batch,
            }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def add_prediction_op(self):
        """Runs an rnn on the input using TensorFlows's
        @tf.nn.dynamic_rnn function, and returns the final state as a prediction.

        TODO: 
            - Call tf.nn.dynamic_rnn using @cell below. See:
              https://www.tensorflow.org/api_docs/python/nn/recurrent_neural_networks
            - Apply a sigmoid transformation on the final state to
              normalize the inputs between 0 and 1.

        Returns:
            preds: tf.Tensor of shape (batch_size, 1)
        """

        # Pick out the cell to use here.
        if self.config.cell == "rnn":
            cell = RNNCell(1, 1)
        elif self.config.cell == "gru":
            cell = GRUCell(1, 1)
        elif self.config.cell == "lstm":
            cell = tf.nn.rnn_cell.LSTMCell(1)
        else:
            raise ValueError("Unsupported cell type.")

        x = self.inputs_placeholder
        ### YOUR CODE HERE (~2-3 lines)
        ### END YOUR CODE

        return preds #state # preds

    def add_loss_op(self, preds):
        """Adds ops to compute the loss function.
        Here, we will use a simple l2 loss.

        Tips:
            - You may find the functions tf.reduce_mean and tf.l2_loss
              useful.

        Args:
            pred: A tensor of shape (batch_size, 1) containing the last
            state of the neural network.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        y = self.labels_placeholder

        ### YOUR CODE HERE (~1-2 lines)

        ### END YOUR CODE

        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        TODO:
            - Get the gradients for the loss from optimizer using
              optimizer.compute_gradients.
            - if self.clip_gradients is true, clip the global norm of
              the gradients using tf.clip_by_global_norm to self.config.max_grad_norm
            - Compute the resultant global norm of the gradients using
              tf.global_norm and save this global norm in self.grad_norm.
            - Finally, actually create the training operation by calling
              optimizer.apply_gradients.
        See: https://www.tensorflow.org/api_docs/python/train/gradient_clipping
        Args:
            loss: Loss tensor.
        Returns:
            train_op: The Op for training.
        """

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.lr)

        ### YOUR CODE HERE (~6-10 lines)

        # - Remember to clip gradients only if self.config.clip_gradients
        # is True.
        # - Remember to set self.grad_norm

        ### END YOUR CODE

        assert self.grad_norm is not None, "grad_norm was not set properly!"
        return train_op

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        """Perform one step of gradient descent on the provided batch of data.
        This version also returns the norm of gradients.
        """
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
        _, loss, grad_norm = sess.run([self.train_op, self.loss, self.grad_norm], feed_dict=feed)
        return loss, grad_norm

    def run_epoch(self, sess, train):
        prog = Progbar(target=1 + int(len(train) / self.config.batch_size))
        losses, grad_norms = [], []
        for i, batch in enumerate(minibatches(train, self.config.batch_size)):
            loss, grad_norm = self.train_on_batch(sess, *batch)
            losses.append(loss)
            grad_norms.append(grad_norm)
            prog.update(i + 1, [("train loss", loss)])

        return losses, grad_norms

    def fit(self, sess, train):
        losses, grad_norms = [], []
        for epoch in range(self.config.n_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            loss, grad_norm = self.run_epoch(sess, train)
            losses.append(loss)
            grad_norms.append(grad_norm)

        return losses, grad_norms


