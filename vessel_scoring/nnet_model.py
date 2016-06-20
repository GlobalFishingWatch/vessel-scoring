# Copyright 2016 SkyTruth
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Some of this code comes from Google Tensor flow demo:
# https://github.com/tensorflow/tensorflow/blob/r0.9/tensorflow/examples/tutorials/mnist/fully_connected_feed.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from vessel_scoring.utils import get_polynomial_cols
import numpy as np
import time


def leaky_relu(x, alpha=0.01):
    return tf.maximum(alpha*x,x)

def maxout(x, width):
    bsize = tf.shape(x)[0]
    return tf.reshape(
        tf.nn.max_pool(
            tf.reshape(x, (bsize, -1, 1, 1)),
            (1, width, 1, 1), (1, width, 1, 1), 'VALID'),
        (bsize, -1))

class NNetModel:
    LEARNING_RATE = 0.5
    MAX_EPOCHS = 20
    HIDDEN_1 = 1024
    HIDDEN_2 = 1024
    BATCH_SIZE = 128
    TRAIN_DIR = "dumps"
    DECAY_SCALE = 0.98

    N_WINDOWS = 6
    N_BASE_FEATURES = 3
    N_FEATURES = N_WINDOWS * N_BASE_FEATURES


    windows = ['10800', '1800', '21600', '3600', '43200', '86400']

    def __init__(self, **args):
        """
        windows - list of window sizes to use in features
        See RandomForestClassifier docs for other parameters.
        """
        self.ses = None

    def dump_arg_dict(self):
        raise NotImplementedError()

    def _make_features(self, X):
        x = np.transpose(get_polynomial_cols(X, self.windows))
        return (x.astype('float32') - self.mean) / self.std

    def predict_proba(self, X):
        X = self._make_features(X)
        y = np.zeros([len(X), 2], dtype='float32')
        #
        X1 = self.complete_batch(X)
        ds = self.DataSet(X1,None)
        chunks = []
        steps = len(X1) // self.BATCH_SIZE
        assert len(X1) % self.BATCH_SIZE == 0
        for step in range(steps):
            feed_dict = self.fill_test_dict(ds)

            chunks.append(self.sess.run(self.predictions, feed_dict=feed_dict))
        ps = np.concatenate(chunks)
        #
        y[:,1] = ps.reshape(-1)[:len(X)]
        y[:,0] = 1 - y[:,1]
        return y

    def fit(self, X, y):
        self.mean = 0
        self.std = 1
        X = self._make_features(X)
        self.mean = X.mean(axis=0, keepdims=True)
        self.std = X.mean(axis=0, keepdims=True)
        X = (X - self.mean) / self.std
        #
        n = len(X)
        n_train = int(self.DECAY_SCALE * n)
        inds = np.arange(n)
        np.random.shuffle(inds)
        #
        train_ds = self.DataSet(X[inds[:n_train]], y[inds[:n_train]])
        eval_ds = self.DataSet(X[inds[n_train:]], y[inds[n_train:]])
        self.run_training(train_ds, eval_ds)

        return self

    def placeholder_inputs(self, batch_size):
      """Generate placeholder variables to represent the input tensors.
      These placeholders are used as inputs by the rest of the model building
      code and will be fed from the downloaded data in the .run() loop, below.
      Args:
        batch_size: The batch size will be baked into both placeholders.
      Returns:
        featurse_placeholder: Features placeholder.
        labels_placeholder: Labels placeholder.
      """
      # Note that the shapes of the placeholders match the shapes of the full
      # image and label tensors, except the first dimension is now batch_size
      features_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                               self.N_FEATURES))
      labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size))
      return features_placeholder, labels_placeholder


    def fill_train_dict(self, data_set):
      """Fills the feed_dict for training the given step.
      A feed_dict takes the form of:
      feed_dict = {
          <placeholder>: <tensor of values to be passed for placeholder>,
          ....
      }
      Args:
        data_set: The set of features and labels, from input_data.read_data_sets()
      Returns:
        feed_dict: The feed dictionary mapping from placeholders to values.
      """
      # Create the feed_dict for the placeholders filled with the next
      # `batch size ` examples.
      features_feed, labels_feed = data_set.next_batch(self.BATCH_SIZE)
      feed_dict = {
          self.features_placeholder: features_feed,
          self.labels_placeholder: labels_feed,
      }
      return feed_dict

    def fill_test_dict(self, data_set):
      """Fills the feed_dict for training the given step.
      A feed_dict takes the form of:
      feed_dict = {
          <placeholder>: <tensor of values to be passed for placeholder>,
          ....
      }
      Args:
        data_set: The set of features and labels, from input_data.read_data_sets()
      Returns:
        feed_dict: The feed dictionary mapping from placeholders to values.
      """
      # Create the feed_dict for the placeholders filled with the next
      # `batch size ` examples.
      features_feed, labels_feed = data_set.next_batch(self.BATCH_SIZE)
      feed_dict = {
          self.features_placeholder: features_feed,
      }
      return feed_dict


    def do_eval(self, sess,
                eval_correct,
                features_placeholder,
                labels_placeholder,
                data_set,
                name):
      """Runs one evaluation against the full epoch of data.
      Args:
        sess: The session in which the model has been trained.
        eval_correct: The Tensor that returns the number of correct predictions.
        features_placeholder: The features placeholder.
        labels_placeholder: The labels placeholder.
        data_set: The set of features and labels to evaluate, from
          input_data.read_data_sets().
      """
      # And run one epoch of eval.
      true_count = 0  # Counts the number of correct predictions.
      steps_per_epoch = data_set.num_examples // self.BATCH_SIZE
      num_examples = steps_per_epoch * self.BATCH_SIZE
      for step in range(steps_per_epoch):
        feed_dict = self.fill_train_dict(data_set)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
      precision = true_count / num_examples
      print(name, ' %d / %d = %0.04f' %
            (true_count, num_examples, precision))

    def inference(self, features):
        """Build the model up to where it may be used for inference.
        Args:
        features: features placeholder, from inputs().
        hidden_units: Size of the hidden layers.
        Returns:
        softmax_linear: Output tensor with the computed logits.
        """


        batch_size = tf.shape(features)[0]
        # Hidden 1
        with tf.name_scope('hidden1'):
            weights = tf.Variable(
                tf.truncated_normal([self.N_FEATURES, self.HIDDEN_1],
                                    stddev=1.0 / np.sqrt(self.N_FEATURES)),
                name='weights')
            biases = tf.Variable(tf.zeros([self.HIDDEN_1]),
                                 name='biases')
            hidden1 = leaky_relu(tf.matmul(features, weights) + biases)
        # Dropout 1
        dropout1 = tf.nn.dropout(hidden1, 0.6)
        # Hidden 2
        with tf.name_scope('hidden2'):
            weights = tf.Variable(
                tf.truncated_normal([self.HIDDEN_1, self.HIDDEN_2],
                                    stddev=1.0 / np.sqrt(self.HIDDEN_1)),
                name='weights')
            biases = tf.Variable(tf.zeros([self.HIDDEN_2]),
                                 name='biases')
            hidden2 = leaky_relu((tf.matmul(dropout1, weights) + biases))
        # Dropout2
        dropout2 = tf.nn.dropout(hidden2, 0.6)
        # Linear
        with tf.name_scope('logit'):
            weights = tf.Variable(
                tf.truncated_normal([self.HIDDEN_2, 1],
                                    stddev=1.0 / np.sqrt(self.HIDDEN_2)),
                name='weights')
            biases = tf.Variable(tf.zeros([1]),
                                 name='biases')
            logits = tf.reshape(tf.matmul(dropout2, weights) + biases, (-1,))
        return logits



    def lossfunc(self, logits, labels):
      """Calculates the loss from the logits and the labels.
      Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size].
      Returns:
        loss: Loss tensor of type float.
      """
      cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
          logits, labels, name='xentropy')
      loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
      return loss


    def training(self, loss, learning_rate):
      """Sets up the training Ops.
      Creates a summarizer to track the loss over time in TensorBoard.
      Creates an optimizer and applies the gradients to all trainable variables.
      The Op returned by this function is what must be passed to the
      `sess.run()` call to cause the model to train.
      Args:
        loss: Loss tensor, from loss().
        learning_rate: The learning rate to use for gradient descent.
      Returns:
        train_op: The Op for training.
      """
      # Add a scalar summary for the snapshot loss.
      tf.scalar_summary(loss.op.name, loss)
      # Create the gradient descent optimizer with the given learning rate.
      optimizer = tf.train.GradientDescentOptimizer(learning_rate)
      # Create a variable to track the global step.
      global_step = tf.Variable(0, name='global_step', trainable=False)
      # Use the optimizer to apply the gradients that minimize the loss
      # (and also increment the global step counter) as a single training step.
      train_op = optimizer.minimize(loss, global_step=global_step)
      return train_op


    def evaluation(self, logits, labels):
      """Evaluate the quality of the logits at predicting the label.
      Args:
        logits: Logits tensor, float - [batch_size].
        labels: Labels tensor, float - [batch_size]
      Returns:
        A scalar int32 tensor with the number of examples (out of batch_size)
        that were predicted correctly.
      """
      # For a classifier model, we can use the in_top_k Op.
      # It returns a bool tensor with shape [batch_size] that is true for
      # the examples where the label is in the top k (here k=1)
      # of all logits for that example.
      correct = tf.equal(tf.round(tf.sigmoid(logits)), labels)
      # Return the number of true entries.
      return tf.reduce_sum(tf.cast(correct, tf.int32))



    class DataSet(object):

        def __init__(self,
                     features,
                     labels):
            """Construct a DataSet.
            """
            dtype = 'float32'

            assert labels is None or features.shape[0] == labels.shape[0], (
              'features.shape: %s labels.shape: %s' % (features.shape, labels.shape))
            self._num_examples = features.shape[0]
            self._features = features
            self._labels = labels
            self._epochs_completed = 0
            self._index_in_epoch = 0

        @property
        def features(self):
            return self._features

        @property
        def labels(self):
            return self._labels

        @property
        def num_examples(self):
            return self._num_examples

        @property
        def epochs_completed(self):
            return self._epochs_completed

        def next_batch(self, batch_size, fake_data=False):
            """Return the next `batch_size` examples from this data set."""
            start = self._index_in_epoch
            self._index_in_epoch += batch_size
            if self._index_in_epoch > self._num_examples:
              # Finished epoch
              self._epochs_completed += 1
              # Shuffle the data
              perm = np.arange(self._num_examples)
              np.random.shuffle(perm)
              self._features = self._features[perm]
              self._labels = None if (self._labels is None) else self._labels[perm]
              # Start next epoch
              start = 0
              self._index_in_epoch = batch_size
              assert batch_size <= self._num_examples
            end = self._index_in_epoch
            return (self._features[start:end],
                    None if (self._labels is None) else self._labels[start:end])



    def run_training(self, train_ds, eval_ds):
      """Train for a number of steps."""
      # Get the sets of features and labels for training, validation, and
      # test on .
    #   data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)

      # Tell TensorFlow that the model will be built into the default Graph.
      with tf.Graph().as_default():
        # Generate placeholders for the features and labels.
        self.features_placeholder, self.labels_placeholder = self.placeholder_inputs(
            self.BATCH_SIZE)

        # Build a Graph that computes predictions from the inference model.
        logits = self.inference(self.features_placeholder)

        # Build a final output prediction
        predictions = tf.nn.sigmoid(logits)

        # Add to the Graph the Ops for loss calculation.
        loss = self.lossfunc(logits, self.labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        learning_rate = tf.Variable(self.LEARNING_RATE, name="learning_rate")
        #
        train_op = self.training(loss, learning_rate)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = self.evaluation(logits, self.labels_placeholder)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Add the variable initializer Op.
        init = tf.initialize_all_variables()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.train.SummaryWriter(self.TRAIN_DIR, sess.graph)

        # And then after everything is built:

        # Run the Op to initialize the variables.
        sess.run(init)

        # Start the training loop.
        epoch = 0
        last_epoch = 0
        step = 0
        while epoch < self.MAX_EPOCHS:
          try:
              start_time = time.time()

              # Fill a feed dictionary with the actual set of features and labels
              # for this particular training step.
              feed_dict = self.fill_train_dict(train_ds)

              # Run one step of the model.  The return values are the activations
              # from the `train_op` (which is discarded) and the `loss` Op.  To
              # inspect the values of your Ops or variables, you may include them
              # in the list passed to sess.run() and the value tensors will be
              # returned in the tuple from the call.
              _, loss_value = sess.run([train_op, loss],
                                       feed_dict=feed_dict)

              duration = time.time() - start_time

              # Write the summaries and print an overview fairly often.
              if step % 100 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()


              epoch = (step * self.BATCH_SIZE) // train_ds.num_examples
              if epoch != last_epoch or epoch >= self.MAX_EPOCHS:
                learning_rate.assign(0.95 * learning_rate)
                # Save a checkpoint and evaluate the model .
                saver.save(sess, self.TRAIN_DIR + '/save', global_step=step)
                # Evaluate against the training set.
                print("Epoch:", epoch)
                self.do_eval(sess,
                        eval_correct,
                        self.features_placeholder,
                        self.labels_placeholder,
                        train_ds, "Training:")
                # Evaluate against the validation set.
                self.do_eval(sess,
                        eval_correct,
                        self.features_placeholder,
                        self.labels_placeholder,
                        eval_ds, "Validation:")
              last_epoch = epoch
              step += 1
          except KeyboardInterrupt:
              break

        self.sess = sess
        self.logits = logits
        self.predictions = predictions



    def complete_batch(self, x):
        n = len(x)
        assert n > self.BATCH_SIZE // 2 # This limitation can be fixed
        if n % self.BATCH_SIZE == 0:
            return x
        else:
            while len(x) < self.BATCH_SIZE // 2:
                x = np.concatenate([x, x], axis=0)
            extra = self.BATCH_SIZE - n % self.BATCH_SIZE
            return np.concatenate([x, x[:extra]], axis=0)
