import tensorflow as tf
import numpy as np
from SOM import SOM
from data_generator import WordVector
from plotting import show_organization


class Hebbian(object):
    def __init__(self, word_vector, som, n_iterations=10, n_episodes=100, alpha=None):
        """
        initializes Hebbian links and defines necessary Tensorflow operations.

        word_vector: WordVector instance.
        som: SOM object.
        n_iterations: number of iterations when self.train() is used.
        n_episodes: number of episodes for appropriate decrease of learning rates.
        alpha: initial learnign rate.
        """

        self.som = som
        self.word_vector = word_vector

        self.n_iterations = abs(int(n_iterations))
        self.n_episodes = n_episodes
        if alpha is None:
            self.alpha = 0.2
        else:
            self.alpha = alpha

        # initialize session to None, set it to actual session later.
        self._sess = None

        # initialize hebbian links to 0.
        self._hebbian_weights_WM = tf.Variable(tf.zeros([self.word_vector.dim, self.som.m*self.som.n]), dtype=tf.float32)
        self._hebbian_weights_MW = tf.Variable(tf.zeros([self.som.m*self.som.n, self.word_vector.dim]), dtype=tf.float32)

        # initialize placeholders: inputs and category probability multiplier for bootstrapping.
        self.input_vect1 = tf.placeholder("float", [self.word_vector.dim])
        self.input_vect2 = tf.placeholder("float", [self.som.dim])

        self.input_iter = tf.placeholder("float")
        self.category_probability = tf.placeholder("float")


        # compute learning rate (based on current iteration) and adjust alpha accordingly
        # also added: _alpha reflects probability that input (just one word) is of certain type
        learning_rate = tf.subtract(1.0, tf.div(self.input_iter, self.n_episodes))
        _alpha = tf.multiply(tf.multiply(self.alpha, learning_rate), self.category_probability)
        # USE THIS ONLY WHEN TESTING TRAINING WITHOUT BOOTSTRAPPING.
        # _alpha = tf.multiply(self.alpha, learning_rate)             

        # get activation map
        som_activation_map = som.get_activations(self.input_vect2)

        # calculate weight delta (learning rate * som_activations * word_vector) and add it to previous values
        self._weights_delta_op = tf.multiply(tf.reshape(self.input_vect1, (self.word_vector.dim, 1)), tf.reshape(tf.multiply(_alpha, som_activation_map), (1, self.som.m*self.som.n)))

        # placeholder for hebbian weigths, followed by normalization.
        self._weights_delta = tf.placeholder("float", [self.word_vector.dim, self.som.m*self.som.n])
        # add it to W --> M hebbian links
        new_hebbian_weights_WM = tf.add(self._hebbian_weights_WM, self._weights_delta)
        # and to M --> W hebbian links
        new_hebbian_weights_MW = tf.add(self._hebbian_weights_MW, tf.transpose(self._weights_delta))

        # normalize
        # first W --> M
        weight_sums_WM = tf.reduce_sum(new_hebbian_weights_WM, 1)
        weight_sums_WM = tf.transpose(tf.stack([weight_sums_WM for i in range(self.som.m*self.som.n)]))
        new_hebbian_weights_WM = tf.where(tf.equal(weight_sums_WM, 0), new_hebbian_weights_WM, tf.div(new_hebbian_weights_WM, weight_sums_WM))
        # then M --> W
        weight_sums_MW = tf.reduce_sum(new_hebbian_weights_MW, 1)
        weight_sums_MW = tf.transpose(tf.stack([weight_sums_MW for i in range(self.word_vector.dim)]))
        new_hebbian_weights_MW = tf.where(tf.equal(weight_sums_MW, 0), new_hebbian_weights_MW, tf.div(new_hebbian_weights_MW, weight_sums_MW))

        # replace previous weights
        self._training_WM_op = tf.assign(self._hebbian_weights_WM, new_hebbian_weights_WM)
        self._training_MW_op = tf.assign(self._hebbian_weights_MW, new_hebbian_weights_MW)

        # get error
        self._error_op = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(self._hebbian_weights_WM, tf.transpose(self._hebbian_weights_MW)), 2)))

    def train(self, inputs1, inputs2, episode_no, train_som=False):
        """
        training pipeline not used in the most recent implementation. can additionally train a SOM too.

        inputs1: list of language inputs.
        inputs2: list of vision inputs.
        episode_no: current episode number.
        train_som: whether to train SOM also.
        """

        hebb_result = None
        som_result = None
        hebb_error = []
        som_error = []

        for iter_no in range(self.n_iterations):
            input1 = inputs1[iter_no]
            input2 = inputs2[iter_no]
            # print("INPUTS:", input1, input2)
            if input2 is None:
                print("THIS SHOULDN'T HAPPEN")
                continue
            if train_som:
                _som_error, som_result = self.som.fit(input2, episode_no)
                som_error.append(_som_error)

            beta = 1.0
            _hebb_error, hebb_result = self.fit(input1, input2, beta, episode_no)
            hebb_error.append(_hebb_error)

        hebb_error = np.mean(hebb_error)
        if train_som:
            som_error = np.mean(som_error)
            return som_error, som_result, hebb_error, hebb_result
        return hebb_error, hebb_result

    def fit(self, input1, input2, category_probability, curr_iteration):
        """
        fits hebbian links for one iteration given the inputs.

        input1: language input.
        input2: vision input.
        category_probability: (float) category probability for this position as given by length_x_position.
        curr_iteration: current episode number.
        """

        # calculates weights delta first.
        weights_delta = self._sess.run(self._weights_delta_op, feed_dict={self.input_vect1: input1, self.input_vect2: input2, self.input_iter: curr_iteration, self.category_probability: category_probability})
        # normalizes depending on the weight type.
        result_WM = self._sess.run(self._training_WM_op, feed_dict={self._weights_delta: weights_delta})
        result_MW = self._sess.run(self._training_MW_op, feed_dict={self._weights_delta: weights_delta})
        # calculates error and returns everything
        error = self._sess.run(self._error_op)
        return error, (result_WM, result_MW)


if __name__ == "__main__":
    pass
