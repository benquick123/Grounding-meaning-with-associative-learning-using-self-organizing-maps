import tensorflow as tf
import numpy as np
from SOM import SOM
from data_generator import WordVector
from plotting import show_organization


class Hebbian(object):

    _trained = False
    word_type_matrix = None
    n_instances = 4

    def __init__(self, word_vector, som, instance_number, n_iterations=10, n_episodes=100, alpha=None, sigma=None, activation_map_constant=1.0):
        self.som = som
        self.word_vector = word_vector
        self.instance_number = instance_number

        self.n_iterations = abs(int(n_iterations))
        self.n_episodes = n_episodes
        if alpha is None:
            self.alpha = 0.2
        else:
            self.alpha = alpha

        if sigma is None:
            self.sigma = (self.som.m * self.som.n) / 2
        else:
            self.sigma = sigma

        # initialize session to None, set it to actual session later.
        self._sess = None

        self._hebbian_weights_WM = tf.Variable(tf.zeros([self.word_vector.dim, self.som.m*self.som.n]), dtype=tf.float32)
        self._hebbian_weights_MW = tf.Variable(tf.zeros([self.som.m*self.som.n, self.word_vector.dim]), dtype=tf.float32)

        # initialize placeholders (inputs)
        self.input_vect1 = tf.placeholder("float", [self.word_vector.dim])
        self.input_vect2 = tf.placeholder("float", [self.som.dim])

        self.input_iter = tf.placeholder("float")
        self.category_probability = tf.placeholder("float")

        # find bmu for SOM (location, no index needed)
        bmu_loc = som.get_bmu_location(self.input_vect2)

        # compute learning rate (based on current iteration) and adjust alpha & sigma accordingly
        # also added: _alpha reflects probability that input (just one word) is of certain type
        learning_rate = tf.subtract(1.0, tf.div(self.input_iter, self.n_episodes))
        _alpha = tf.multiply(tf.multiply(self.alpha, learning_rate), self.category_probability)
        # _alpha = tf.multiply(self.alpha, learning_rate)             # COMMENT THIS!!!
        _sigma = tf.multiply(self.sigma, learning_rate)

        # calculate distances from bmu2, then calc neighbourhood function (e^(-(distance^2 / sigma^2)))
        # and finally calc learning rate function (includes alpha)
        bmu_distances_squared = tf.cast(tf.reduce_sum(tf.pow(tf.subtract(self.som.location_vects, tf.stack([bmu_loc for i in range(self.som.m*self.som.n)])), 2), 1), tf.float32)
        neighbourhood_func = tf.exp(tf.negative(tf.div(bmu_distances_squared, tf.pow(_sigma, 2))))
        learning_rate = tf.multiply(_alpha, neighbourhood_func)

        # get activation map
        som_activation_map = som.get_activations(self.input_vect2)

        # calculate weight delta (learning rate * som_activations * word_vector) and add it to previous values
        self._weights_delta_op = tf.multiply(tf.reshape(self.input_vect1, (self.word_vector.dim, 1)), tf.reshape(tf.multiply(learning_rate, som_activation_map), (1, self.som.m*self.som.n)))

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
        weights_delta = self._sess.run(self._weights_delta_op, feed_dict={self.input_vect1: input1, self.input_vect2: input2, self.input_iter: curr_iteration, self.category_probability: category_probability})
        result_WM = self._sess.run(self._training_WM_op, feed_dict={self._weights_delta: weights_delta})
        result_MW = self._sess.run(self._training_MW_op, feed_dict={self._weights_delta: weights_delta})
        error = self._sess.run(self._error_op)
        # entropy = self._sess.run(self._update_word_type_matrix_op)
        return error, (result_WM, result_MW)


if __name__ == "__main__":
    tf.enable_eager_execution()
    graph = tf.Graph()

    # a = Hebbian(SOM(5, 5, 3, graph), WordVector(), graph)
