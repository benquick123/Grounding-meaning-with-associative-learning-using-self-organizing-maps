import tensorflow as tf
import numpy as np
from SOM import SOM


class Hebbian(object):

    _trained = False

    def __init__(self, som1, som2, graph, n_iterations=100, alpha=None, sigma=None):
        self.som1 = som1
        self.som2 = som2

        self.n_iterations = abs(int(n_iterations))
        if alpha is None:
            self.alpha = 0.3
        else:
            self.alpha = alpha

        if sigma is None:
            self.sigma = (self.som2.m * self.som2.n) / 2
        else:
            self.sigma = sigma

        self._graph = graph

        with self._graph.as_default():
            # initialize weights to 0
            self._hebbian_weights = tf.Variable(tf.zeros([som1.m*som1.n, som2.m*som2.n]))

            # initialize placeholders (inputs)
            self.input_vect1 = tf.placeholder("float", [self.som1.dim])
            self.input_vect2 = tf.placeholder("float", [self.som2.dim])
            self.iter_input = tf.placeholder("float")

            # find bmu for SOM1 (only index)
            bmu_index1 = tf.argmin(tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(self.som1.weightage_vects, tf.stack([self.input_vect1 for i in range(self.som1.m*self.som1.n)])), 2), 1)))

            # find bmu for SOM2 (index & location)
            bmu_index2 = tf.argmin(tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(self.som2.weightage_vects, tf.stack([self.input_vect2 for i in range(self.som2.m*self.som2.n)])), 2), 1)))
            slice_begin2 = tf.pad(tf.reshape(bmu_index2, [1]), [[0, 1]])
            bmu_loc2 = tf.reshape(tf.slice(self.som2.location_vects, slice_begin2, [1, 2]), [2])

            # compute learning rate (based on current iteration) and adjust alpha & sigma accordingly
            learning_rate = tf.subtract(1.0, tf.div(self.iter_input, self.n_iterations))
            _alpha = tf.multiply(self.alpha, learning_rate)
            _sigma = tf.multiply(self.sigma, learning_rate)

            # calculate distances from bmu2, then calc neighbourhood function (e^(-(distance^2 / sigma^2)))
            # and finally calc learning rate function (includes alpha)
            bmu_distances2_squared = tf.cast(tf.reduce_sum(tf.pow(tf.subtract(self.som2.location_vects, tf.stack([bmu_loc2 for i in range(self.som2.m*self.som2.n)])), 2), 1), tf.float32)
            neighbourhood_func = tf.exp(tf.negative(tf.div(bmu_distances2_squared, tf.pow(_sigma, 2))))
            learning_rate = tf.multiply(_alpha, neighbourhood_func)

            # calculate weight delta (learning rate * SOM2 * SOM1[bmu_index1]) and add it to previous values.
            ########### is it okay if i take mean of the SOM weight? ############
            ########### is it okay if i only take positive numbers? #############
            weights_delta = tf.multiply(tf.multiply(learning_rate, tf.reduce_mean(tf.abs(self.som2.weightage_vects), 1)), tf.slice(tf.reduce_mean(tf.abs(self.som1.weightage_vects) , 1), [bmu_index1], [1]))
            hebbian_weights_slice = tf.slice(self._hebbian_weights, [bmu_index1, 0], [1, -1])
            new_hebbian_weights = tf.add(hebbian_weights_slice, weights_delta)

            # normalize
            new_hebbian_weights = tf.div(new_hebbian_weights, tf.reduce_sum(new_hebbian_weights))

            # first pad new weights, then replace all zeros with old values. overwrite old values with new.
            new_hebbian_weights = tf.pad(new_hebbian_weights, [[bmu_index1, (self.som1.m*self.som1.n) - bmu_index1 - 1], [0, 0]])
            new_hebbian_weights = tf.where(tf.equal(new_hebbian_weights, 0), self._hebbian_weights, new_hebbian_weights)
            self._training_op = tf.assign(self._hebbian_weights, new_hebbian_weights)

            # formalities
            self._sess = tf.Session()

            init_op = tf.global_variables_initializer()
            self._sess.run(init_op)

    def train(self, input_vects_pairs):
        result = None
        for iter_no in range(self.n_iterations):
            i = np.random.randint(len(input_vects_pairs))
            input1, input2 = input_vects_pairs[i]
            result = self.fit(input1, input2, iter_no)

        self._trained = True
        return result

    def fit(self, input1, input2, curr_iteration):
        result = self._sess.run(self._training_op, feed_dict={self.input_vect1: input1, self.input_vect2: input2, self.iter_input: curr_iteration})
        return result


# tf.enable_eager_execution()
# graph = tf.Graph()
# a = Hebbian(SOM(4, 4, 3, graph), SOM(5, 5, 3, graph), graph)
