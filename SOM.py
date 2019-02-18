import tensorflow as tf
import numpy as np
 
 
class SOM(object):
    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    """
 
    def __init__(self, m, n, dim, c=1.0, n_iterations=10, n_episodes=100, alpha=None, sigma=None):
        """
        Initializes all necessary components of the TensorFlow
        Graph.

        m X n are the dimensions of the SOM. 'n_iterations' should
        should be an integer denoting the number of iterations undergone
        while training.
        'dim' is the dimensionality of the training inputs.
        'alpha' is a number denoting the initial time(iteration no)-based
        learning rate. Default value is 0.3
        'sigma' is the the initial neighbourhood value, denoting
        the radius of influence of the BMU while training. By default, its
        taken to be half of max(m, n).
        """

        # Assign required variables first
        self.m = m
        self.n = n
        self.dim = dim
        self.c = c
        if alpha is None:
            self.alpha = 0.3
        else:
            self.alpha = float(alpha)
        if sigma is None:
            self.sigma = max(m, n) / 2.0
        else:
            self.sigma = float(sigma)
        self.n_iterations = abs(int(n_iterations))
        self.n_episodes = abs(int(n_episodes))

        # initialize session to None, set it right later.
        self._sess = None

        # VARIABLES AND CONSTANT OPS FOR DATA STORAGE

        # Randomly initialized weightage vectors for all neurons,
        # stored together as a matrix Variable of size [m*n, dim]
        self.weightage_vects = tf.Variable(tf.random_normal([m*n, dim]))

        # Matrix of size [m*n, 2] for SOM grid locations
        # of neurons
        self.location_vects = tf.constant(np.array(list(self._neuron_locations(m, n))))

        # PLACEHOLDERS FOR TRAINING INPUTS
        # We need to assign them as attributes to self, since they
        # will be fed in during training

        # The training vector
        self.vect_input = tf.placeholder("float", [dim])
        # Iteration number
        self.iter_input = tf.placeholder("float")

        # CONSTRUCT TRAINING OP PIECE BY PIECE
        # Only the final, 'root' training op needs to be assigned as
        # an attribute to self, since all the rest will be executed
        # automatically during training

        # Extract BMU location
        bmu_loc = self.get_bmu_location(self.vect_input)

        # To compute the alpha and sigma values based on iteration number
        learning_rate_op = tf.subtract(1.0, tf.div(self.iter_input, self.n_episodes))
        _alpha_op = tf.multiply(self.alpha, learning_rate_op)
        _sigma_op = tf.multiply(self.sigma, learning_rate_op)

        # Construct the op that will generate a vector with learning
        # rates for all neurons, based on iteration number and location
        # wrt BMU.
        bmu_distance_squares = tf.reduce_sum(tf.pow(tf.subtract(self.location_vects, tf.stack([bmu_loc for i in range(m*n)])), 2), 1)
        neighbourhood_func = tf.exp(tf.negative(tf.div(tf.cast(bmu_distance_squares, "float32"), tf.pow(_sigma_op, 2))))
        learning_rate_op = tf.multiply(_alpha_op, neighbourhood_func)

        # Finally, the op that will use learning_rate_op to update
        # the weightage vectors of all neurons based on a particular
        # input
        learning_rate_multiplier = tf.stack([tf.tile(tf.slice(learning_rate_op, np.array([i]), np.array([1])), [dim]) for i in range(m*n)])
        weightage_delta = tf.multiply(learning_rate_multiplier, tf.subtract(tf.stack([self.vect_input for i in range(m*n)]), self.weightage_vects))
        new_weightages_op = tf.add(self.weightage_vects, weightage_delta)

        # define operations
        self._training_op = tf.assign(self.weightage_vects, new_weightages_op)
        self._error_op = tf.reduce_min(tf.sqrt(self.get_ed2(self.vect_input)))
 
    @staticmethod
    def _neuron_locations(m, n):
        """
        iterator function that yields positions of SOM units.
        """
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])

    def get_ed2(self, vect_input):
        """
        Calculates squared Euclidean distance between input and every neuron
        """

        return tf.reduce_sum(tf.pow(tf.subtract(self.weightage_vects, tf.stack([vect_input for i in range(self.m*self.n)])), 2), 1)

    def get_activations(self, vect_input):
        """
        Calculates map activations
        """

        # calculate SOM activations
        som_activations = tf.exp(tf.multiply(tf.negative(self.c), self.get_ed2(vect_input)))
        # ...normalize and return
        return tf.div(som_activations, tf.reduce_sum(som_activations))

    def get_bmu_index(self, vect_input):
        """
        Calculates the Euclidean distance between every
        neuron's weightage vector and the input, and returns the
        index of the neuron which gives the least value
        """

        return tf.argmin(tf.sqrt(self.get_ed2(vect_input)), 0)

    def get_bmu_location(self, vect_input):
        """
        This will extract the location of the BMU based on the BMU's index
        """

        bmu_index = self.get_bmu_index(vect_input)
        slice_input = tf.pad(tf.reshape(bmu_index, [1]), np.array([[0, 1]]))
        return tf.reshape(tf.slice(self.location_vects, slice_input, tf.constant(np.array([1, 2]), dtype=tf.int64)), [2])
 
    def train(self, input_vects, episode_no):
        """
        Trains the SOM.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Current weightage vectors for all neurons(initially random) are
        taken as starting conditions for training.
        """

        result = None
        error = []
        # Training iterations
        for iter_no in range(self.n_iterations):
            input_vect = input_vects[iter_no]
            if input_vect is None:
                continue
            _error, result = self.fit(input_vect, episode_no)

            error.append(_error)

        error = np.mean(error)

        return error, result

    def fit(self, input_vect, curr_iteration):
        """
        fits the SOM given the current iteration based on input vector.

        input_vect: vision input vector.
        curr_iteration: current episode number.
        """

        result = self._sess.run(self._training_op, feed_dict={self.vect_input: input_vect, self.iter_input: curr_iteration})
        error = self._sess.run(self._error_op, feed_dict={self.vect_input: input_vect})
        return error, result
