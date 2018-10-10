import tensorflow as tf
import numpy as np
 
 
class SOM(object):
    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    """
 
    #To check if the SOM has been trained
    _trained = False
 
    def __init__(self, m, n, dim, graph, n_iterations=100, alpha=None, sigma=None):
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
        if alpha is None:
            self.alpha = 0.3
        else:
            self.alpha = float(alpha)
        if sigma is None:
            self.sigma = max(m, n) / 2.0
        else:
            self.sigma = float(sigma)
        self.n_iterations = abs(int(n_iterations))

        # INITIALIZE GRAPH
        self._graph = graph

        # POPULATE GRAPH WITH NECESSARY COMPONENTS
        with self._graph.as_default():

            # VARIABLES AND CONSTANT OPS FOR DATA STORAGE

            # Randomly initialized weightage vectors for all neurons,
            # stored together as a matrix Variable of size [m*n, dim]
            ########### should i initialize to only positive numbers? ###########
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

            # To compute the Best Matching Unit given a vector
            # Basically calculates the Euclidean distance between every
            # neuron's weightage vector and the input, and returns the
            # index of the neuron which gives the least value
            bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(self.weightage_vects, tf.stack([self.vect_input for i in range(m*n)])), 2), 1)), 0)

            # This will extract the location of the BMU based on the BMU's
            # index
            slice_input = tf.pad(tf.reshape(bmu_index, [1]), np.array([[0, 1]]))
            bmu_loc = tf.reshape(tf.slice(self.location_vects, slice_input, tf.constant(np.array([1, 2]), dtype=tf.int64)), [2])

            # To compute the alpha and sigma values based on iteration
            # number
            learning_rate_op = tf.subtract(1.0, tf.div(self.iter_input, self.n_iterations))
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
            self._training_op = tf.assign(self.weightage_vects, new_weightages_op)

            # INITIALIZE SESSION
            self._sess = tf.Session()

            # INITIALIZE VARIABLES
            init_op = tf.global_variables_initializer()
            self._sess.run(init_op)
 
    def _neuron_locations(self, m, n):
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])
 
    def train(self, input_vects):
        """
        Trains the SOM.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Current weightage vectors for all neurons(initially random) are
        taken as starting conditions for training.
        """

        result = None
        # Training iterations
        for iter_no in range(self.n_iterations):
            i = np.random.randint(len(input_vects))
            input_vect = input_vects[i]
            result = self.fit(input_vect, iter_no)

        self._trained = True
        return result

    def fit(self, input_vect, curr_iteration):
        result = self._sess.run(self._training_op, feed_dict={self.vect_input: input_vect, self.iter_input: curr_iteration})
        return result
