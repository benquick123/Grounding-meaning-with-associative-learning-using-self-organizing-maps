import numpy as np
import tensorflow as tf
import pickle
import itertools

from data_generator import WordVector


class WordXCategory(object):
    """ stores and updates probabilities of words belonging to certain categories """
    def __init__(self, word_vector, som_size, n_categories, alpha=1.0):
        """
        initialize variables and Tensorflow operations.
        """

        self.word_vector = word_vector
        self.som_size = som_size
        self.n_categories = n_categories
        self.alpha = alpha

        self._sess = None

        # initialize word_x_category table.
        self.words_x_categories = tf.Variable(tf.zeros([n_categories, word_vector.dim]), dtype=tf.float32)

        # initialize placeholder for w-->m hebbian links for this SOM.
        self._hebbian_weights = tf.placeholder("float", [word_vector.dim, som_size])
        # get hebbian links logaritms in base n=som_size
        hebb_weights_log_n = tf.div(tf.log(tf.clip_by_value(self._hebbian_weights, 1e-10, 1)), tf.log(tf.cast(som_size, dtype=tf.float32)))
        # calculate entropies. this produces vector of size (word_vector.dim)
        self.entropy_op = tf.negative(tf.reduce_sum(tf.multiply(self._hebbian_weights, hebb_weights_log_n), axis=1))

        # initialize placeholders for category numbers and 1-entropies values.
        self.index = tf.placeholder(tf.int32)
        self.new_inverse_entropies = tf.placeholder("float", word_vector.dim)

        # multiply entropies with alpha and add them to current words_x_categories table values.
        new_inverse_entropies = tf.multiply(self.new_inverse_entropies, [self.alpha])
        new_inverse_entropies = tf.pad(tf.reshape(new_inverse_entropies, [1, -1]), [[self.index, n_categories-1-self.index], [0, 0]])
        new_inverse_entropies = tf.add(self.words_x_categories, new_inverse_entropies)
        self.update_op = tf.assign(self.words_x_categories, new_inverse_entropies)

        # performs normalization.
        entropy_sums = tf.reduce_sum(self.words_x_categories, axis=1)
        self.normalize_op = tf.assign(self.words_x_categories, tf.div(self.words_x_categories, tf.transpose(tf.stack([entropy_sums for i in range(word_vector.dim)]))))

    def entropy(self, hebb_weights):
        """
        calculates and returns entropies of hebbian links going from each cell
        in language input.
        """

        return self._sess.run(self.entropy_op, feed_dict={self._hebbian_weights: hebb_weights})

    def inverse_entropy(self, entropy_weights):
        """
        simply calculates 1 - entropy and returns.
        """

        return 1.0 - entropy_weights

    def update_words_x_categories(self, new_inverse_entropies, index):
        """
        updates and returns word_x_category table with new inverse entropies table.
        """

        return self._sess.run(self.update_op, feed_dict={self.new_inverse_entropies: new_inverse_entropies, self.index: index})

    def normalize_all(self):
        """
        normalizes and returns word_x_category table.
        """

        return self._sess.run(self.normalize_op)


if __name__ == "__main__":
    """
    code for testing purposes. probably doesn't run with current implementation.
    """
    word_vector = WordVector()
    som_size = 16*16
    word_to_meaning = WordXCategory(word_vector, som_size, 4)
    word_to_meaning._sess = tf.Session()
    word_to_meaning._sess.run(tf.global_variables_initializer())

    h = pickle.load(open("hebb_weights_1542724910.pickle", "rb"))
    hebb_weights = []
    for _h in h:
        hebb_weights.append(_h[0])
    hebb_weights = np.array(hebb_weights)

    entropy_weights = np.zeros([4, 16])
    for i, h in enumerate(hebb_weights):
        inv_entropy = word_to_meaning.inverse_entropy(word_to_meaning.entropy(h))
        entropy_weights = word_to_meaning.update_words_x_categories(inv_entropy, i)
        print(entropy_weights)
    entropy_weights = word_to_meaning.normalize_all()
    print(entropy_weights)
    # pickle.dump(entropy_weights, open("entropy_weights.pickle", "wb"))
    exit()
