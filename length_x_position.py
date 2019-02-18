import tensorflow as tf
import numpy as np

from data_generator import WordVector
from word_x_category import WordXCategory


class LengthXPosition(object):

    def __init__(self, word_vector, word_x_category, n_categories, max_words, alpha=1.0, threshold=0.9):
        """
        defines variables and Tensorflow operations for training and retrieving length_x_position probabilities.

        word_vector: instance of WordVector class.
        word_x_category: instance of WordXCategory table containing probabilites of each word belonging to a category.
        n_categories: number of all categories.
        max_words: number of all words in vocabulary.
        alpha: bootstrapping learning rate.
        threshold: language vector word probability threshold.
        """

        self.word_x_category_table = word_x_category.words_x_categories
        self.word_vector = word_vector
        self.n_categories = n_categories
        self.max_words = max_words
        self.alpha = alpha
        self.threshold = threshold

        # initialize length_x_position table as a dictionary.
        self.lengths_x_positions_table = dict()

        self._sess = None

        # initialize Tensorflow placeholders.
        self.word_index = tf.placeholder(tf.int32)
        self.category_prob = tf.placeholder(tf.float32, [self.n_categories])

        # get the right slice of word_x_category table and multiply values with learning rate.
        word_probabilities = tf.reshape(tf.slice(self.word_x_category_table, [0, self.word_index], [-1, 1]), [-1])
        word_probabilities = tf.multiply(word_probabilities, self.alpha)

        # add to previous probabilities.
        self.update_prob_word_op = tf.add(self.category_prob, word_probabilities)

        # and normalize.
        self.normalize_input = tf.placeholder(tf.float32, [self.n_categories])
        self.normalize_op = tf.div(self.normalize_input, tf.reduce_sum(self.normalize_input))

    def update_prob_array(self):
        """
        updates probability array based on previously changed word_x_category table.
        """

        # determine length of current phrase.
        input_vector = self.word_vector.string_vector
        word_indexes = np.reshape(np.argwhere(input_vector >= self.threshold), -1)
        length = len(word_indexes)

        # some error catching, probabily not needed in current implementation.
        if length == 0:
            return
        # if word of this length was not presented so far, add the entry for this length
        # in length_x_positions table.
        if length-1 not in self.lengths_x_positions_table:
            self.add_table_row(length)

        # update probabilities for each position in word-phrase.
        all_pos_cat_probs = self.lengths_x_positions_table[length-1]
        new_pos_cat_probs = np.array(all_pos_cat_probs)
        for i, (word_index, word_cat_probs) in enumerate(zip(word_indexes, all_pos_cat_probs)):
            new_pos_cat_probs[i] = self.update_prob_word(word_index, word_cat_probs)

        self.lengths_x_positions_table[length-1] = new_pos_cat_probs

    def update_prob_word(self, word_index, word_cat_probs):
        """
        updates just probabilites corresponding to one position.
        returns new category probabilites for this position.

        word_index: index of the word in language_vector.
        word_cat_probs: current category probability vector for current position.
        """

        return self._sess.run(self.update_prob_word_op, feed_dict={self.word_index: word_index, self.category_prob: word_cat_probs})

    def normalize_all(self):
        """
        normalizes whole entry that represents length of last word-phrase.
        must be called after every finish of an iteration.
        """

        # determine length of last word-phrase.
        input_vector = self.word_vector.string_vector
        word_indexes = np.reshape(np.argwhere(input_vector >= self.threshold), -1)
        length = len(word_indexes)
        if length == 0:
            return

        # retrieves table for that length
        normalized = np.array(self.lengths_x_positions_table[length-1])
        for i, pos_cat_probs in enumerate(self.lengths_x_positions_table[length-1]):
            # normalizes each row.
            normalized[i] = self._sess.run(self.normalize_op, feed_dict={self.normalize_input: pos_cat_probs})
        # saves the normalized vector back to original table.
        self.lengths_x_positions_table[length-1] = normalized

    def add_table_row(self, length):
        """
        adds an entry for phrase of length length. 
        initializes new table with uniform probability distribution.
        """

        sample = [1 / self.n_categories for i in range(self.n_categories)]
        txp = []
        for i in range(length):
            txp.append(np.array(sample))
        self.lengths_x_positions_table[length-1] = np.array(txp)

    def get_type_probabilities(self, length):
        """
        returns the correct entry from LXP table based on specified length.
        """

        # if entry for this length doesn't exist, create new one.
        if length-1 not in self.lengths_x_positions_table:
            self.add_table_row(length)
        return self.lengths_x_positions_table[length-1]

    def set_word_x_category_table(self, wxc_table):
        """
        accepts word_x_category table and sets it as object variable.
        this might be totally redundant.
        """
        self.word_x_category_table = wxc_table


if __name__ == "__main__":
    """
    some testing code used when implementing the class.
    """

    word_vector = WordVector()
    word_x_category = WordXCategory(word_vector, 256, 4)
    length_x_position = LengthXPosition(word_vector, word_x_category, 4, 4)

    length_x_position._sess = tf.Session()
    length_x_position._sess.run(tf.global_variables_initializer())

    a, _ = word_vector.generate_new_input_pair()
    length_x_position.update_prob_array()
    length_x_position.normalize_all()
