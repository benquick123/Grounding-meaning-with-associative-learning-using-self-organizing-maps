import tensorflow as tf
import numpy as np

from data_generator import WordVector
from word_x_category import WordXCategory


class TypeXProbability(object):

    def __init__(self, word_vector, word_x_category, n_categories, max_words, threshold=0.9):
        self.word_x_category_table = word_x_category.words_x_categories
        self.word_vector = word_vector
        self.n_categories = n_categories
        self.max_words = max_words
        self.threshold = threshold

        self.types_x_probabilities_table = dict()

        self._sess = None

        self.word_index = tf.placeholder(tf.int32)
        self.alpha = tf.placeholder(tf.float32)
        self.category_prob = tf.placeholder(tf.float32, [self.n_categories])
        word_probabilities = tf.reshape(tf.slice(self.word_x_category_table, [0, self.word_index], [-1, 1]), [-1])
        word_probabilities = tf.multiply(word_probabilities, self.alpha)

        # add to previous probabilities and normalize
        self.update_prob_word_op = tf.add(self.category_prob, word_probabilities)

        self.normalize_input = tf.placeholder(tf.float32, [self.n_categories])
        self.normalize_op = tf.div(self.normalize_input, tf.reduce_sum(self.normalize_input))

    def update_prob_array(self, alpha=1.0):
        input_vector = self.word_vector.string_vector
        word_indexes = np.reshape(np.argwhere(input_vector >= self.threshold), -1)
        length = len(word_indexes)
        if length == 0:
            return
        if length-1 not in self.types_x_probabilities_table:
            self.add_table_row(length)

        all_pos_cat_probs = self.types_x_probabilities_table[length-1]
        new_pos_cat_probs = np.array(all_pos_cat_probs)
        for i, (word_index, word_cat_probs) in enumerate(zip(word_indexes, all_pos_cat_probs)):
            new_pos_cat_probs[i] = self.update_prob_word(word_index, word_cat_probs, alpha)

        self.types_x_probabilities_table[length-1] = new_pos_cat_probs

    def update_prob_word(self, word_index, word_cat_probs, alpha=1.0):
        return self._sess.run(self.update_prob_word_op, feed_dict={self.word_index: word_index, self.category_prob: word_cat_probs, self.alpha: alpha})

    def normalize_all(self):
        input_vector = self.word_vector.string_vector
        word_indexes = np.reshape(np.argwhere(input_vector >= self.threshold), -1)
        length = len(word_indexes)
        if length == 0:
            return

        normalized = np.array(self.types_x_probabilities_table[length-1])
        for i, pos_cat_probs in enumerate(self.types_x_probabilities_table[length-1]):
            normalized[i] = self._sess.run(self.normalize_op, feed_dict={self.normalize_input: pos_cat_probs})
        self.types_x_probabilities_table[length-1] = normalized

    def add_table_row(self, length):
        sample = [1 / self.n_categories for i in range(self.n_categories)]
        txp = []
        for i in range(length):
            txp.append(np.array(sample))
        self.types_x_probabilities_table[length-1] = np.array(txp)

    def get_type_probabilities(self, length):
        if length-1 not in self.types_x_probabilities_table:
            self.add_table_row(length)
        return self.types_x_probabilities_table[length-1]


if __name__ == "__main__":
    word_vector = WordVector()
    word_x_category = WordXCategory(word_vector, 256, 4)
    type_x_probability = TypeXProbability(word_vector, word_x_category, 4, 4)

    type_x_probability._sess = tf.Session()
    type_x_probability._sess.run(tf.global_variables_initializer())

    a, _ = word_vector.generate_new_input_pair()
    type_x_probability.update_prob_array(int(sum(a)), alpha=0.25)
    type_x_probability.normalize_all(int(sum(a)))
