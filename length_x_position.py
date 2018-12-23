import tensorflow as tf
import numpy as np

from data_generator import WordVector
from word_x_category import WordXCategory


class LengthXPosition(object):

    def __init__(self, word_vector, word_x_category, n_categories, max_words, alpha=1.0, n_episodes=100, threshold=0.9):
        self.word_x_category_table = word_x_category.words_x_categories
        self.word_vector = word_vector
        self.n_categories = n_categories
        self.max_words = max_words
        self.n_episodes = n_episodes
        self.alpha = alpha
        self.threshold = threshold

        self.lengths_x_positions_table = dict()

        self._sess = None

        self.word_index = tf.placeholder(tf.int32)
        self.category_prob = tf.placeholder(tf.float32, [self.n_categories])
        self.input_iter = tf.placeholder(tf.float32)

        learning_rate = tf.subtract(1.0, tf.div(self.input_iter, self.n_episodes))
        _alpha = tf.multiply(self.alpha, learning_rate)

        word_probabilities = tf.reshape(tf.slice(self.word_x_category_table, [0, self.word_index], [-1, 1]), [-1])
        word_probabilities = tf.multiply(word_probabilities, _alpha)

        # add to previous probabilities and normalize
        self.update_prob_word_op = tf.add(self.category_prob, word_probabilities)

        self.normalize_input = tf.placeholder(tf.float32, [self.n_categories])
        self.normalize_op = tf.div(self.normalize_input, tf.reduce_sum(self.normalize_input))

    def update_prob_array(self, curr_iteration):
        input_vector = self.word_vector.string_vector
        word_indexes = np.reshape(np.argwhere(input_vector >= self.threshold), -1)
        length = len(word_indexes)
        if length == 0:
            return
        if length-1 not in self.lengths_x_positions_table:
            self.add_table_row(length)

        all_pos_cat_probs = self.lengths_x_positions_table[length-1]
        new_pos_cat_probs = np.array(all_pos_cat_probs)
        for i, (word_index, word_cat_probs) in enumerate(zip(word_indexes, all_pos_cat_probs)):
            new_pos_cat_probs[i] = self.update_prob_word(word_index, word_cat_probs, curr_iteration)

        self.lengths_x_positions_table[length-1] = new_pos_cat_probs

    def update_prob_word(self, word_index, word_cat_probs, curr_iteration):
        return self._sess.run(self.update_prob_word_op, feed_dict={self.word_index: word_index, self.category_prob: word_cat_probs, self.input_iter: float(curr_iteration)})

    def normalize_all(self):
        input_vector = self.word_vector.string_vector
        word_indexes = np.reshape(np.argwhere(input_vector >= self.threshold), -1)
        length = len(word_indexes)
        if length == 0:
            return

        normalized = np.array(self.lengths_x_positions_table[length-1])
        for i, pos_cat_probs in enumerate(self.lengths_x_positions_table[length-1]):
            normalized[i] = self._sess.run(self.normalize_op, feed_dict={self.normalize_input: pos_cat_probs})
        self.lengths_x_positions_table[length-1] = normalized

    def add_table_row(self, length):
        sample = [1 / self.n_categories for i in range(self.n_categories)]
        txp = []
        for i in range(length):
            txp.append(np.array(sample))
        self.lengths_x_positions_table[length-1] = np.array(txp)

    def get_type_probabilities(self, length):
        if length-1 not in self.lengths_x_positions_table:
            self.add_table_row(length)
        return self.lengths_x_positions_table[length-1]

    def set_word_x_category_table(self, wxc_table):
        self.word_x_category_table = wxc_table


if __name__ == "__main__":
    word_vector = WordVector()
    word_x_category = WordXCategory(word_vector, 256, 4)
    length_x_position = LengthXPosition(word_vector, word_x_category, 4, 4)

    length_x_position._sess = tf.Session()
    length_x_position._sess.run(tf.global_variables_initializer())

    a, _ = word_vector.generate_new_input_pair()
    length_x_position.update_prob_array(0)
    length_x_position.normalize_all()
