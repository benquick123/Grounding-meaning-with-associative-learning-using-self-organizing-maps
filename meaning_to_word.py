import numpy as np
import tensorflow as tf
import pickle
import itertools

from data_generator import WordVector


class Meaning2Word(object):
    """
    for reconstruction to work with current implementation, all SOMs must have the same size.
    """

    def __init__(self, hebb_filename, som_filename, wxc_filename, input_vector_dim, som_dim, n_categories):
        self.input_vector_dim = input_vector_dim
        self.som_dim = som_dim
        self.n_categories = n_categories

        hebb_weights = pickle.load(open(hebb_filename, "rb"))
        self.hebb_weights = np.array([h[1] for h in hebb_weights])
        self.som_filename = som_filename
        self.som_weights = pickle.load(open(som_filename, "rb"))[0]
        self.wxc_weights = np.array(pickle.load(open(wxc_filename, "rb")))

        self._sess = tf.Session()

        self.som_number = tf.placeholder(tf.int32)
        self.bmu_index = tf.placeholder(tf.int32)

        # this gives just weights of dimension (16,)
        hebb_slice = tf.reshape(tf.slice(self.hebb_weights, [self.som_number, self.bmu_index, 0], [1, 1, -1]), [-1, input_vector_dim])
        partial_input_vector = tf.multiply(hebb_slice, tf.slice(self.wxc_weights, [self.som_number, 0], [1, -1]))
        partial_input_vector = tf.div(partial_input_vector, tf.reduce_sum(partial_input_vector))

        self.partial_input_vector_op = tf.reshape(partial_input_vector, [-1])       # tf.div(partial_input_vector, tf.reduce_sum(partial_input_vector))

        self._sess.run(tf.global_variables_initializer())

    def get_ed2(self, vect_input):
        return tf.reduce_sum(tf.pow(tf.subtract(self.som_weights, tf.stack([vect_input for i in range(self.som_dim)])), 2), 1)

    def get_bmu_index(self, vect_input):
        return tf.argmin(tf.sqrt(self.get_ed2(vect_input)), 0)

    def get_partial_input_vector(self, vision_vector, i):
        self.som_weights = pickle.load(open(self.som_filename, "rb"))[i]
        bmu_index = self.get_bmu_index(vision_vector[i])
        bmu_index = self._sess.run(bmu_index)
        partial_input_vector = self._sess.run(self.partial_input_vector_op, feed_dict={self.som_number: i, self.bmu_index: bmu_index})
        return partial_input_vector

    def get_input_vector(self, vision_vector):
        input_vector = np.zeros(self.input_vector_dim)
        for i in range(self.n_categories):
            partial_input_vector = self.get_partial_input_vector(vision_vector, i)
            # print(partial_input_vector)
            input_vector += partial_input_vector

        # input_vector = input_vector / np.sum(input_vector)
        return input_vector

    def get_best_match(self, vision_vector, input_vectors):
        pass


if __name__ == "__main__":
    hebb_f = "weights/hebb_weights_1544576700.pickle"
    som_f = "weights/som_weights_1544576700.pickle"
    wxc_f = "weights/word_x_category_1544576700.pickle"

    word_vector = WordVector()
    meaning2word = Meaning2Word(hebb_f, som_f, wxc_f, word_vector.dim, 256, 4)
    distances = []

    word_indices = np.array([0, 4, 6, 12])

    for i in range(100):
        # meaning2word = Meaning2Word(hebb_f, som_f, wxc_f, word_vector.dim, 256, 4)
        input_vector, vision_vector = word_vector.generate_new_input_pair(full=True)

        input_array = np.zeros((len(word_indices), word_vector.dim))
        for j in range(len(word_indices)):
            try:
                input_array[j, word_indices[j]:word_indices[j+1]] = input_vector[word_indices[j]:word_indices[j+1]]
            except IndexError:
                input_array[j, word_indices[j]:] = input_vector[word_indices[j]:]
            partial_input = meaning2word.get_partial_input_vector(vision_vector, j)
            print(partial_input.tolist())
            print(input_array[j, :])
            KL = KL_divergence(input_array[j, :], partial_input)
            print("KL:", KL)
            print()


        # print(vision_vector)
        # print()
        reconstructed_vector = meaning2word.get_input_vector(vision_vector)
        # print("predicted:", reconstructed_vector)
        # print("original:", input_vector)
        # print("distance:", distance(reconstructed_vector, input_vector))
        distances.append(distance(reconstructed_vector, input_vector))
        # print(distances)

    avg_distance = np.mean(distances)
    print(avg_distance)
