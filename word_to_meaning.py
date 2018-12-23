import tensorflow as tf
import numpy as np
import pickle

from data_generator import WordVector


class Word2Meaning(object):
    """
    for reconstruction to work with current implementation, all SOMs must have the same size.
    """

    def __init__(self, som_filename, hebb_filename, wxc_filename, input_vector_dim, som_dim, n_categories):
        self.n_categories = n_categories
        self.input_vector_dim = input_vector_dim
        self.som_dim = som_dim

        hebb_weights = pickle.load(open(hebb_filename, "rb"))
        self.hebb_weights = np.array([h[0] for h in hebb_weights])
        self.som_filename = som_filename
        self.som_weights = pickle.load(open(self.som_filename, "rb"))[0]
        self.wcx_weights = pickle.load(open(wxc_filename, "rb"))

        self._sess = tf.Session()

        self.som_number = tf.placeholder(tf.int32)
        self.input_vector = tf.placeholder(tf.float32, [input_vector_dim])

        wxc_slice = tf.reshape(tf.slice(self.wcx_weights, [self.som_number, 0], [1, -1]), [-1])
        input_vector = tf.multiply(self.input_vector, wxc_slice)

        hebb_slice = tf.reshape(tf.slice(self.hebb_weights, [self.som_number, 0, 0], [1, -1, -1]), [input_vector_dim, -1])
        partial_input_vector = tf.argmax(tf.reduce_sum(tf.multiply(hebb_slice, tf.cast(tf.transpose(tf.stack([input_vector for i in range(self.som_dim)])), dtype=tf.float32)), axis=0))
        self.som_bmu_index_op = partial_input_vector           # tf.slice(self.som_weights, [best_activation_index, 0], [1, -1])

        self._sess.run(tf.global_variables_initializer())

    def get_vision_vector(self, input_vector):
        vision_vector = []
        for i in range(self.n_categories):
            self.som_weights = pickle.load(open(self.som_filename, "rb"))[i]
            bmu_index = self._sess.run(self.som_bmu_index_op, feed_dict={self.som_number: i, self.input_vector: input_vector})
            som_bmu = self.som_weights[bmu_index]
            vision_vector.append(som_bmu.tolist())
        return vision_vector

    def get_best_match(self, input_vector, vision_vectors):
        pass


if __name__ == "__main__":
    hebb_f = "weights/hebb_weights_1544576700.pickle"
    som_f = "weights/som_weights_1544576700.pickle"
    wxc_f = "weights/word_x_category_1544576700.pickle"

    word_vector = WordVector()
    word2meaning = Word2Meaning(som_f, hebb_f, wxc_f, word_vector.dim, 256, 4)
    distances = []
    for i in range(1000):
        input_vector, vision_vector = word_vector.generate_new_input_pair(full=True)
        # print(word_vector.generate_string_from_vector(input_vector))
        # print(input_vector)
        # print()
        reconstructed_vector = word2meaning.get_vision_vector(input_vector)
        # print("predicted:", reconstructed_vector)
        # print("original:", vision_vector)
        # print("distance:", distance(vision_vector, reconstructed_vector))
        distances.append(distance(reconstructed_vector, vision_vector))
    avg_distance = np.mean(distances)
    print(avg_distance)
