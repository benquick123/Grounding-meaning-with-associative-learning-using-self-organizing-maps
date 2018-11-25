import numpy as np
import tensorflow as tf
import pickle
import itertools

from data_generator import WordVector


class MeaningToWord(object):

    def __init__(self, word_vector, n_vision_vectors):
        self.word_vectoer = word_vector
        self.n_vision_vectors = n_vision_vectors

        h = pickle.load(open("hebb_weights_1540937327.pickle", "rb"))
        self.hebb_weights = []
        for _h in h:
            self.hebb_weights.append(_h[0])
        self.hebb_weights = np.array(self.hebb_weights)

        self.som_weights = pickle.load(open("som_weights_1540937327.pickle", "rb"))

        self._sess = tf.Session()

        # TODO: change code to accept any number of modalities / SOMs.
        self.pos_vision_vector = tf.placeholder("float", [self.som_weights[0].shape[1]])
        self.size_vision_vector = tf.placeholder("float", [self.som_weights[1].shape[1]])
        self.color_vision_vector = tf.placeholder("float", [self.som_weights[2].shape[1]])
        self.shape_vision_vector = tf.placeholder("float", [self.som_weights[3].shape[1]])

        pos_activations = self.get_activations(self.som_weights[0], self.pos_vision_vector)
        size_activations = self.get_activations(self.som_weights[1], self.size_vision_vector)
        color_activations = self.get_activations(self.som_weights[2], self.color_vision_vector)
        shape_activations = self.get_activations(self.som_weights[3], self.shape_vision_vector)

        pos_index = tf.argmax(tf.reduce_sum(tf.multiply(tf.transpose(self.hebb_weights[0]), tf.transpose(tf.stack([pos_activations for i in range(self.hebb_weights[0].shape[0])]))), axis=0))
        size_index = tf.argmax(tf.reduce_sum(tf.multiply(tf.transpose(self.hebb_weights[1]), tf.transpose(tf.stack([size_activations for i in range(self.hebb_weights[1].shape[0])]))), axis=0))
        color_index = tf.argmax(tf.reduce_sum(tf.multiply(tf.transpose(self.hebb_weights[2]), tf.transpose(tf.stack([color_activations for i in range(self.hebb_weights[2].shape[0])]))), axis=0))
        shape_index = tf.argmax(tf.reduce_sum(tf.multiply(tf.transpose(self.hebb_weights[3]), tf.transpose(tf.stack([shape_activations for i in range(self.hebb_weights[3].shape[0])]))), axis=0))

        indices = [[pos_index], [size_index], [color_index], [shape_index]]
        values = [1.0, 1.0, 1.0, 1.0]
        self.output_word_vector_op = tf.sparse_tensor_to_dense(tf.SparseTensor(indices, values, [word_vector.dim]))

        init_op = tf.global_variables_initializer()
        self._sess.run(init_op)

    @staticmethod
    def get_ed2(som_weights, vect_input):
        """
        Calculates squared Euclidean distance between input and every neuron
        """

        return tf.reduce_sum(tf.pow(tf.subtract(som_weights, tf.stack([vect_input for i in range(som_weights.shape[0])])), 2), 1)

    def get_activations(self, som_weights, vect_input, alpha=1.0):
        """
        Calculates map activations
        """

        # calculate SOM activations
        som_activations = tf.exp(tf.multiply(tf.negative(alpha), self.get_ed2(som_weights, vect_input)))
        # ...normalize and return
        return tf.div(som_activations, tf.reduce_sum(som_activations))

    def get_word_vector(self, vision_vector):
        return self._sess.run(self.output_word_vector_op, feed_dict={self.pos_vision_vector: vision_vector[0], self.size_vision_vector: vision_vector[1], self.color_vision_vector: vision_vector[2], self.shape_vision_vector: vision_vector[3]})


if __name__ == "__main__":
    n_vision_vectors = 3
    word_vector = WordVector()
    meaning_to_word = MeaningToWord(word_vector, n_vision_vectors)

    input_pair = word_vector.generate_new_input_pair(full=True)
    input_vector = input_pair[0]

    # get input vector based on vision_input
    print(input_vector)
    result_vector = meaning_to_word.get_word_vector(input_pair[1])
    print(result_vector)
    print()




