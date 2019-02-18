import numpy as np
import tensorflow as tf
import pickle
from sklearn.metrics.classification import recall_score, precision_score, accuracy_score

from data_generator import WordVector
from utils import LoggerHelper
import plotting


class Meaning2Word(object):
    """
    for reconstruction to work with current implementation, all SOMs must have the same size.
    gets the vision vector and outputs the probability for each word that it represents that vision vector.
    """

    def __init__(self, hebb_filename, som_filename, wxc_filename, input_vector_dim, som_dim, n_categories):
        """
        intializes variables and Tensorflow operations.

        hebb_filename: filename of Hebbian links weights.
        som_filename: filename of all SOMs weights.
        wxc_filename: filename of word_x_category table weights.
        input_vector_dim: dimension of language vector.
        som_dim: number of cells is SOM.
        n_categories: number of all categories.
        """

        self.input_vector_dim = input_vector_dim
        self.som_dim = som_dim
        self.n_categories = n_categories

        # load all weights.
        hebb_weights = pickle.load(open(hebb_filename, "rb"))
        self.hebb_weights = np.array([h[1] for h in hebb_weights])
        self.som_filename = som_filename
        self.soms = pickle.load(open(som_filename, "rb"))
        self.som_weights = self.soms[0]
        self.wxc_weights = np.array(pickle.load(open(wxc_filename, "rb")))

        self._sess = tf.Session()

        # initialize placeholder for choosing correct cell and weights.
        self.som_number = tf.placeholder(tf.int32)
        self.bmu_index = tf.placeholder(tf.int32)

        # this gives just weights of dimension (16,)
        hebb_slice = tf.reshape(tf.slice(self.hebb_weights, [self.som_number, self.bmu_index, 0], [1, 1, -1]), [-1, input_vector_dim])
        # multiplies with wxc row for current category (category == som_number)
        partial_input_vector = tf.multiply(hebb_slice, tf.slice(self.wxc_weights, [self.som_number, 0], [1, -1]))
        # normalize.
        partial_input_vector = tf.div(partial_input_vector, tf.reduce_sum(partial_input_vector))
        self.partial_input_vector_op = tf.reshape(partial_input_vector, [-1])

        self._sess.run(tf.global_variables_initializer())

    def get_ed2(self, vect_input):
        """ calculates and returns euclidean distance. """
        return tf.reduce_sum(tf.pow(tf.subtract(self.som_weights, tf.stack([vect_input for i in range(self.som_dim)])), 2), 1)

    def get_bmu_index(self, vect_input):
        """ returns best matching unit given the vect_input. """
        return tf.argmin(tf.sqrt(self.get_ed2(vect_input)), 0)

    def get_partial_input_vector(self, vision_vector, i):
        """
        constructs partial language inptu vector. 
        vision_vector: vision input for current category.
        i: current category number.
        """

        # select SOM weights matching current category.
        self.som_weights = self.soms[i]
        # get best matching unit.
        bmu_index = self.get_bmu_index(vision_vector[i])
        bmu_index = self._sess.run(bmu_index)
        # get partial reconstructed language and return.
        partial_input_vector = self._sess.run(self.partial_input_vector_op, feed_dict={self.som_number: i, self.bmu_index: bmu_index})
        return partial_input_vector

    def get_input_vector(self, vision_vector):
        """
        reconstructs and returns whole language vector.
        vision_vector: complete vision vector.
        """
        # iterate through all parts of vision vector.
        input_vector = np.zeros(self.input_vector_dim)
        for i in range(self.n_categories):
            # get partial reconstruction and add it to current reconstructed input_vector.
            partial_input_vector = self.get_partial_input_vector(vision_vector, i)
            input_vector += partial_input_vector

        return input_vector


def load_and_plot(path_to_models):
    """
    loads data from file and plots reconstruction metrics.

    path_to_models: path to .csv log file.
    """

    header = "distance: position/tdistance: size/tdistance: color/tdistance: type/tdist_avg/tKL divergence: position/tKL divergence: size/tKL divergence: color/tKL divergence: type/tkl_avg/tprecision/trecall/taccuracy"
    header = header.split("/t")
    distances = np.loadtxt(path_to_models + "meaning2word_metrics   .csv")
    save_path = "Plotting/" + path_to_models.split("/")[1] + "/v2l_"
    mode = "show"
    plotting.plot_per_epoch(np.transpose(distances)[:4], header[:4], mode=mode, path=save_path + "dist_no_boot")
    plotting.plot_per_epoch(np.transpose(distances)[5:9], header[5:9], mode=mode, path=save_path + "kl_no_boot")
    plotting.plot_per_epoch(np.transpose(distances)[10:], header[10:], mode=mode, path=save_path + "PRA_no_boot")
    print(np.transpose(distances)[10:, -1])


def generate_metrics(n, path_to_models):
    """
    traverses through all models and reconstructs language; then it collects metrics and logs them.
    not the most beautiful piece of code here, but works as long as the data has same structure.
    n: number of all episodes to traverse though.
    path_to_models: path, where models are stored.
    """

    header = "dist_pos/tdist_siz/tdist_col/tdist_typ/tdist_avg/tkl_pos/tkl_siz/tkl_col/tkl_typ/tkl_avg/tprecision/trecall/taccuracy"

    # initialize structures necessary for reconstruction.
    word_vector = WordVector()
    logger_helper = LoggerHelper(None, None)
    language_vector = []
    vision_vector = []
    word_indices = np.array([0, 4, 6, 12, 16])

    # generate 100 examples of inputs pairs.
    for i in range(100):
        _lang, _vis = word_vector.generate_new_input_pair(full=True)
        language_vector.append(_lang)
        vision_vector.append(_vis)

    # initialize global error metrics.
    distances = []
    kl_divergences = []
    precisions = []
    recalls = []
    accuracies = []
    for iteration in range(n):
        # construct filenames for current episode and initialize meaning2word object.
        hebb_f = path_to_models + "Episodes/Episode " + str(iteration) + "/hebb_weights.pickle"
        som_f = path_to_models + "Episodes/Episode " + str(iteration) + "/som_weights.pickle"
        wxc_f = path_to_models + "Episodes/Episode " + str(iteration) + "/word_x_category.pickle"
        meaning2word = Meaning2Word(hebb_f, som_f, wxc_f, word_vector.dim, 256, 4)

        # initialize error metrics for current episode.
        _distances = [[], [], [], []]
        _kl_divergences = [[], [], [], []]
        _precisions = []
        _recalls = []
        _accuracies = []
        for j in range(100):
            # iterate through all input pairs.
            partial_language_vector = np.zeros((4, word_vector.dim))
            reconstructed_language = np.zeros(word_vector.dim)
            for k in range(4):
                # iterate through all categories and reconstruct language for every partial vision vector.
                partial_language_vector[k, word_indices[k]:word_indices[k+1]] += language_vector[j][word_indices[k]:word_indices[k+1]]
                partial_reconstructed_language = meaning2word.get_partial_input_vector(vision_vector[j], k)
                reconstructed_language += partial_reconstructed_language

                # save KL divergence and distances between reconstruction and original.
                _kl_divergences[k].append(logger_helper.kl_divergence(partial_language_vector[k], partial_reconstructed_language))
                _distances[k].append(logger_helper.euclidean_distance(partial_language_vector[k], partial_reconstructed_language))

            # binarizes the reconstructed language vector and calculates precision, recall and accuracy.
            reconstructed_language = np.where(reconstructed_language > 0.35, 1.0, 0.0)
            _precisions.append(precision_score(language_vector[j], reconstructed_language))
            _recalls.append(recall_score(language_vector[j], reconstructed_language))
            _accuracies.append(accuracy_score(language_vector[j], reconstructed_language))

        # after reconstructing 100 inputs, saves average metrics for this model.
        _distances = np.array(_distances)
        _kl_divergences = np.array(_kl_divergences)
        distances.append(np.hstack((np.mean(_distances, axis=1), np.mean(_distances))))
        kl_divergences.append(np.hstack((np.mean(_kl_divergences, axis=1), np.mean(_kl_divergences))))
        precisions.append(np.mean(_precisions))
        recalls.append(np.mean(_recalls))
        accuracies.append(np.mean(_accuracies))
        print("iteration", iteration, ":", precisions[-1], accuracies[-1], accuracies[-1])

        meaning2word._sess.close()
        tf.reset_default_graph()

    # tidies the data and saves to a .csv.
    distances = np.array(distances)
    kl_divergences = np.array(kl_divergences)
    precisions = np.array(precisions).reshape((-1, 1))
    recalls = np.array(recalls).reshape((-1, 1))
    accuracies = np.array(accuracies).reshape((-1, 1))
    to_save = np.hstack((distances, kl_divergences, precisions, recalls, accuracies))
    np.savetxt(path_to_models + "meaning2word_metrics.csv", to_save, header=header)


if __name__ == "__main__":
    # generates metrics and plots them.
    path_to_models = "log-files/Jan-23_17.59.34_700_boot/"

    generate_metrics(700, path_to_models)
    load_and_plot(path_to_models)
