import tensorflow as tf
import numpy as np
import pickle
from utils import LoggerHelper

from data_generator import WordVector
import plotting


class Word2Meaning(object):
    """
    for reconstruction to work with current implementation, all SOMs must have the same size.
    """

    def __init__(self, som_filename, hebb_filename, wxc_filename, input_vector_dim, som_dim, n_categories):
        """
        initializes word2meaning object variables and Tensorflow operations.

        som_filename: filename of all SOMs weights.
        hebb_filename: filename of Hebbian links weights.
        wxc_filename: filename of word_x_category table weights.
        input_vector_dim: dimension of language vector.
        som_dim: number of cells is SOM.
        n_categories: number of all categories.
        """

        self.n_categories = n_categories
        self.input_vector_dim = input_vector_dim
        self.som_dim = som_dim

        # load all weights.
        hebb_weights = pickle.load(open(hebb_filename, "rb"))
        self.hebb_weights = np.array([h[0] for h in hebb_weights])
        self.som_filename = som_filename
        self.som_weights = pickle.load(open(self.som_filename, "rb"))[0]
        self.wcx_weights = pickle.load(open(wxc_filename, "rb"))

        self._sess = tf.Session()

        # initialize placeholder for choosing correct cell and weights.
        self.som_number = tf.placeholder(tf.int32)
        self.input_vector = tf.placeholder(tf.float32, [input_vector_dim])

        # get the slice for current category from WXC table and multiply it with language vector.
        wxc_slice = tf.reshape(tf.slice(self.wcx_weights, [self.som_number, 0], [1, -1]), [-1])
        input_vector = tf.multiply(self.input_vector, wxc_slice)

        # get the slice going to appropriate SOM depending on current category (=som_number).
        hebb_slice = tf.reshape(tf.slice(self.hebb_weights, [self.som_number, 0, 0], [1, -1, -1]), [input_vector_dim, -1])
        # find best matching unit according to most active Hebbian link after multiplying with input_vector.
        partial_input_vector_index = tf.argmax(tf.reduce_sum(tf.multiply(hebb_slice, tf.cast(tf.transpose(tf.stack([input_vector for i in range(self.som_dim)])), dtype=tf.float32)), axis=0))
        self.som_bmu_index_op = partial_input_vector_index

        self._sess.run(tf.global_variables_initializer())

    def get_vision_vector(self, input_vector):
        """
        calculates and returns reconstructed vision vector based on language input.

        input_vector: language vector from which vision is to be reconstructed.
        """
        vision_vector = []
        for i in range(self.n_categories):
            # for every category, find best matching unit in SOM and save append its weights to vision_vector.
            self.som_weights = pickle.load(open(self.som_filename, "rb"))[i]
            bmu_index = self._sess.run(self.som_bmu_index_op, feed_dict={self.som_number: i, self.input_vector: input_vector})
            som_bmu = self.som_weights[bmu_index]
            vision_vector.append(som_bmu.tolist())
        return vision_vector


def load_and_plot(header, path_to_models):
    """ 
    loads .csv file and plots reconstruction distances.
    
    path_to_models: path to .csv file to be plotted.
    """

    header = header.split("\t")[:4]
    distances = np.loadtxt(path_to_models + "word2meaning_distances.csv")
    save_path = "Plotting/" + path_to_models.split("/")[1] + "/" + "l2v"
    plotting.plot_per_epoch(np.transpose(distances)[:4], header, mode="save", path=save_path)


def generate_distances(n, header, path_to_models):
    """
    goes through all models saved during training and calculates average distances between
    100 generated and reconstructed vision input.

    header: header to be used in saved file.
    path_to_models: path to models that we want to get reconstruction distances from.
    """

    # initialize structures necessary for reconstruction.
    word_vector = WordVector()
    logger_helper = LoggerHelper(None, None)
    language_vector = []
    vision_vector = []

    # generate 100 examples of inputs pairs.
    for i in range(100):
        _lang, _vis = word_vector.generate_new_input_pair(full=True)
        language_vector.append(_lang)
        vision_vector.append(_vis)

    # initialize global distances array.
    distances = []
    for iteration in range(n):
        # construct filenames and inizialize word2meaning instance.
        hebb_f = path_to_models + "Episodes/Episode " + str(iteration) + "/hebb_weights.pickle"
        som_f = path_to_models + "Episodes/Episode " + str(iteration) + "/som_weights.pickle"
        wxc_f = path_to_models + "Episodes/Episode " + str(iteration) + "/word_x_category.pickle"
        word2meaning = Word2Meaning(som_f, hebb_f, wxc_f, word_vector.dim, 256, 4)


        _distances = [[], [], [], []]
        for i in range(100):
            # for every generated input pair, reconstruct whole vision vector.
            reconstructed_vision = word2meaning.get_vision_vector(language_vector[i])
            for j, (_rec_vis, _vis) in enumerate(zip(reconstructed_vision, vision_vector[i])):
                # for every part of reconstructed vision vector, calculate distance 
                # between original and reconstruction. add the measure to local distances array.
                _distances[j].append(logger_helper.euclidean_distance(np.array(_rec_vis), np.array(_vis)))
        _distances = np.array(_distances)
        # calculate mean distances for all 100 reconstructions for every part of vision vector
        # also add average of distances for whole reconstructed vision vector.
        distances.append(np.hstack((np.mean(_distances, axis=1), np.mean(_distances))))
        print("iteration", iteration, ":", distances[-1][4])
    
    # save the array to csv.
    distances = np.array(distances)
    np.savetxt(path_to_models + "word2meaning_distances.csv", distances, header=header)


def plot_among_many(path_to_models):
    """
    plots accuracies for choosing-among-20 experiment.

    path_to_models: path to .csv file.
    """

    label = ["Choosing accuracy"]
    accuracies = np.loadtxt(path_to_models + "word2meaning_choose_boot.csv")
    print(accuracies)
    save_path = "Plotting/" + path_to_models.split("/")[1] + "/" + "l2v_choose_boot"
    plotting.plot_per_epoch([accuracies], label, ylim=(0.0, 1.0), mode="save", path=save_path)


def create_unique_pairs(n, word_vector):
    """
    creates n unique pairs.

    n: number of pairs to create.
    word_vector: WordVector instance.
    """

    pairs = dict()
    i = 0
    while i < n:
        l, v = word_vector.generate_new_input_pair(full=True)
        w = word_vector.generate_string_from_vector(l)
        if w not in pairs:
            pairs[i] = (w, l, v)
            i += 1
    return pairs


def choose_among_many(n, path_to_models):
    """
    traverse through all models saved during training, computes reconstructions and chooses 
    the closest original vision vector. saves the accuracy levels for every model.

    n: number of all models.
    path_to_models: path to models saved during training.
    """

    word_vector = WordVector()
    logger_helper = LoggerHelper(None, None)
    m = 20

    # generate 500 examples of inputs pairs.
    x_pairs = []
    for i in range(500):
        pairs = create_unique_pairs(m, word_vector)
        rand_i = np.random.randint(0, m)
        x_pairs.append((rand_i, pairs))

    correct_p = []
    for i in range(n):
        # construct filenames for current episode and initialize word2meaning object.
        hebb_f = path_to_models + "Episodes/Episode " + str(i) + "/hebb_weights.pickle"
        som_f = path_to_models + "Episodes/Episode " + str(i) + "/som_weights.pickle"
        wxc_f = path_to_models + "Episodes/Episode " + str(i) + "/word_x_category.pickle"
        word2meaning = Word2Meaning(som_f, hebb_f, wxc_f, word_vector.dim, 256, 4)
        _correct_p = []

        for j in range(500):
            # for every input previously generated, compute reconstructed vision vector.
            rand_i, pairs = x_pairs[j]
            reconstructed_vision = word2meaning.get_vision_vector(pairs[rand_i][1])
            v1 = np.array([_v for _l in reconstructed_vision for _v in _l])
            distances = []

            # calculates distances between reconstructed vector and all 20 original vision vectors.
            for k in range(m):
                v2 = np.array([_v for _l in pairs[k][2] for _v in _l])
                distances.append(logger_helper.euclidean_distance(v1, v2))
            
            # chooses the closest and checks if it is the correct one.
            closest = np.argmin(distances)
            _correct_p.append(1.0 if closest == rand_i else 0.0)
        
        # calculates mean of all 500 reconstructions.
        _correct_p = np.mean(_correct_p)
        print("Iteration", i, ":", _correct_p)
        correct_p.append(_correct_p)

    # saves the accuracies.
    np.savetxt(path_to_models + "word2meaning_choose_boot.csv", np.array(correct_p[::-1]))


if __name__ == "__main__":
    path_to_models = "log-files/Jan-23_17.59.34_700_boot/"
    header = "position\tsize\tcolor\ttype\taverage"

    # generates and plots all necessary metrics and plots.
    generate_distances(700, header, path_to_models)
    load_and_plot(header, path_to_models)
    choose_among_many(700, path_to_models)
    plot_among_many(path_to_models)

    # this code is for plotting comparison between bootstrapping on or off.
    acc1 = np.loadtxt(path_to_models + "word2meaning_choose.csv")
    acc2 = np.loadtxt(path_to_models + "word2meaning_choose_no_boot.csv")
    labels = ["Accuraccy when using WXC table", "Accuracy without using WXC table"]
    plotting.plot_per_epoch([acc1, acc2], labels, ylim=(0.0, 1.0), mode="save", path="Plotting/" + path_to_models.split("/")[1] + "/l2v_choose_compare")

