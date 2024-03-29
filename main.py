from SOM import SOM
from habibi import Hebbian
from data_generator import WordVector
from word_x_category import WordXCategory
from length_x_position import LengthXPosition
from utils import Logger, LoggerHelper
from datetime import datetime

import tensorflow as tf
import numpy as np


def set_sessions(objects, sess):
    """
    sets Tensorflow sessions to all objects using that library.
    necessary so that calculations can be shared between objects.
    
    objects: list or list of lists of objects
    sess: session to initialize objects with.
    """
    try:
        len(objects)
    except TypeError:
        return
    for o in objects:
        try:
            o._sess = sess
        except AttributeError:
            set_sessions(o, sess)


def prepare():
    """
    prepares and returns all data structures to be trained.
    """

    # initialize parameters.
    n_iter = 20                 # number of iterations performed per episode.
    n_episodes = 700            # number of all episodes. 20 * 700 = 14000
    n_categories = 4
    max_words = 4
    alpha1 = 0.3                # SOM learning rate.
    alpha2 = 0.2                # Hebbian links learning rate.
    alpha3 = 0.1                # Bootstrapping tables learning rate.
    input_threshold = 0.9       # Input threshold used to determine if word was "said".

    # initialize SOMs
    # c is empirically determined based on number of possible words for each SOM.
    position_som = SOM(16, 16, 3, c=10.0, alpha=alpha1, n_iterations=n_iter, n_episodes=n_episodes)
    size_som = SOM(16, 16, 1, c=16.0, alpha=alpha1, n_iterations=n_iter, n_episodes=n_episodes)
    color_som = SOM(16, 16, 3, c=10.0, alpha=alpha1, n_iterations=n_iter, n_episodes=n_episodes)
    shape_som = SOM(16, 16, 4, c=5.0, alpha=alpha1, n_iterations=n_iter, n_episodes=n_episodes)
    soms = [position_som, size_som, color_som, shape_som]

    # initialize WordVector a.k.a data generator.
    word_vector = WordVector()

    # initialize Hebbian links to/from each SOM.
    hebb1 = Hebbian(word_vector, position_som, alpha=alpha2, n_iterations=n_iter, n_episodes=n_episodes)
    hebb2 = Hebbian(word_vector, size_som, alpha=alpha2, n_iterations=n_iter, n_episodes=n_episodes)
    hebb3 = Hebbian(word_vector, color_som, alpha=alpha2, n_iterations=n_iter, n_episodes=n_episodes)
    hebb4 = Hebbian(word_vector, shape_som, alpha=alpha2, n_iterations=n_iter, n_episodes=n_episodes)
    hebbs = [hebb1, hebb2, hebb3, hebb4]

    # initialize bootstrapping tables.
    word_x_category = WordXCategory(word_vector, position_som.m*position_som.n, n_categories, alpha=alpha3)
    length_x_position = LengthXPosition(word_vector, word_x_category, n_categories, max_words, alpha=alpha3)

    # initialize Tensorflow session.
    sess = tf.Session()

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # set correct session to all previously generated objects.
    set_sessions([soms, hebbs, word_x_category, length_x_position], sess)
    return word_vector, soms, hebbs, word_x_category, length_x_position, n_iter, n_episodes, input_threshold


def train_iteration(word_vector, soms, hebbs, word_x_category, length_x_position, episode_no, input_threshold):
    """
    performs one iteration of training. Trains all SOMs, Hebbian links and bootstrapping tables. 
    returns training errors.

    word_vector: instance of WordVector.
    soms: list of SOMs.
    hebbs: list of Hebbian links instances.
    word_x_category: bootstrapping table, instance of WordXCategory.
    length_x_position: bootstrapping table, instance of LengthXPosition.
    episode_no: current episode number.
    input_threshold: threshold for determining which words were "said".
    """

    # initialize lists for storing weights and errors.
    som_weights = [None] * length_x_position.n_categories
    hebb_weights = [None] * length_x_position.n_categories
    som_error, hebb_error = [[] for i in range(length_x_position.n_categories)], [[] for i in range(length_x_position.n_categories)]

    # generate new input pair and error check.
    # since omitting size word can cause language input to have length 0, we need to check that.
    # current implementation with sum() only works on this dataset.
    string_vect, vision_vect = word_vector.generate_new_input_pair()
    if sum(string_vect) == 0:
        return train_iteration(word_vector, soms, hebbs, word_x_category, length_x_position, episode_no, input_threshold)

    # SPLITTING INPUTS BY MODALITY
    # this only works for exactly the type of input generated by this program. 
    # perhaps, a more general solution would be needed.
    input_indices = np.argwhere(string_vect > input_threshold)
    input_length = len(input_indices)

    # this creates an array of shape (input_length, len(string_vect)) and assigns each row to one word.
    # this way each words gets trained seperately.
    string_vect_split = np.zeros((input_length, len(string_vect)))
    string_vect_split[list(range(len(input_indices))), input_indices] = string_vect[input_indices]

    # get current LXP probabilites given input length.
    type_probabilities = length_x_position.get_type_probabilities(input_length)

    for cat in range(length_x_position.n_categories):
        # train each category's SOM and save errors.
        _som_error, som_weights[cat] = soms[cat].fit(vision_vect[cat], episode_no)
        som_error[cat].append(_som_error)

        for word_pos, one_word_vector in enumerate(string_vect_split):
            # for each word in input vector, train hebbian links for current category 
            # and save errors on the way. extract category probability before training.
            category_probability = type_probabilities[word_pos][cat]
            _hebb_error, hebb_weights[cat] = hebbs[cat].fit(one_word_vector, vision_vect[cat], category_probability, episode_no)
            hebb_error[cat].append(_hebb_error)

        # calculate words_x_categories table
        entropy = word_x_category.entropy(hebb_weights[cat][0])
        inv_entropy = word_x_category.inverse_entropy(entropy)
        word_x_category.update_words_x_categories(inv_entropy, cat)

    # ... & normalize
    word_x_category_table = word_x_category.normalize_all()

    # calculate lengths_x_positions & then normalize
    length_x_position.set_word_x_category_table(word_x_category_table)
    length_x_position.update_prob_array()
    length_x_position.normalize_all()

    return som_error, som_weights, hebb_error, hebb_weights


def train(word_vector, soms, hebbs, word_x_category, length_x_position, n_iter, n_episodes, input_threshold, save=False):
    """
    trains the model for n_episodes iterations.

    word_vector: instance of WordVector used for data generator.
    soms: list of SOMs.
    hebbs: list of Hebbian links instances.
    word_x_category: WordXCategory object.
    length_x_position: LengthXPostion object.
    n_iter: number of iterations per episode.
    n_episodes: number of episodes, i.e. repetitions of main for loop.
    input_threshold: threshold at which to consider the word was "said".
    save: whether to save the model in each episode.
    """
    
    # predefine some additional variables.
    train_type = ["position", "size", "color", "shape"]
    som_weights = [None] * len(train_type)
    hebb_weights = [None] * len(train_type)

    # initialize logger.
    now = datetime.utcnow().strftime("%b-%d_%H.%M.%S")
    logger = Logger(now)
    logger_helper = LoggerHelper(logger, train_type)

    for episode_no in range(n_episodes):
        # initialize som and hebbian error arrays.
        som_error, hebb_error = [[] for i in range(len(train_type))], [[] for i in range(len(train_type))]

        for iter_no in range(n_iter):
            # run an iteration of training and save errors.
            _som_error, som_weights, _hebb_error, hebb_weights = train_iteration(word_vector, soms, hebbs, word_x_category, length_x_position, episode_no, input_threshold)
            som_error = [s + _s for s, _s in zip(som_error, _som_error)]
            hebb_error = [h + _h for h, _h in zip(hebb_error, _hebb_error)]

        if save:
            # save all necessary objects for analysis.
            models = [som_weights, hebb_weights, length_x_position.lengths_x_positions_table, length_x_position.word_x_category_table]
            filenames = ["som_weights.pickle", "hebb_weights.pickle", "length_x_position.pickle", "word_x_category.pickle"]
            logger.write_models(models, filenames, episode_no)

        log_dict = dict()
        for cat in range(len(train_type)):
            # som error & hebbian links distances logging
            log_dict[train_type[cat] + " SOM error"] = np.mean(som_error[cat])
            log_dict["_" + train_type[cat] + " HEBB distances"] = np.mean(hebb_error[cat])

            # hebbian links entropies logging
            entropies_dict = logger_helper.hebb_entropies(hebb_weights[cat], train_type[cat], word_vector.cat_indices)
            log_dict.update(entropies_dict)

        # bootstrapping probability tables logging
        entropies_dict = logger_helper.wxc_entropies(length_x_position.word_x_category_table)
        log_dict.update(entropies_dict)
        entropies_dict = logger_helper.lxp_entropies(length_x_position.lengths_x_positions_table)
        log_dict.update(entropies_dict)

        # save metrics to file.
        log_dict["_episode"] = episode_no
        logger.log(log_dict)
        logger.write()


if __name__ == "__main__":
    # do things

    ######### training #########
    # collect objects from prepare and send them to train().
    word_vector, soms, hebbs, word_x_category, length_x_position, n_iter, n_episodes, input_threshold = prepare()
    train(word_vector, soms, hebbs, word_x_category, length_x_position, n_iter, n_episodes, input_threshold, save=True)
    exit()
