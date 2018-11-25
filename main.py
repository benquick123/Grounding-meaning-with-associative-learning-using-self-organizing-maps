from SOM import SOM
from habibi import Hebbian
from data_generator import WordVector
from word_x_category import WordXCategory
from type_x_probability import TypeXProbability

import pickle
import time
import tensorflow as tf
import numpy as np


def prepare():
    n_iter = 50
    n_epochs = 200
    n_categories = 4
    max_words = 4
    alpha = 0.6
    input_threshold = 0.9

    # initialize SOMs
    position_som = SOM(16, 16, 3, alpha=alpha, n_iterations=n_iter, n_epochs=n_epochs)
    size_som = SOM(16, 16, 1, alpha=alpha, n_iterations=n_iter, n_epochs=n_epochs)
    color_som = SOM(16, 16, 3, alpha=alpha, n_iterations=n_iter, n_epochs=n_epochs)
    shape_som = SOM(16, 16, 4, alpha=alpha, n_iterations=n_iter, n_epochs=n_epochs)
    soms = [position_som, size_som, color_som, shape_som]
    # soms = [size_som]

    word_vector = WordVector()

    hebb1 = Hebbian(word_vector, position_som, 0, n_iterations=n_iter, n_epochs=n_epochs)
    hebb2 = Hebbian(word_vector, size_som, 1, n_iterations=n_iter, n_epochs=n_epochs)
    hebb3 = Hebbian(word_vector, color_som, 2, n_iterations=n_iter, n_epochs=n_epochs)
    hebb4 = Hebbian(word_vector, shape_som, 3, n_iterations=n_iter, n_epochs=n_epochs)
    hebbs = [hebb1, hebb2, hebb3, hebb4]
    # hebbs = [hebb2]

    word_x_category = WordXCategory(word_vector, position_som.m*position_som.n, n_categories)
    type_x_probability = TypeXProbability(word_vector, word_x_category, n_categories, max_words)

    sess = tf.Session()

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    set_sessions([soms, hebbs, word_x_category, type_x_probability], sess)
    return word_vector, soms, hebbs, word_x_category, type_x_probability, n_iter, n_epochs, input_threshold


def train_iteration(word_vector, soms, hebbs, word_x_category, type_x_probability, epoch_no, input_threshold):
    som_weights = [None, None, None, None]
    hebb_weights = [None, None, None, None]
    som_error, hebb_error = 0, 0

    string_vect, vision_vect = word_vector.generate_new_input_pair()

    input_indices = np.argwhere(string_vect > input_threshold)
    input_length = len(input_indices)

    string_vect_split = np.zeros((input_length, len(string_vect)))
    string_vect_split[list(range(len(input_indices))), input_indices] = string_vect[input_indices]

    type_probabilities = type_x_probability.get_type_probabilities(input_length)

    for cat in range(type_x_probability.n_categories):
        som_error, som_weights[cat] = soms[cat].fit(vision_vect, epoch_no)

        for word_pos, one_word_vector in enumerate(string_vect_split):
            category_probability = type_probabilities[word_pos][cat]
            hebb_error, hebb_weights[cat] = hebbs[cat].fit(one_word_vector, vision_vect, category_probability, epoch_no)

        # calculate words_x_categories table
        inv_entropy = word_x_category.inverse_entropy(word_x_category.entropy(hebb_weights[cat][0]))
        word_x_category.update_words_x_categories(inv_entropy, cat, alpha=0.25)

    # ... & normalize
    word_x_category.normalize_all()

    # calculate types_x_probabilities & then normalize
    type_x_probability.update_prob_array(alpha=0.25)
    type_x_probability.normalize_all()
    return som_error, som_weights, hebb_error, hebb_weights


def train(word_vector, soms, hebbs, word_x_category, type_x_probability, n_iter, n_epochs, input_threshold, save=False):
    # for loops bla bla bla
    # get input vectors

    train_type = ["position", "size", "color", "shape"]
    som_weights = [None, None, None, None]
    hebb_weights = [None, None, None, None]
    som_error, hebb_error = 0, 0

    for epoch_no in range(n_epochs):
        print("### Epoch:", epoch_no, "/", n_epochs, "###")

        for iter_no in range(n_iter):
            som_error, som_weights, hebb_error, hebb_weights = train_iteration(word_vector, soms, hebbs, word_x_category, type_x_probability, epoch_no, input_threshold)

        for cat in range(type_x_probability.n_categories):
            print("SOM error -", train_type[cat], "-", som_error)
            print("HEBB error -", train_type[cat], "-", hebb_error)

    if save:
        pickle.dump(som_weights, open("som_weights_" + str(int(time.time())) + ".pickle", "wb"))
        pickle.dump(hebb_weights, open("hebb_weights_" + str(int(time.time())) + ".pickle", "wb"))


def set_sessions(objects, sess):
    try:
        a = len(objects)
    except TypeError:
        return
    for o in objects:
        try:
            o._sess = sess
        except AttributeError:
            set_sessions(o, sess)


if __name__ == "__main__":
    # do things

    ######### training #########
    word_vector, soms, hebbs, word_x_category, type_x_probability, n_iter, n_epochs, input_threshold = prepare()
    train(word_vector, soms, hebbs, word_x_category, type_x_probability, n_iter, n_epochs, save=True)
    exit()
