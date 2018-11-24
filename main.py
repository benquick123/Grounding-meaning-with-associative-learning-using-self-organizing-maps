from SOM import SOM
from habibi import Hebbian
from data_generator import WordVector
from word_x_category import WordXCategory

import pickle
import time
import tensorflow as tf
import numpy as np


def prepare():
    n_iter = 50
    n_epochs = 200
    n_categories = 4
    alpha = 0.6

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

    sess = tf.Session()

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    set_sessions([soms, hebbs], sess)
    return word_vector, soms, hebbs, word_x_category, n_iter, n_epochs


def train(word_vector, soms, hebbs, word_x_category, n_iter, n_epochs, save=False):
    # for loops bla bla bla
    # get input vectors

    train_type = ["position", "size", "color", "shape"]
    som_weights = [None, None, None, None]
    hebb_weights = [None, None, None, None]
    word_x_category_table = None
    lenght_x_probability = None

    for i in range(n_epochs):
        print("### Epoch:", i, "/", n_epochs, "###")
        string_vectors, vision_vectors = [], []
        for j in range(n_iter):
            string_vector, vision_vector = word_vector.generate_new_input_pair()
            string_vectors.append(string_vector)
            vision_vectors.append(vision_vector)

        vision_vectors = np.array(vision_vectors).transpose()
        # vision_vectors = [vision_vectors[0]]

        for j, (som, som_input, hebb) in enumerate(zip(soms, vision_vectors, hebbs)):
            # som_error, som_weights[j] = som.train(som_input, i)
            som_error, som_weights[j], hebb_error, hebb_weights[j] = hebb.train(string_vectors, som_input, i, train_som=True)
            inv_entropy = word_x_category.inverse_entropy(word_x_category.entropy(hebb_weights[j][0]))
            word_x_category_table = word_x_category.update_words_x_categories(inv_entropy, j, alpha=0.4)


            print("SOM error -", train_type[j], "-", som_error)
            print("HEBB error -", train_type[j], "-", hebb_error)
        print()

        # normalize word X category table
        word_x_category_table = word_x_category.normalize_all(word_x_category_table)

    if save:
        pickle.dump(som_weights, open("som_weights_" + str(int(time.time())) + ".pickle", "wb"))
        pickle.dump(hebb_weights, open("hebb_weights_" + str(int(time.time())) + ".pickle", "wb"))
        # pickle.dump(word_x_category_table, open("word_X_category_" + str(int(time.time())) + ".pickle", "wb"))


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
    word_vector, soms, hebbs, word_x_category, n_iter, n_epochs = prepare()
    train(word_vector, soms, hebbs, word_x_category, n_iter, n_epochs, save=True)
    exit()
