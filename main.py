from SOM import SOM
from habibi import Hebbian

import tensorflow as tf


def train(som1, som2, hebb):
    # for loops bla bla bla
    # get input vectors
    input_vect1 = [1.0, -0.33, 0.5]
    input_vect2 = [1.0, 0.0, 0.0, 1.0]
    curr_iteration = 1

    w1 = som1.fit(input_vect1, curr_iteration)
    w2 = som2.fit(input_vect2, curr_iteration)

    w3 = hebb.fit(input_vect1, input_vect2, curr_iteration)

    print("SOM 1 weights:")
    print(w1)
    print("\nSOM2 weights:")
    print(w2)
    print("\nHEBB weights:")
    print(w3)


if __name__ == "__main__":
    # do things
    graph = tf.Graph()
    n_iter = 500

    # initialize SOMs
    som1 = SOM(5, 5, 3, graph, n_iterations=n_iter)
    som2 = SOM(3, 3, 4, graph, n_iterations=n_iter)

    # initialize hebbian
    hebb = Hebbian(som1, som2, graph, n_iterations=n_iter)

    train(som1, som2, hebb)
    exit()
