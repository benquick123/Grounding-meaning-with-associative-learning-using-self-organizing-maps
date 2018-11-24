import matplotlib.pyplot as plt
import pickle
import numpy as np
from data_generator import WordVector


def show_organization(_c, plot_title):
    if _c is None:
        _c = pickle.load(open("som_weights_1539871205.pickle", "rb"))[3]

    if len(_c[0]) < 3:
        _c = np.array([_c for j in range(3)]).transpose()[0]

    m, n = _c.shape
    # print(m, n)
    dim = int(np.sqrt(m))

    img = [[] for i in range(dim)]
    for i in range(m):
        img[int(i / dim)].append(_c[i][:3])

    plt.title(plot_title)
    plt.imshow(img)
    plt.show()


def show_organization_multi(h, vocabulary, plot_dim, plot_title, som_dim=16, scale=True):
    f, subplts = plt.subplots(plot_dim[0], plot_dim[1], sharex='col', sharey='row')
    if scale:
        h = (h - np.min(h)) / (np.max(h) - np.min(h))

    for i, row in enumerate(h):
        img = [[] for j in range(som_dim)]
        row = [[j, j, j] for j in row]
        for k in range(len(row)):
            img[int(k / som_dim)].append(row[k])
        subplts[i].imshow(img)
        subplts[i].set_title(vocabulary[i])

    plt.suptitle(plot_title, y=0.58)
    plt.show()


def show_organization_multi_v2(h, plot_dim, plot_title, voc_dim=16, scale=True):
    plt.rcParams["figure.figsize"] = (10, 10)
    f, subplts = plt.subplots(plot_dim[0], plot_dim[1], sharex='col', sharey='row')
    if scale:
        h = (h - np.min(h)) / (np.max(h) - np.min(h))

    for i, row in enumerate(h):
        row = [[j, j, j] for j in row]
        img = np.stack([row for i in range(voc_dim)])
        subplts[int(i / plot_dim[0]), i % plot_dim[0]].imshow(img)
        subplts[int(i / plot_dim[0]), i % plot_dim[0]].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        subplts[int(i / plot_dim[0]), i % plot_dim[0]].tick_params(axis="y", which="both", right=False, left=False, labelleft=False)

    plt.suptitle(plot_title, y=0.92)
    plt.show()


def show_mapping_to_word_vector(h):
    _hebbian_sums = np.sum(h)
    # _hebbian_sums = h[128, :]
    _hebbian_sums = (h - np.min(h)) / (np.max(h) - np.min(h))

    print(_hebbian_sums.shape)
    img = [[] for i in range(len(h))]
    for i in range(len(h)):
        for j in range(len(h[0])):
            img[i].append([_hebbian_sums[i, j] for k in range(3)])

    plt.imshow(img)
    plt.show()


def plot_SOMs(filename):
    # PLOTTING SOMS
    _s = pickle.load(open(filename, "rb"))[0]
    show_organization(_s, "Position SOM")
    _s = pickle.load(open(filename, "rb"))[1]
    show_organization(_s, "Size SOM")
    _s = pickle.load(open(filename, "rb"))[2]
    show_organization(_s, "Color SOM")
    _s = pickle.load(open(filename, "rb"))[3]
    show_organization(_s, "Shapes SOM")


def plot_WM(filename):
    word_vector = WordVector()
    # PLOTTING W --> M
    _h = pickle.load(open(filename, "rb"))[0]
    _h_WM = _h[0]
    show_organization_multi(_h_WM, word_vector.reverse_dict, (1, 16), "W --> M positions plot")
    show_organization_multi_v2(_h_WM.transpose(), (16, 16), "W --> M positions plot")

    _h = pickle.load(open(filename, "rb"))[1]
    _h_WM = _h[0]
    show_organization_multi(_h_WM, word_vector.reverse_dict, (1, 16), "W --> M sizes plot")
    show_organization_multi_v2(_h_WM.transpose(), (16, 16), "W --> M sizes plot")

    _h = pickle.load(open(filename, "rb"))[2]
    _h_WM = _h[0]
    show_organization_multi(_h_WM, word_vector.reverse_dict, (1, 16), "W --> M colors plot")
    show_organization_multi_v2(_h_WM.transpose(), (16, 16), "W --> M colors plot")

    _h = pickle.load(open(filename, "rb"))[3]
    _h_WM = _h[0]
    show_organization_multi(_h_WM, word_vector.reverse_dict, (1, 16), "W --> M shapes plot")
    show_organization_multi_v2(_h_WM.transpose(), (16, 16), "W --> M shapes plot")


def plot_MW(filename):
    word_vector = WordVector()
    # PLOTTING M --> W
    _h = pickle.load(open(filename, "rb"))[0]
    _h_MW = _h[1]
    show_organization_multi(_h_MW.transpose(), word_vector.reverse_dict, (1, 16), "M --> W positions plot")
    show_organization_multi_v2(_h_MW, (16, 16), "M --> W positions plot")

    _h = pickle.load(open(filename, "rb"))[1]
    _h_MW = _h[1]
    show_organization_multi(_h_MW.transpose(), word_vector.reverse_dict, (1, 16), "M --> W sizes plot")
    show_organization_multi_v2(_h_MW, (16, 16), "M --> W sizes plot")

    _h = pickle.load(open(filename, "rb"))[2]
    _h_MW = _h[1]
    show_organization_multi(_h_MW.transpose(), word_vector.reverse_dict, (1, 16), "M --> W colors plot")
    show_organization_multi_v2(_h_MW, (16, 16), "M --> W colors plot")

    _h = pickle.load(open(filename, "rb"))[3]
    _h_MW = _h[1]
    show_organization_multi(_h_MW.transpose(), word_vector.reverse_dict, (1, 16), "M --> W shapes plot")
    show_organization_multi_v2(_h_MW, (16, 16), "M --> W shapes plot")


def entropy_plotting(filename):
    img = [[] for j in range(4)]
    entropy_weights = pickle.load(open(filename, "rb"))
    for i, w in enumerate(entropy_weights):
        for _w in w:
            img[i].append([_w, _w, _w])
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    hebb_f = "hebb_weights_1542666542.pickle"
    som_f = "som_weights_1542666542.pickle"
    entropy_f = "word_X_category1542724910.pickle"

    entropy_plotting(entropy_f)
    exit()

    plot_SOMs(som_f)
    plot_WM(hebb_f)
    plot_MW(hebb_f)

    exit()
