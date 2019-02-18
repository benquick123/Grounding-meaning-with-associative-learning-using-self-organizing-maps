import matplotlib.pyplot as plt
from matplotlib import rcParams
import pickle
import numpy as np
import os
from datetime import datetime
from data_generator import WordVector

path = ""
rcParams["font.family"] = "Latin Modern Roman"


def show_organization(_c, plot_title, mode="show"):
    """
    plots SOM organizations.
    """

    plt.rcParams["figure.figsize"] = (2.5, 2.5)
    plt.figure()
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
    if mode == "show":
        plt.show()
    if mode == "save":
        plt.savefig(path + plot_title + "_organization.svg")
        plt.close("all")


def show_organization_multi(h, vocabulary, plot_dim, plot_title, som_dim=16, scale=True, mode="show"):
    """
    plots n plots that show Hebbian links strength for each word in vocabulary.
    """
    plt.rcParams["figure.figsize"] = (12, 5)
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
        subplts[i].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        subplts[i].tick_params(axis="y", which="both", right=False, left=False, labelleft=False)

    plt.suptitle(plot_title, y=0.65)
    if mode == "show":
        plt.show()
    if mode == "save":
        plt.savefig((path + plot_title + "_organization_multi.svg").replace(">", ""))
        plt.close("all")


def show_organization_multi_v2(h, plot_dim, plot_title, voc_dim=16, scale=True, mode="show"):
    """
    plots m*n plots that show activations of words from each cell in SOM.
    """
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
    if mode == "show":
        plt.show()
    if mode == "save":
        plt.savefig((path + plot_title + "_organization_multi_v2.svg").replace(">", ""))
        plt.close("all")


def plot_SOMs(filename, mode="show"):
    """
    goes through all SOMs in specified filename and plots organizations.
    """
    # PLOTTING SOMS
    _s = pickle.load(open(filename, "rb"))[0]
    _s = (_s + 1) / 2
    show_organization(_s, "Position SOM", mode=mode)

    _s = pickle.load(open(filename, "rb"))[1]
    show_organization(_s, "Size SOM", mode=mode)
    _s = pickle.load(open(filename, "rb"))[2]
    show_organization(_s, "Color SOM", mode=mode)
    _s = pickle.load(open(filename, "rb"))[3]
    show_organization(_s, "Type SOM", mode=mode)


def plot_WM(filename, mode="show"):
    """
    goes through W-->M Hebbian links for all categories.
    it plots both 1x16, as well as 16x16 graphs.
    """
    word_vector = WordVector()
    # PLOTTING W --> M
    _h = pickle.load(open(filename, "rb"))[0]
    _h_WM = _h[0]
    show_organization_multi(_h_WM, word_vector.reverse_dict, (1, 16), "L --> V positions plot", mode=mode)
    show_organization_multi_v2(_h_WM.transpose(), (16, 16), "L --> V positions plot", mode=mode)

    _h = pickle.load(open(filename, "rb"))[1]
    _h_WM = _h[0]
    show_organization_multi(_h_WM, word_vector.reverse_dict, (1, 16), "L --> V sizes plot", mode=mode)
    show_organization_multi_v2(_h_WM.transpose(), (16, 16), "L --> V sizes plot", mode=mode)

    _h = pickle.load(open(filename, "rb"))[2]
    _h_WM = _h[0]
    show_organization_multi(_h_WM, word_vector.reverse_dict, (1, 16), "L --> V colors plot", mode=mode)
    show_organization_multi_v2(_h_WM.transpose(), (16, 16), "L --> V colors plot", mode=mode)

    _h = pickle.load(open(filename, "rb"))[3]
    _h_WM = _h[0]
    show_organization_multi(_h_WM, word_vector.reverse_dict, (1, 16), "L --> V types plot", mode=mode)
    show_organization_multi_v2(_h_WM.transpose(), (16, 16), "L --> V types plot", mode=mode)


def plot_MW(filename, mode="show"):
    """
    plots M--W Hebbian links in both 1x16 and 16x16 formats.
    """
    word_vector = WordVector()
    # PLOTTING M --> W
    _h = pickle.load(open(filename, "rb"))[0]
    _h_MW = _h[1]
    show_organization_multi(_h_MW.transpose(), word_vector.reverse_dict, (1, 16), "V --> L positions plot", mode=mode)
    show_organization_multi_v2(_h_MW, (16, 16), "V --> L positions plot", mode=mode)

    _h = pickle.load(open(filename, "rb"))[1]
    _h_MW = _h[1]
    show_organization_multi(_h_MW.transpose(), word_vector.reverse_dict, (1, 16), "V --> L sizes plot", mode=mode)
    show_organization_multi_v2(_h_MW, (16, 16), "V --> L sizes plot", mode=mode)

    _h = pickle.load(open(filename, "rb"))[2]
    _h_MW = _h[1]
    show_organization_multi(_h_MW.transpose(), word_vector.reverse_dict, (1, 16), "V --> L colors plot", mode=mode)
    show_organization_multi_v2(_h_MW, (16, 16), "V --> L colors plot", mode=mode)

    _h = pickle.load(open(filename, "rb"))[3]
    _h_MW = _h[1]
    show_organization_multi(_h_MW.transpose(), word_vector.reverse_dict, (1, 16), "V --> L types plot", mode=mode)
    show_organization_multi_v2(_h_MW, (16, 16), "V --> L types plot", mode=mode)


def entropy_plotting(filename, title="", mode="show"):
    """ plots entropies. mainly used for plotting WXC table """

    plt.rcParams["figure.figsize"] = (6, 3)
    plt.figure()
    img = [[] for j in range(4)]
    entropy_weights = pickle.load(open(filename, "rb"))
    for i, w in enumerate(entropy_weights):
        for _w in w:
            img[i].append([_w, _w, _w])
    plt.xticks(np.arange(0, len(img[0])), np.arange(0, len(img[0])))
    plt.imshow(img)
    plt.title(title)
    if mode == "show":
        plt.show()
    if mode == "save":
        plt.savefig(path + "wxc_entropy.svg")
        plt.close("all")


def type_plotting(filename, title="", mode="show"):
    """ plotting of LXP tables """

    plt.rcParams["figure.figsize"] = (10, 3)
    type_weights = pickle.load(open(filename, "rb"))
    f, subplts = plt.subplots(ncols=len(type_weights), sharex='col', sharey='row')

    for i in range(4):
        v = type_weights[i]
        img = []
        for j, el in enumerate(v):
            a = np.array(list(el) * 3).reshape((-1, 4)).transpose()
            img.append(a)
        subplts[i].imshow(img)
        subplts[i].set_title("phrase length " + str(i+1))
    plt.suptitle(title, y=0.92)

    if mode == "show":
        plt.show()
    if mode == "save":
        plt.savefig(path + "lxp_tables.svg")
        plt.close("all")


def plot_per_epoch(data, labels, title="", ylim=None, mode="show", path="Plotting/"):
    """ function for plotting changing values through time """
    plt.rcParams["figure.figsize"] = (9, 5)
    for row, label in zip(data, labels):
        r = np.array(range(len(row)))*20
        plt.plot(r, row, label=label)
    plt.title(title)
    plt.xlabel("episode number")
    if ylim is not None:
        plt.ylim(ylim)
    else:
        plt.ylim(0)
    plt.xlim(0, len(data[0])*20)
    plt.legend()

    if mode == "show":
        plt.show()
    if mode == "save":
        plt.savefig(path + title + "_per_epoch.svg")
        plt.close("all")


def plot_data_distributions(mode="show"):
    """ plotting of data generation statistics """
    labels = ["left", "right", "top", "bottom", "big", "small", "red", "green", "blue", "purple", "black", "white", "cube", "sphere", "cylinder", "cone"]
    data1 = np.loadtxt("Plotting/data_distr_1.csv")
    barwidth = 0.3
    plt.rcParams["figure.figsize"] = (11, 5)
    plt.bar(np.arange(0, len(data1[0])), data1[0], width=barwidth, label="Non-tweaked probability  of size words")
    plt.bar(np.arange(0, len(data1[1])) + barwidth, data1[1], width=barwidth, label="Reduced probability of size words (p=0.5)")
    plt.xticks(np.arange(0, len(data1[1])) + (barwidth / 2), labels)
    plt.legend()

    print("Data mean (old vs. new):")
    print(np.mean(data1, axis=1), np.std(data1, axis=1))
    if mode == "show":
        plt.show()
    if mode == "save":
        plt.savefig(path + "data_probabilities.svg")


if __name__ == "__main__":
    # initializes matplotlib parameters and speficies filenames.
    rcParams["font.family"] = "Latin Modern Roman"
    mode = "save"
    hebb_f = "log-files/Jan-23_17.59.34_700_boot/Episodes/Episode 699/hebb_weights.pickle"
    som_f = "log-files/Jan-23_17.59.34_700_boot/Episodes/Episode 699/som_weights.pickle"
    type_f = "log-files/Jan-23_17.59.34_700_boot/Episodes/Episode 699/length_x_position.pickle"
    entropy_f = "log-files/Jan-23_17.59.34_700_boot/Episodes/Episode 699/word_x_category.pickle"

    # creates plotting directory.
    if mode == "save":
        path = "Plotting/" + hebb_f.split("/")[1] + "/"
        os.makedirs(path, exist_ok=True)

    # calls plotting functions.
    plot_data_distributions(mode=mode)

    entropy_plotting(entropy_f, "WXC table", mode)
    type_plotting(type_f, "LXP tables", mode)

    plot_SOMs(som_f, mode)
    plot_WM(hebb_f, mode)
    plot_MW(hebb_f, mode)

    exit()
