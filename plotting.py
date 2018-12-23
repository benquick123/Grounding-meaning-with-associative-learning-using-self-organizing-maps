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
    _s = (_s + 1) / 2
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


def type_plotting(filename):
    type_weights = pickle.load(open(filename, "rb"))
    f, subplts = plt.subplots(nrows=len(type_weights))

    for i in range(4):
        v = type_weights[i]
        img = []
        for j, el in enumerate(v):
            a = np.array(list(el) * 3).reshape((-1, 4)).transpose()
            img.append(a)
        subplts[i].imshow(img)

    plt.show()


def plot_per_epoch(data, labels, title="", ylim=None):
    for row, label in zip(data, labels):
        r = list(range(len(row)))
        plt.plot(r, row, label=label)
    plt.title(title)
    plt.xlabel("epoch #")
    if ylim is not None:
        plt.ylim(ylim)
    else:
        plt.ylim(0)
    plt.xlim(0, len(data[0]))
    plt.legend()
    plt.show()


def plot_multiple_per_epoch(filename):
    som_errors, hebb_dist, wm_entropies, mw_entropies, wxc_learn, txp_learn = pickle.load(open(filename, "rb"))
    labels = ["position", "size", "color", "type"]

    # SOM errors
    plot_per_epoch(som_errors, labels, "Average error per epoch (1000 epochs)")

    # HEBB distances
    plot_per_epoch(hebb_dist, labels, "Distance between W -> M and M -> W hebbian weights")

    # HEBB W-M entropies
    wm_entropies = [[np.mean(wm_entropies_cat)
                     for wm_entropies_cat in wm_entropies[i]] for i in range(len(wm_entropies))]
    # wm_entropies = [[entropy([wm_entropies_cat])
    #                  for wm_entropies_cat in wm_entropies[i]] for i in range(len(wm_entropies))]
    plot_per_epoch(wm_entropies, labels, "W -> M average entropy", ylim=(0.75, 1.0))

    # HEBB M-W entropies
    mw_entropies = [[np.mean(mw_entropies_cat)
                     for mw_entropies_cat in mw_entropies[i]] for i in range(len(mw_entropies))]
    # mw_entropies = [[entropy([mw_entropies_cat])
    #                  for mw_entropies_cat in mw_entropies[i]] for i in range(len(mw_entropies))]
    plot_per_epoch(mw_entropies, labels, "M --> W average entropy", ylim=(0.75, 1.0))

    # WORD_x_CATEGORY entropies
    wxc_learn = np.array(wxc_learn)
    # wxc_learn = [entropy(d) for d in wxc_learn[np.arange(0, wxc_learn.shape[0], 4).tolist()]]
    wxc_learn = [entropy(d) for d in wxc_learn]
    plot_per_epoch([wxc_learn], ["w_x_c average entropies"], title="Bootstraping (word x category) average entropies",
                   ylim=(0.6, 1.0))

    # TYPE_x_PROBABILITY entropies
    txp_learn = np.array(txp_learn)
    # txp_learn = txp_learn[np.arange(0, txp_learn.shape[0], 4).tolist()]
    # _txp_learn = [[], [], [], []]
    txp_learn = np.array([[entropy(epoch[i]) for i in range(len(epoch))] for epoch in txp_learn]).transpose()
    plot_per_epoch(txp_learn, ["phrase length: 1", "phrase length: 2", "phrase length: 3", "phrase length: 4"],
                   title="Bootstraping (type x probability) average entropies", ylim=(0.4, 1.0))



def entropy(data):
    data = np.array(data)
    a = []
    for row in data:
        a.append(-np.sum(row * (np.log(row) / np.log(len(row)))))
    return np.mean(a)


if __name__ == "__main__":
    hebb_f = "weights/hebb_weights_1544576700.pickle"
    som_f = "weights/som_weights_1544576700.pickle "
    type_f = "weights/type_x_probability_1544576700.pickle"
    entropy_f = "weights/word_x_category_1544576700.pickle"
    error_f = "errors_w_bootstraping_1544576700.pickle"

    # plot_multiple_per_epoch(error_f)
    # exit()

    # entropy_plotting(entropy_f)
    # type_plotting(type_f)
    # exit()

    plot_SOMs(som_f)
    exit()
    plot_WM(hebb_f)
    plot_MW(hebb_f)

    exit()
