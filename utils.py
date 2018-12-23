import os
import shutil
import csv
import pickle
import numpy as np
from word_to_meaning import Word2Meaning
from meaning_to_word import Meaning2Word


class Logger(object):
    """ Simple training logger: saves to file and optionally prints to stdout """
    def __init__(self, now):
        """
        Args:
            logname: name for log (e.g. 'Hopper-v1')
            now: unique sub-directory name (e.g. date/time string)
        """
        self.path = "/".join(["log-files", now])
        os.makedirs(self.path)
        filenames = ["data_generator.py", "habibi.py", "SOM.py", "word_x_category.py", "length_x_position.py",
                     "meaning_to_word.py", "word_to_meaning.py", "main.py"]
        for filename in filenames:     # for reference
            shutil.copy(filename, self.path)
        path = os.path.join(self.path, 'log.csv')

        self.write_header = True
        self.log_entry = {}
        self.f = open(path, 'w')
        self.writer = None  # DictWriter created with first call to write() method

    def write_models(self, models, filenames, episode_n):
        folder_name = "Episode " + str(episode_n)
        os.makedirs(self.path + "/" + folder_name, exist_ok=True)
        for model, filename in zip(models, filenames):
            filename_path = self.path + "/" + folder_name + "/" + filename
            pickle.dump(model, open(filename_path, "wb"))

    def get_models(self, filenames, episode_n):
        """ gets the models writeen in episode_n. """
        folder_name = "Episode " + str(episode_n)
        path = self.path + "/" + folder_name + "/"
        models = []
        for filename in filenames:
            models.append(pickle.load(open(path + filename, "rb")))
        return models

    def write(self, display=True):
        """ Write 1 log entry to file, and optionally to stdout
        Log fields preceded by '_' will not be printed to stdout

        Args:
            display: boolean, print to stdout
        """
        if display:
            self.disp(self.log_entry)
        if self.write_header:
            fieldnames = [x for x in self.log_entry.keys()]
            self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
            self.writer.writeheader()
            self.write_header = False
        self.writer.writerow(self.log_entry)
        self.log_entry = {}

    @staticmethod
    def disp(log):
        """Print metrics to stdout"""
        log_keys = [k for k in log.keys()]
        log_keys.sort()
        print('***** Episode {} *****'.format(log['_episode']))
        for key in log_keys:
            if key[0] != '_':  # don't display log items with leading '_'
                print('{:s}: {:.4g}'.format(key, log[key]))
        print('\n')

    def log(self, items):
        """ Update fields in log (does not write to file, used to collect updates.

        Args:
            items: dictionary of items to update
        """
        self.log_entry.update(items)

    def close(self):
        """ Close log file - log cannot be written after this """
        self.f.close()


class LoggerHelper(object):
    def __init__(self, logger, train_type, word_threshold=0.9):
        self.word_threshold = word_threshold
        self.logger = logger
        self.train_type = train_type
        self.word2meaning = None
        self.meaning2word = None

    @staticmethod
    def entropy(p):
        return -np.sum([p * (np.log(p) / np.log(len(p)))])

    @staticmethod
    def kl_divergence(p, q):
        kl = np.sum(np.where(p == 0.0, 0.0, p * np.log(p / q))) / np.log(p.shape[0])
        return kl

    @staticmethod
    def euclidean_distance(p, q):
        return np.sqrt(np.sum((p - q) ** 2))

    def hebb_entropies(self, hebb_weights, cat_name, cat_indices):
        d = dict()
        h_wm = hebb_weights[0]
        h_mw = hebb_weights[1]

        for i in range(len(cat_indices)):
            try:
                subsection_wm = h_wm[cat_indices[i]:cat_indices[i + 1], :]
                subsection_mw = h_mw[:, cat_indices[i]:cat_indices[i + 1]]
            except IndexError:
                subsection_wm = h_wm[cat_indices[i]:, :]
                subsection_mw = h_mw[:, cat_indices[i]:]

            _entropy_wm = np.mean([self.entropy(slice) for slice in subsection_wm])
            _entropy_mw = np.mean([self.entropy(slice) for slice in subsection_mw])

            key_wm = cat_name + " wm entropies - " + self.train_type[i]
            key_mw = cat_name + " mw entropies - " + self.train_type[i]
            if cat_name != self.train_type[i]:
                key_wm = "_" + key_wm
                key_mw = "_" + key_mw
            d[key_wm] = _entropy_wm
            d[key_mw] = _entropy_mw
        return d

    def wxc_entropies(self, wxc):
        d = dict()
        for row, cat in zip(wxc, self.train_type):
            key = cat + " word_x_category entropy"
            d[key] = self.entropy(row)
        return d

    def lxp_entropies(self, lxp):
        d = dict()
        for i in range(len(lxp)):
            key = "length: " + str(i + 1) + " length_x_position entropy"
            d[key] = np.mean([self.entropy(row) for row in lxp[i]])
        return d

    def input_reconstruction(self, word_vector, logger, episode_no, n_iter, som_dim):
        d = dict()
        input_pairs = word_vector.input_pairs_history[-n_iter:]
        som_filename = logger.path + "/Episode " + str(episode_no) + "/som_weights.pickle"
        hebb_filename = logger.path + "/Episode " + str(episode_no) + "/hebb_weights.pickle"
        wxc_filename = logger.path + "/Episode " + str(episode_no) + "/word_x_category.pickle"
        if self.word2meaning is None or self.meaning2word is None:
            self.word2meaning = Word2Meaning(som_filename, hebb_filename, wxc_filename, word_vector.dim, som_dim, len(word_vector.properties))
            self.meaning2word = Meaning2Word(hebb_filename, som_filename, wxc_filename, word_vector.dim, som_dim, len(word_vector.properties))

        vision_recreation_data = []
        language_recreation_data = []
        kl_divergences = [[] for i in range(self.meaning2word.n_categories)]
        language_distances = [[] for i in range(self.meaning2word.n_categories)]
        vision_distances = [[] for i in range(self.meaning2word.n_categories)]

        # for every input, calculate reconstruction.
        for language_vector, vision_vector in input_pairs:
            n_words = np.sum(language_vector > self.word_threshold)
            if n_words == 4:
                recreated_vision = self.word2meaning.get_vision_vector(language_vector)

                language_vector_sliced = np.zeros((n_words, word_vector.dim))
                recreated_language = np.zeros((n_words, word_vector.dim))

                # slice the language vector, so each word gets reconstructed separately.
                for i in range(self.meaning2word.n_categories):
                    try:
                        language_vector_sliced[i, word_vector.cat_indices[i]:word_vector.cat_indices[i + 1]] = language_vector[word_vector.cat_indices[i]:word_vector.cat_indices[i + 1]]
                    except IndexError:
                        language_vector_sliced[i, word_vector.cat_indices[i]:] = language_vector[word_vector.cat_indices[i]:]

                    partial_recreated_language = self.meaning2word.get_partial_input_vector(vision_vector, i)
                    recreated_language[i, :] = partial_recreated_language

                    kl_divergences[i].append(self.kl_divergence(language_vector_sliced[i, :], partial_recreated_language))
                    language_distances[i].append(self.euclidean_distance(language_vector_sliced[i, :], partial_recreated_language))
                    vision_distances[i].append(self.euclidean_distance(np.array(vision_vector[i]), np.array(recreated_vision[i])))

                language_recreation_data.append((language_vector_sliced, recreated_language))
                vision_recreation_data.append((vision_vector, recreated_vision))

        for i in range(self.meaning2word.n_categories):
            d["_" + self.train_type[i] + " language recreation kl divergence"] = np.mean(kl_divergences[i])
            d[self.train_type[i] + " language recreation distance"] = np.mean(language_distances[i])
            d[self.train_type[i] + " vision recreation distance"] = np.mean(vision_distances[i])

        filenames = ["language_recreation_data.pickle", "vision_recreation_data.pickle"]
        self.logger.write_models([language_recreation_data, vision_recreation_data], filenames, episode_no)

        return d

