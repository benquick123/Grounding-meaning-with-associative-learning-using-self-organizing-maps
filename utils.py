import os
import shutil
import csv
import pickle
import numpy as np


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
        folder_name = "Episodes/Episode " + str(episode_n)
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
    """
    Helper used for calculation of training metrics.
    """
    def __init__(self, logger, train_type, word_threshold=0.9):
        """
        pass logger, train_type and word thresholy.
        train_type: list all category names.
        """

        self.word_threshold = word_threshold
        self.logger = logger
        self.train_type = train_type

    @staticmethod
    def entropy(p, log_n=True):
        r = -np.sum([p * np.log(p)])
        if log_n:
            r /= np.log(len(p))
        return r

    @staticmethod
    def kl_divergence(p, q):
        kl = np.sum(np.where(p == 0.0, 0.0, p * np.log(p / q))) / np.log(p.shape[0])
        return kl

    @staticmethod
    def euclidean_distance(p, q):
        return np.sqrt(np.sum((p - q) ** 2))

    def hebb_entropies(self, hebb_weights, cat_name, cat_indices):
        """
        calculates w-->m hebbian links entropies for given cat_name for each word type.
        also calculates m-->w hebbian links for given cat_name.
        function must be called after training of each category in iteration.
        returns log dictionary that is later fed to Logger.

        hebb_weights: slice of hebbian weights in both directions for current category.
        cat_name: name of current category.
        cat_indices: starting indices of words in language_input for each category.
        """

        d = dict()
        h_wm = hebb_weights[0]
        h_mw = hebb_weights[1]

        for i in range(len(cat_indices)):
            try:
                subsection_wm = h_wm[cat_indices[i]:cat_indices[i + 1], :]
            except IndexError:
                subsection_wm = h_wm[cat_indices[i]:, :]

            _entropy_wm = np.mean([self.entropy(slice) for slice in subsection_wm])

            key_wm = cat_name + " wm entropies - " + self.train_type[i]
            if cat_name != self.train_type[i]:
                key_wm = "_" + key_wm
            d[key_wm] = _entropy_wm

        _entropy_mw = np.mean([self.entropy(_h) for _h in h_mw])
        key_mw = cat_name + " mw entropies"
        d[key_mw] = _entropy_mw
        return d

    def wxc_entropies(self, wxc):
        """
        calculates WXC table entropy for each row (category).
        returns log dictionary that is later fed to Logger.

        wxc: WXC table.
        """

        d = dict()
        for row, cat in zip(wxc, self.train_type):
            key = cat + " word_x_category entropy"
            d[key] = self.entropy(row)
        return d

    def lxp_entropies(self, lxp):
        """
        calculates LXP tables entropies for each length's entry.
        returns log dictionary that is later fed to Logger.

        lxp: LXP table.
        """

        d = dict()
        for i in range(len(lxp)):
            key = "length: " + str(i + 1) + " length_x_position entropy"
            d[key] = np.mean([self.entropy(row) for row in lxp[i]])
        return d
