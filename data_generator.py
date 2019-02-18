import numpy as np
from collections import Counter


class WordVector(object):
    def __init__(self):
        """
        initializes word vector, and data structures needed for data generation.
        """

        # all properties (words) get initialized here.
        self.positions = ["left", "right", "top", "bottom"]
        self.sizes = ["big", "small"]
        self.colors = ["red", "blue", "green", "purple", "black", "white"]
        self.shapes = ["cube", "sphere", "cylinder", "cone"]
        self.properties = [(self.positions, self._generate_position), (self.sizes, self._generate_sizes),
                           (self.colors, self._generate_colors),
                           (self.shapes, self._generate_shapes)]

        # generate vocabulary and starting indices in for each category in that vocabulary.
        self.vocabulary = []
        self.cat_indices = [0]
        for a, f in self.properties:
            self.vocabulary += a
            self.cat_indices.append(self.cat_indices[-1] + len(a))
        self.cat_indices = self.cat_indices[:-1]
        self.vocabulary = {a: i for i, a in enumerate(self.vocabulary)}
        self.reverse_dict = {i: a for a, i in self.vocabulary.items()}

        # dimension of input vector.
        self.dim = len(self.vocabulary)

        self.string_vector = [""]                       # contains last phrase in words.
        self.word_input = None                          # contains last language vector.
        self.vision_data = [None, None, None, None]     # contains last vision data.
        self.input_pairs_history = []                   # containts input pairs history.

    @staticmethod
    def _generate_position(position):
        """
        generates and returns x, y, z coordinates given the position.
        
        position: "right", "left", "top" or "bottom".
        """

        x = 0
        y = 0
        if position == "right":
            x = np.random.uniform(0, 1)
            y = np.random.uniform(-x, x)
        elif position == "left":
            x = np.random.uniform(-1, 0)
            y = np.random.uniform(x, -x)
        elif position == "top":
            y = np.random.uniform(0, 1)
            x = np.random.uniform(-y, y)
        elif position == "bottom":
            y = np.random.uniform(-1, 0)
            x = np.random.uniform(y, -y)
        z = (y + 1) * 0.2
        return [x, y, z]

    @staticmethod
    def _generate_sizes(size):
        """
        generates and returns size given the probability s.

        size: "big" or "small".
        """
        
        s = 0.5
        if size == "big":
            s = np.random.uniform(1-s, 1)
        elif size == "small":
            s = np.random.uniform(0, s)
        return [s]

    @staticmethod
    def _generate_colors(color):
        """
        generates and returns r,g,b values given the input color.

        color: "red", "blue", "green", "purple", "white" or "black".
        """

        r = 0
        g = 0
        b = 0
        if color == "red":
            r = np.random.uniform(0.7, 1)
        elif color == "blue":
            b = np.random.uniform(0.7, 1)
        elif color == "green":
            g = np.random.uniform(0.7, 1)
        elif color == "purple":
            r = np.random.uniform(0.425, 0.575)
            b = np.random.uniform(0.425, 0.575)
        elif color == "white":
            r = np.random.uniform(0.9, 1)
            g = np.random.uniform(0.9, 1)
            b = np.random.uniform(0.9, 1)
        elif color == "black":
            r = np.random.uniform(0, 0.1)
            g = np.random.uniform(0, 0.1)
            b = np.random.uniform(0, 0.1)
        return [r, g, b]

    @staticmethod
    def _generate_shapes(shape):
        """
        generates and returns probabilities of object corresponding to certain shape (type)

        shape: "cube", "sphere", "cylinder", "cone".
        """

        sh = [0, 0, 0, 0]
        if shape == "cube":
            sh[0] = np.random.uniform(0.9, 1)
        elif shape == "sphere":
            sh[1] = np.random.uniform(0.9, 1)
        elif shape == "cylinder":
            sh[2] = np.random.uniform(0.9, 1)
        elif shape == "cone":
            sh[3] = np.random.uniform(0.9, 1)

        arr = np.arange(len(sh))
        np.random.shuffle(arr)
        for i in arr:
            if sh[i] == 0:
                sh[i] = np.random.uniform(0, 1 - sum(sh))

        return sh

    def generate_string_from_vector(self, input_vector):
        """
        generates input vector generated from generate_new_input_pair() 
        and returns its string representation.

        input_vector: vector of length l, containing probabilities for each word.
        """

        s = []
        for i, v in enumerate(input_vector):
            if v == 1.0:
                s.append(self.reverse_dict[i])
        s = " ".join(s)
        return s

    def generate_string(self, one_word=False, full=False):
        """
        generates and returns string given the vocabulary.

        one_word: whether it must be one-word phrase.
        full: whether it must be full (length==4) word-phrase.
        """

        # determines the length of phrase.
        if full:
            v_size = len(self.properties)
        elif one_word:
            v_size = 1
        else:
            v_size = np.random.randint(1, len(self.properties)+1)

        # chooses the categories to be used given v_site.
        _properties = np.arange(len(self.properties))
        np.random.shuffle(_properties)
        _properties = _properties[:v_size]
        _properties = sorted(_properties)

        # generates string word-phrase.
        word_vector = []
        for i in _properties:
            attribute = self.properties[i]
            word_vector.append(np.random.choice(attribute[0], 1)[0])
        return word_vector

    def generate_new_input_pair(self, word_input=None, one_word=False, full=False):
        """
        generates and returns language input (string_vector) and vision input (vision_data).

        word_input: predefined input in string form. When None, it is generated by this function.
        one_word: whether generated output must represent 1-word phrase.
        full: whether generated output must represent full (4-word) phrase.
        """

        # generate new phrase if word_input is None.
        if word_input is None:
            self.word_input = self.generate_string(one_word, full)
            word_input = self.word_input

        # initialize language and vision inputs.
        string_vector = np.zeros(len(self.vocabulary))
        vision_data = [None, None, None, None]
        for word in word_input:
            # change word probability in string_vector.
            string_vector[self.vocabulary[word]] = 1

            # generate vision data.
            _vision_data = None
            for i, (p, f) in enumerate(self.properties):
                if word in set(p):
                    vision_data[i] = f(word)
                    
                    if not full and not one_word:
                        # adjustment for equal representation of words "big" and "small".
                        # also taking into account 'full' argument.
                        if (word == "small" or word == "big") and np.random.rand() >= 0.5:
                            string_vector[self.vocabulary[word]] = 0
                    break

        # generate vision data for all categories not represented in word-phrase.
        for i in range(len(vision_data)):
            if vision_data[i] is None:
                _word = np.choose(np.random.randint(0, len(self.properties[i][0])), self.properties[i][0])
                vision_data[i] = self.properties[i][1](_word)

        # save and return.
        self.string_vector = string_vector
        self.vision_data = vision_data
        self.input_pairs_history.append((string_vector, vision_data))
        return string_vector, vision_data


if __name__ == "__main__":
    """
    some testing code. mainly to generate data statistics used for report.
    """

    word_vector = WordVector()
    a, b = word_vector.generate_new_input_pair()

    lengths = Counter()
    categories = Counter()
    words = Counter()
    cat = ["positions", "sizes", "colors", "types"]
    np.random.seed(0)

    """i = 0
    while i < 1000:
        a, _ = word_vector.generate_new_input_pair()
        w = word_vector.generate_string_from_vector(a).split(" ")
        if len(w) == 1 and w[0] == "":
            continue
        lengths[len(w)] += 1
        for _w in w:
            words[_w] += 1

        for j, _c in enumerate(cat):
            for _w in w:
                if _w in set(word_vector.properties[j][0]):
                    categories[_c] += 1
                    break
        i += 1
    print(lengths)
    print(words)
    print(categories)"""

    i = 0
    values = dict()
    while i < 1000:
        l, v = word_vector.generate_new_input_pair(full=True)
        w = word_vector.generate_string_from_vector(l).split(" ")
        if len(w) == 1 and w[0] == "":
            continue

        for word, vision_input in zip(w, v):
            if word not in values:
                values[word] = []
            values[word].append(vision_input)
        i += 1

    for key in values:
        values[key] = (np.mean(values[key], axis=0).tolist(), np.std(values[key], axis=0).tolist())
        print(key, values[key])

