import numpy as np


class WordVector(object):
    def __init__(self):
        self.positions = ["left", "right", "top", "bottom"]
        self.sizes = ["big", "small"]
        self.colors = ["red", "blue", "green", "purple", "black", "white"]
        self.shapes = ["cube", "sphere", "cylinder", "cone"]
        self.properties = [(self.positions, self._generate_position), (self.sizes, self._generate_sizes),
                           (self.colors, self._generate_colors),
                           (self.shapes, self._generate_shapes)]

        # self.properties = [(self.sizes, self._generate_sizes)]

        self.vocabulary = []
        for a, f in self.properties:
            self.vocabulary += a
        self.vocabulary = {a: i for i, a in enumerate(self.vocabulary)}
        self.reverse_dict = {i: a for a, i in self.vocabulary.items()}

        self.dim = len(self.vocabulary)

        self.string_vector = [""]
        self.word_input = None
        self.vision_data = [None, None, None, None]

    @staticmethod
    def _generate_position(position):
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
        s = 0.5
        if size == "big":
            s = np.random.uniform(1-s, 1)
        elif size == "small":
            s = np.random.uniform(0, s)
        return [s]

    @staticmethod
    def _generate_colors(color):
        r = 0
        g = 0
        b = 0
        if color == "red":
            r = np.random.uniform(0, 1)
        elif color == "blue":
            b = np.random.uniform(0, 1)
        elif color == "green":
            g = np.random.uniform(0, 1)
        elif color == "purple":
            r = np.random.uniform(0.25, 0.75)
            b = np.random.uniform(0.25, 0.75)
        elif color == "white":
            r = np.random.uniform(0.75, 1)
            g = np.random.uniform(0.75, 1)
            b = np.random.uniform(0.75, 1)
        elif color == "black":
            r = np.random.uniform(0, 0.25)
            g = np.random.uniform(0, 0.25)
            b = np.random.uniform(0, 0.25)
        return [r, g, b]

    @staticmethod
    def _generate_shapes(shape):
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
        s = []
        for i, v in enumerate(input_vector):
            if v == 1.0:
                s.append(self.reverse_dict[i])
        s = " ".join(s)
        return s

    def generate_string(self, one_word=False, full=False):
        if full:
            v_size = len(self.properties)
        else:
            if one_word:
                v_size = 1
            else:
                v_size = np.random.randint(1, len(self.properties)+1)

        _properties = np.arange(len(self.properties))
        np.random.shuffle(_properties)
        _properties = _properties[:v_size]
        _properties = sorted(_properties)

        word_vector = []
        for i in _properties:
            attribute = self.properties[i]
            word_vector.append(np.random.choice(attribute[0], 1)[0])
        return word_vector

    def generate_new_input_pair(self, word_input=None, one_word=False, full=False):
        if word_input is None:
            self.word_input = self.generate_string(one_word, full)
            word_input = self.word_input

        string_vector = np.zeros(len(self.vocabulary))
        vision_data = [None, None, None, None]
        for word in word_input:
            string_vector[self.vocabulary[word]] = 1

            _vision_data = None
            for i, (p, f) in enumerate(self.properties):
                if word in set(p):
                    vision_data[i] = f(word)
                    # adjustment for equal representation of words "big" and "small".
                    # also taking into account 'full' argument.
                    if (word == "small" or word == "big") and 0.25 < vision_data[i][0] <= 0.75 and full is False and one_word is False:
                        string_vector[self.vocabulary[word]] = 0
                    break

        for i in range(len(vision_data)):
            if vision_data[i] is None:
                _word = np.choose(np.random.randint(0, len(self.properties[i][0])), self.properties[i][0])
                vision_data[i] = self.properties[i][1](_word)

        self.string_vector = string_vector
        self.vision_data = vision_data
        return string_vector, vision_data


if __name__ == "__main__":
    word_vector = WordVector()
    a, b = word_vector.generate_new_input_pair()
    word_vector.generate_string_from_vector(a)
    exit()
