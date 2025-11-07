try:
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.utils import seeding
except Exception:
    import gym
    from gym import spaces
    from gym.utils import seeding

import string
import numpy as np
import random
import collections
from sklearn.feature_extraction.text import CountVectorizer
import yaml
import logging

config = None

MAX_WORDLEN = 25

with open("config.yaml", 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

logger = logging.getLogger('root')


class HangmanEnv(gym.Env):

    def __init__(self):
        # super().__init__()
        self.vocab_size = 26
        self.mistakes_done = 0
        f = open("./words.txt", 'r').readlines()
        self.wordlist = [w.strip() for w in f if w.strip()]
        self.action_space = spaces.Discrete(26)

        self.vectorizer = CountVectorizer(tokenizer=lambda x: list(x))
        self.vectorizer.fit([string.ascii_lowercase])

        self.config = config

        self.char_to_id = {chr(97 + x): x for x in range(self.vocab_size)}
        self.char_to_id['_'] = self.vocab_size
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}

        # OBSERVATION SPACE:
        #  - first element: the current obscured string represented as a
        #    vector of length 27 (padded). Each entry can take values 0..26 (27 categories)
        #  - second element: binary vector of length 26 indicating actions used
        #
        # We use MultiDiscrete for the first (nvec entries) and MultiBinary for the second.
        # Note: if your code expects different ranges, change `nvec_first` accordingly.
        nvec_first = np.full(27, self.vocab_size + 1, dtype=np.int64)  # 27 slots, each with 27 categories (0..26)
        self.observation_space = spaces.Tuple((
            spaces.MultiDiscrete(nvec_first),
            spaces.MultiBinary(26)
        ))

        # DO NOT assign to observation_space.shape (it's read-only)
        self.seed()

    def filter_and_encode(self, word, vocab_size, min_len, char_to_id):
        """
        checks if word length is greater than threshold and returns one-hot encoded array
        :param word: word string
        :param vocab_size: size of vocabulary (26 in this case)
        :param min_len: word with length less than this is not added to the dataset
        :param char_to_id: mapping from char to id
        """
        word = word.strip().lower()
        if len(word) < min_len:
            return None

        # one-hot encode characters, with an extra column for '_' (unknown)
        encoding = np.zeros((len(word), vocab_size + 1), dtype=np.int8)
        for i, c in enumerate(word):
            idx = char_to_id.get(c, vocab_size)  # fallback to '_' idx if char unexpected
            encoding[i][idx] = 1

        # pad to MAX_WORDLEN rows
        if encoding.shape[0] < MAX_WORDLEN:
            zero_vec = np.zeros((MAX_WORDLEN - encoding.shape[0], vocab_size + 1), dtype=np.int8)
            encoding = np.concatenate((encoding, zero_vec), axis=0)

        return encoding

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def choose_word(self):
        return random.choice(self.wordlist)

    def count_words(self, word):
        lens = [len(w) for w in self.wordlist]
        counter = dict(collections.Counter(lens))
        return counter.get(len(word), 0)

    def reset(self):
        self.mistakes_done = 0
        self.word = self.choose_word()
        self.wordlen = len(self.word)
        self.gameover = False
        self.win = False
        self.guess_string = "_" * self.wordlen
        self.actions_used = set()
        self.actions_correct = set()
        logger.info("Reset: Resetting for new word")
        logger.info("Reset: Selected word= {0}".format(self.word))

        # initial state: encoded guess string + zero actions-used vector
        self.state = (
            self.filter_and_encode(self.guess_string, 26, 0, self.char_to_id),
            np.array([0] * 26, dtype=np.int8)
        )

        logger.debug("Reset: Init State = {self.state}")

        return self.state

    def vec2letter(self, action_index):
        letters = string.ascii_lowercase
        return letters[action_index]

    def getGuessedWord(self, secretWord, lettersGuessed):
        secretString = ''
        for letter in secretWord:
            if letter not in lettersGuessed:
                secretString += '_'
            else:
                secretString += letter
        return secretString

    def check_guess(self, letter):
        if letter in self.word:
            self.prev_string = self.guess_string
            self.actions_correct.add(letter)
            self.guess_string = self.getGuessedWord(self.word, self.actions_correct)
            return True
        else:
            return False

    def step(self, action):
        """
        action is expected to be either:
        - an int (0..25) OR
        - a one-hot / array-like (e.g. numpy array) where argmax() gives the index
        """
        # normalize action to index
        if isinstance(action, (list, tuple, np.ndarray)):
            action_idx = int(np.argmax(action))
        else:
            action_idx = int(action)

        done = False
        reward = 0.0
        letter = string.ascii_lowercase[action_idx]

        if letter in self.actions_used:
            reward = -4.0
            self.mistakes_done += 1
            logger.info(f"Env Step: repeated action, action was= {letter}")
            logger.info(f"ENV STEP: Mistakes done = {self.mistakes_done}")
            if self.mistakes_done >= 6:
                done = True
                self.win = False
                self.gameover = True
        elif letter in self.actions_correct:
            reward = -3.0
            logger.info(f"Env Step: repeated correct action, action was= {letter}")
            logger.info(f"ENV STEP: Mistakes done = {self.mistakes_done}")
        elif self.check_guess(letter):
            logger.info(f"ENV STEP: Correct guess, guess was = {letter}")
            self.actions_correct.add(letter)
            if set(self.word) == self.actions_correct:
                reward = 10.0
                done = True
                self.win = True
                self.gameover = True
                logger.info(f"ENV STEP: Won Game, guess was = {letter}")
            else:
                reward = 1.0
        else:
            logger.info(f"ENV STEP: Incorrect guess, guess was = {letter}")
            self.mistakes_done += 1
            if self.mistakes_done >= 6:
                reward = -5.0
                done = True
                self.gameover = True
            else:
                reward = -2.0

        self.actions_used.add(letter)
        logger.info("ENV STEP: actions used = {0}".format(" ".join(sorted(self.actions_used))))

        # build next state
        # filter_and_encode returns a padded one-hot matrix for guess_string
        actions_list = sorted(self.actions_used)
        if len(actions_list) == 0:
            actions_vec = np.array([0] * 26, dtype=np.int8)
        else:
            # vectorizer.transform expects a sequence of tokens; convert to a single string token list
            try:
                actions_vec = self.vectorizer.transform(["".join(actions_list)]).toarray()[0].astype(np.int8)
            except Exception:
                # fallback: manual build
                actions_vec = np.array([1 if chr(97 + i) in self.actions_used else 0 for i in range(26)], dtype=np.int8)

        self.state = (
            self.filter_and_encode(self.guess_string, 26, 0, self.char_to_id),
            actions_vec
        )

        logger.debug("Intermediate State = {self.state}")
        return (self.state, reward, done, {'win': self.win, 'gameover': self.gameover})
