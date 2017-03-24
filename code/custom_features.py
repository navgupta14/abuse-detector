from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

'''
class SampleExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, vars):
        self.vars = vars

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        return do_something() # actual feature extraction happens here
'''

class BadWordCounter(BaseEstimator, TransformerMixin):
    def __init__(self):
        with open("../data/google_badwords_list.txt") as f:
            badwords = [l.strip() for l in f.readlines()]
        self.badwords = badwords

    def get_feature_names(self):
        return np.array(['n_words', 'n_chars', 'max_len',
                         'mean_len', '!', '@', 'spaces', 'bad_ratio', 'n_bad'])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        n_words = [len(c.split()) for c in documents]
        n_chars = [len(c) for c in documents]
        max_word_len = [np.max([len(w) for w in c.split()]) for c in documents]
        mean_word_len = [np.mean([len(w) for w in c.split()]) for c in documents]

        # number of google badwords:
        n_bad = [np.sum([c.lower().count(w) for w in self.badwords]) for c in documents]

        exclamation = [c.count("!") for c in documents]
        addressing = [c.count("@") for c in documents]
        spaces = [c.count(" ") for c in documents]

        bad_ratio = np.array(n_bad) / np.array(n_words, dtype=np.float)

        return np.array([n_words, n_chars, max_word_len, mean_word_len, exclamation,
                         addressing, spaces, bad_ratio, n_bad]).T
