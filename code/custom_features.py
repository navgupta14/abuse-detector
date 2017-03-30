from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import re
import timing

'''
class SampleExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, vars):
        self.vars = vars

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        return do_something() # actual feature extraction happens here
'''
class UpperCaseLetters(BaseEstimator, TransformerMixin):
    def get_feature_names(self):
        return np.array(['n_caps'])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        n_caps = []
        for doc in documents:
            caps = sum(1 for c in doc if c.isupper())
            n_caps.append(caps)
        return np.array([n_caps]).T

class BadWordCounter(BaseEstimator, TransformerMixin):
    def __init__(self):
        with open("../data/google_badwords_list.txt") as f:
            badwords = [l.strip() for l in f.readlines()]
        self.badwords = badwords

    def get_feature_names(self):
        return np.array(['n_words', 'n_chars', 'max_len',
                         'mean_len', '!', '@', 'spaces', 'bad_ratio', 'n_bad', 'xexp'])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        n_words = [len(c.split()) for c in documents]
        n_chars = [len(c) for c in documents]
        max_word_len = [np.max([len(w) for w in c.split()]) for c in documents]
        mean_word_len = [np.mean([len(w) for w in c.split()]) for c in documents]

        # number of google badwords:
        n_bad = [np.sum([c.lower().count(w) for w in self.badwords]) for c in documents]

        # number of xexp (**** kind of abuses)
        n_xexp = [c.count("xexp") for c in documents]
        exclamation = [c.count("!") for c in documents]
        addressing = [c.count("@") for c in documents]
        spaces = [c.count(" ") for c in documents]

        bad_ratio = np.array(n_bad) / np.array(n_words, dtype=np.float)

        return np.array([n_words, n_chars, max_word_len, mean_word_len, exclamation,
                         addressing, spaces, bad_ratio, n_bad, n_xexp]).T

class Preprocessing(BaseEstimator, TransformerMixin):

    def get_feature_names(self):
        return np.array(['preprocessed'])

    def fit(self, documents, y=None):
        return self

    def transform(self, comments):
        new_comments = []
        cache = {}
        for comment in comments:
            comment = comment.lower()

            # sanitizing the data.
            comment = comment.replace("\\n", " ").replace("\\t", " ")
            comment = comment.replace("\\xa0", " ").replace("\\xc2", " ")
            # removing html tags
            tags_expr = re.compile('<.*?>')
            comment = re.sub(tags_expr, '', comment)

            # TODO - here we pruned all the urls. Perhaps, we should count the occurences of urls in a comment and use that as a feature.
            # removing urls
            url_expr = re.compile('http\S+')
            comment = re.sub(url_expr, '', comment)
            # expanding short forms
            comment = comment.replace(" u ", " you ").replace(" em ", " them ").replace(" da ", " the "). \
                replace(" yo ", " you ").replace(" ur ", " your ")
            comment = comment.replace("won't", "will not").replace("can't", "can not").replace("don't", "do not"). \
                replace("i'm", "i am").replace("im", "i am").replace("ain't", "is not").replace("ll", "will"). \
                replace("'t", " not").replace("'ve", " have").replace("'s", " is").replace("'re", " are").replace("'d", " would")

            comment = re.sub("ies( |$)", "y ", comment)
            comment = re.sub("s( |$)", " ", comment)
            comment = re.sub("ing( |$)", " ", comment)
            comment = re.sub("ed( |$)", " ", comment)

            # Generalizing custom abuses.
            comment = re.sub(" [*$%&#@][*$%&#@]+", " xexp ", comment)
            comment = re.sub(" [0-9]+ ", " DD ", comment)

            # TODO - Mapping different forms of abuse to their root forms (like f00l, fo0l to Fool). Perhaps this wont be required due to use of char n grams as features.
            new_comments.append(comment)
        return new_comments

class Preprocessing_without_stemming(BaseEstimator, TransformerMixin):

    def get_feature_names(self):
        return np.array(['preprocessed_without_stemming'])

    def fit(self, documents, y=None):
        return self

    def transform(self, comments):
        new_comments = []
        cache = {}
        for comment in comments:
            comment = comment.lower()

            # sanitizing the data.
            comment = comment.replace("\\n", " ").replace("\\t", " ")
            comment = comment.replace("\\xa0", " ").replace("\\xc2", " ")
            # removing html tags
            tags_expr = re.compile('<.*?>')
            comment = re.sub(tags_expr, '', comment)

            # TODO - here we pruned all the urls. Perhaps, we should count the occurences of urls in a comment and use that as a feature.
            # removing urls
            url_expr = re.compile('http\S+')
            comment = re.sub(url_expr, '', comment)
            # expanding short forms
            comment = comment.replace(" u ", " you ").replace(" em ", " them ").replace(" da ", " the "). \
                replace(" yo ", " you ").replace(" ur ", " your ")
            comment = comment.replace("won't", "will not").replace("can't", "can not").replace("don't", "do not"). \
                replace("i'm", "i am").replace("im", "i am").replace("ain't", "is not").replace("ll", "will"). \
                replace("'t", " not").replace("'ve", " have").replace("'s", " is").replace("'re", " are").replace("'d", " would")

            # Generalizing custom abuses.
            comment = re.sub(" [*$%&#@][*$%&#@]+", " xexp ", comment)
            comment = re.sub(" [0-9]+ ", " DD ", comment)

            # TODO - Mapping different forms of abuse to their root forms (like f00l, fo0l to Fool). Perhaps this wont be required due to use of char n grams as features.
            new_comments.append(comment)
        return new_comments
