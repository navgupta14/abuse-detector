from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import string
import enchant
import re

class DayAndTime(BaseEstimator, TransformerMixin):

    def get_feature_names(self):
        return np.array(['weekday'], ['weeknight'], ['weekend_day'], ['weekend_night'])

    def fit(self, documents, y=None):
        return self

    def transform(self, docs):
        documents = docs[1]
        def weekDay_func(year, month, day):
            offset = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
            week = ['Sunday',
                    'Monday',
                    'Tuesday',
                    'Wednesday',
                    'Thursday',
                    'Friday',
                    'Saturday']
            afterFeb = 1
            if month > 2: afterFeb = 0
            aux = year - 1700 - afterFeb
            # dayOfWeek for 1700/1/1 = 5, Friday
            dayOfWeek = 5
            # partial sum of days betweem current date and 1700/1/1
            dayOfWeek += (aux + afterFeb) * 365
            # leap year correction
            dayOfWeek += aux / 4 - aux / 100 + (aux + 100) / 400
            # sum monthly and day offsets
            dayOfWeek += offset[month - 1] + (day - 1)
            dayOfWeek %= 7
            return dayOfWeek + 1, week[dayOfWeek]

        month_of_year = []
        day_of_week = []
        hour_of_day = []
        weekday = []
        weeknight = []
        weekend_day = []
        weekend_night = []
        for date_time in documents:
            if pd.isnull(date_time):
                month_of_year.append(0)
                day_of_week.append(0)
                hour_of_day.append(0)
                weekday.append(0)
                weeknight.append(0)
                weekend_day.append(0)
                weekend_night.append(0)
            else:
                year = int(date_time[:4])
                month = int(date_time[4:6])
                given_date = int(date_time[6:8])
                hour = int(date_time[8:10])
                (day_int, day_str) = weekDay_func(year, month, given_date)
                month_of_year.append(month)
                day_of_week.append(day_int)
                hour_of_day.append(hour)
                if day_int == 1 or day_int == 7:
                    weeknight.append(0)
                    weekday.append(0)
                    if hour > 2 and hour < 18:
                        weekend_day.append(1)
                        weekend_night.append(0)
                    else:
                        weekend_night.append(1)
                        weekend_day.append(0)
                else:
                    weekend_night.append(0)
                    weekend_day.append(0)
                    if hour > 2 and hour < 18:
                        weekday.append(1)
                        weeknight.append(0)
                    else:
                        weeknight.append(1)
                        weekday.append(0)
        return np.array([weekday, weeknight, weekend_day, weekend_night]).T


class CommentLength(BaseEstimator, TransformerMixin):
    def get_feature_names(self):
        return np.array(['n_comment_len'])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        documents = documents[0]
        n_len = [len(c) for c in documents]
        return np.array([n_len]).T


class AverageWordLength(BaseEstimator, TransformerMixin):
    def get_feature_names(self):
        return np.array(['average_word_len'])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        documents = documents[0]
        avg_word_len = []
        for doc in documents:
            dwords = doc.split()
            wordlen = 0.0
            for word in dwords:
                wordlen += len(word)
            avg_word_len.append(wordlen/len(dwords))
        return np.array([avg_word_len]).T


class Punctuations(BaseEstimator, TransformerMixin):
    def get_feature_names(self):
        return np.array(['punctuations'])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        documents = documents[0]
        count = lambda l1, l2: sum([1 for x in l1 if x in l2])
        punctuations = [count(c, set(string.punctuation)) for c in documents]
        return np.array([punctuations]).T


class Misspelling(BaseEstimator, TransformerMixin):
    def get_feature_names(self):
        return np.array(['misspelling'])

    def fit(self, documents, y=None):
        return self

    def preprocess(self, documents):
        new_comments = []
        for comment in documents:
            comment = comment.lower()
            while True:
                if comment.startswith('"'):
                    comment = comment[1:]
                else:
                    break
            while True:
                if comment.endswith('"'):
                    comment = comment[:-1]
                else:
                    break

            # sanitizing the data.
            comment = comment.replace("\\n", " ").replace("\\t", " ")
            comment = comment.replace("\\xa0", " ").replace("\\xc2", " ")
            # removing html tags
            tags_expr = re.compile('<.*?>')
            comment = re.sub(tags_expr, '', comment)

            # removing urls
            url_expr = re.compile('http\S+')
            comment = re.sub(url_expr, '', comment)

            comment = comment.replace("\\", "")
            comment = comment.replace("\/", "")

            new_comments.append(comment)
        return new_comments

    def transform(self, documents):
        documents = documents[0]
        documents = self.preprocess(documents)
        #Below is the nltk word list approach. May be slightly faster than PyEnchant
        #Currently we'll be sticking to PyEnchant
        # words = set(dictwords.words())
        #count = lambda l1: sum([1 for x in l1.split() if x in words])

        #PyEnchant Approach:
        engdict = enchant.Dict("en_US")
        count = lambda l1: sum([1 for x in l1 if engdict.check(x) == False])

        misspellings = []
        for c in documents:
            temp_list = re.split(r'\s+|[,;.-]\s*', c)
            temp_list = filter(None, temp_list)
            misspellings.append(count(temp_list))
        return np.array([misspellings]).T


class UpperCasedWords(BaseEstimator, TransformerMixin):
    def get_feature_names(self):
        return np.array(['n_caps'], ['n_caps_ratio'])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        documents = documents[0]
        n_words = [len(c.split()) for c in documents]
        n_caps = [np.sum([w.isupper() for w in comment.split()]) for comment in documents]
        n_caps_ratio = np.array(n_caps) / np.array(n_words, dtype=np.float)
        return np.array([n_caps, n_caps_ratio]).T


#This feature handles comments containing "you are a", "you're a", "you sound like"
class LikelyAbusePhrase(BaseEstimator, TransformerMixin):
    def get_feature_names(self):
        return np.array(['likely_abuse'])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        documents = documents[0]
        likely_abuse = []
        for doc in documents:
            doc = doc.lower()
            labuse = doc.count("u're a ")
            labuse += doc.count("u r a ")
            labuse += doc.count("u are a ")
            labuse += doc.count("ur a ")
            labuse += doc.count("u'r a ")
            labuse += doc.count("u sound")
            labuse += doc.count("u're such a")
            labuse += doc.count("u r such a")
            labuse += doc.count("u are such a")
            labuse += doc.count("ur such a")
            labuse += doc.count("u'r such a")
            likely_abuse.append(labuse)
        return np.array([likely_abuse]).T


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
        n_bad = []
        for c in documents:
            bad_words = 0
            temp_list = re.split('\W+', c.lower())
            for w in self.badwords:
                for word in temp_list:
                    if w == word:
                        bad_words += 1
            n_bad.append(bad_words)

        # number of xexp (**** kind of abuses)
        n_xexp = [c.count("xexp") for c in documents]
        exclamation = [c.count("!") for c in documents]
        addressing = [c.count("@") for c in documents]
        spaces = [c.count(" ") for c in documents]

        bad_ratio = np.array(n_bad) / np.array(n_words, dtype=np.float)

        return np.array([exclamation,
                         addressing, n_bad, n_xexp]).T


class Preprocessing(BaseEstimator, TransformerMixin):

    def get_feature_names(self):
        return np.array(['preprocessed'])

    def fit(self, documents, y=None):
        return self

    def transform(self, comm):
        comments = comm[0]
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

#            comment = comment.replace(" u ", " you ").replace(" em ", " them ").replace(" da ", " the "). \
#                replace(" yo ", " you ").replace(" ur ", " your ")
#            comment = comment.replace("won't", "will not").replace("can't", "can not").replace("don't", "do not"). \
#                replace("i'm", "i am").replace("im", "i am").replace("ain't", "is not").replace("ll", "will"). \
#                replace("'t", " not").replace("'ve", " have").replace("'s", " is").replace("'re", " are").replace("'d", " would")
#
#            comment = re.sub("ies( |$)", "y ", comment)
#            comment = re.sub("s( |$)", " ", comment)
#            comment = re.sub("ing( |$)", " ", comment)
#            comment = re.sub("ed( |$)", " ", comment)

            # Generalizing custom abuses.
            comment = re.sub("[*$%&#@][*$%&#@]+", "xexp", comment)
            comment = re.sub(" [0-9]+ ", " DD ", comment)

            # TODO - Mapping different forms of abuse to their root forms (like f00l, fo0l to Fool). Perhaps this wont be required due to use of char n grams as features.
            new_comments.append(comment)
        return new_comments
