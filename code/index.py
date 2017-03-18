import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from preprocess import preprocessing

train_data = pd.read_csv('../data/train_sentences.csv')
test_data = pd.read_csv('../data/test_with_solutions.csv')

train_y = np.array(train_data.Insult)
test_y = np.array(test_data.Insult)
train_comments = np.array(train_data.Comment)
train_comments = preprocessing(train_comments)
test_comments = np.array(test_data.Comment)
test_comments = preprocessing(test_comments)

# word n grams - count vectors
word_cv = CountVectorizer(ngram_range=(1, 3), analyzer='word')
# char n grams - count vectors
char_cv = CountVectorizer(ngram_range=(3, 5), analyzer='char_wb')
# word n grams - TfIdf
word_tfidf = TfidfVectorizer(ngram_range=(1, 3), analyzer='word', sublinear_tf=True)
# char n grams - TfIdf
char_tfidf = TfidfVectorizer(ngram_range=(3, 5), analyzer='char_wb', sublinear_tf=True)
combined_features = FeatureUnion([
    #('word_cv', word_cv),
    #('char_cv', char_cv),
    ('word_tfidf', word_tfidf),
    ('char_tfidf', char_tfidf)
])

#fitting a svm
svm = LinearSVC()

pipeline = Pipeline([
    ("features", combined_features),
    ("classifier", svm)
])
# TODO - Hyperparameter tuning and Feature Selection
pipeline.fit(train_comments, y=train_y)
print len(pipeline.named_steps["features"].get_feature_names())
print "Linear svm : ", pipeline.score(test_comments, test_y)

