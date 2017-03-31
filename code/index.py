import timing
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from custom_features import BadWordCounter, Preprocessing, Preprocessing_without_stemming, UpperCaseLetters, LikelyAbusePhrase
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

train_data = pd.read_csv('../data/train_sentences.csv')
test_data = pd.read_csv('../data/test_with_solutions.csv')

train_y = np.array(train_data.Insult)
test_y = np.array(test_data.Insult)
train_comments = np.array(train_data.Comment)
#train_comments = preprocessing(train_comments)
test_comments = np.array(test_data.Comment)
#test_comments = preprocessing(test_comments)

# word n grams - count vectors
word_cv = CountVectorizer(ngram_range=(1, 3), analyzer='word')
# char n grams - count vectors
char_cv = CountVectorizer(ngram_range=(3, 5), analyzer='char_wb')
# word n grams - TfIdf
word_tfidf = TfidfVectorizer(ngram_range=(1, 3), analyzer='word', sublinear_tf=True)
# char n grams - TfIdf
char_tfidf = TfidfVectorizer(ngram_range=(3, 5), analyzer='char_wb', sublinear_tf=True)
preprocessing = Preprocessing()
preprocessing_without_stemming = Preprocessing_without_stemming()
badwords = BadWordCounter()
n_caps = UpperCaseLetters()
likely_abuse = LikelyAbusePhrase()

combined_features = FeatureUnion([
    ('likely_abuse', likely_abuse),
    ('n_caps', n_caps),
    ('word_tfidf', Pipeline([
        ('normalize', preprocessing),
        ('word', word_tfidf)
    ])),
    ('char_tfidf', Pipeline([
        ('normalize', preprocessing),
        ('char', char_tfidf)
    ])),
    ('badwords', Pipeline([
        ('normalize', preprocessing_without_stemming),
        ('badwords', badwords)
    ])),
])

#fitting a svm
svm = LinearSVC()
lr = LogisticRegression(random_state=1)
rfc = RandomForestClassifier(random_state=1)
gnb = GaussianNB()

eclf = VotingClassifier(estimators=[
    ('svm', svm), ('lr', lr), ('rfc', rfc)
], voting='hard')

pipeline = Pipeline([
    ("features", combined_features),
    ("select", SelectKBest(score_func=chi2)),
    ("classifier", eclf)
])
print eclf.get_params().keys()
pg = {'classifier__svm__C': [0.001, 0.01, 0.1, 1, 10], 'classifier__lr__C': [1.0, 100.0],\
      'classifier__rfc__n_estimators': [20, 100], 'select__k': [1000, 2000, 3000, 4000]}
#pg = {'classifier__svm__C': [0.1], 'classifier__lr__C': [1.0],\
#      'classifier__rfc__n_estimators': [20, 30], 'select__k': [1000, 2000, 3000, 4000]}
grid = GridSearchCV(pipeline, param_grid=pg, cv=2, n_jobs=2)
grid.fit(train_comments, train_y)
print grid.best_params_
print grid.best_score_
print "Linear svm - grid: ", grid.score(test_comments, test_y)

