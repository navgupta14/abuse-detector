import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from custom_features import BadWordCounter, Preprocessing, Preprocessing_without_stemming, UpperCaseLetters,\
    LikelyAbusePhrase, DayAndTime, CommentLength, AverageWordLength, Punctuations
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve, precision_recall_fscore_support
from sklearn import svm
import logging
import time

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
start_time = time.time()
logging.info(" ----------- Start Detector ------------")
train_data = pd.read_csv('../data/train_sentences.csv')
test_data = pd.read_csv('../data/test_with_solutions.csv')
train_y = np.array(train_data.Insult)
train_time = np.array(train_data.Date)
test_y = np.array(test_data.Insult)
train_comments = np.array(train_data.Comment)

custom_seperator = "-csep-"
for i in xrange(len(train_comments)):
    time_append = train_time[i]
    if pd.isnull(time_append):
        time_append = '0'
    train_comments[i] = train_comments[i] + custom_seperator + time_append

#train_comments = preprocessing(train_comments)
test_comments = np.array(test_data.Comment)
test_time = np.array(test_data.Date)
#test_comments = preprocessing(test_comments)

# word n grams - count vectors
word_cv = CountVectorizer(ngram_range=(1, 3), analyzer='word')
# char n grams - count vectors
char_cv = CountVectorizer(ngram_range=(3, 5), analyzer='char_wb')
# word n grams - TfIdf
word_tfidf = TfidfVectorizer(ngram_range=(1, 3), analyzer='word', sublinear_tf=True, max_df=0.5, stop_words='english')
# char n grams - TfIdf
char_tfidf = TfidfVectorizer(ngram_range=(3, 5), analyzer='char_wb', sublinear_tf=True)
preprocessing = Preprocessing()
preprocessing_without_stemming = Preprocessing_without_stemming()
badwords = BadWordCounter()
n_caps = UpperCaseLetters()
likely_abuse = LikelyAbusePhrase()
day_and_time = DayAndTime()
comment_length = CommentLength()
avg_word_length = AverageWordLength()
punctuations = Punctuations()

combined_features = FeatureUnion([
    #('punctuations', punctuations),
    #('average_word_length', avg_word_length),
    #('comment_length', comment_length),
    ('time', day_and_time),
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
svm = svm.SVC(kernel='linear', probability=True)
lr = LogisticRegression(random_state=1)
rfc = RandomForestClassifier(random_state=1)
gnb = GaussianNB()
sgd = SGDClassifier(n_iter=15000)

eclf = VotingClassifier(estimators=[
    ('svm', svm), ('lr', lr)
], voting='soft', weights=[0.6, 0.4])

pipeline = Pipeline([
    ("features", combined_features),
    ("select", SelectKBest(score_func=chi2, k=20000)),
    ("classifier", eclf)
])

pg = {'classifier__svm__C': [0.1, 0.3, 0.4], 'classifier__lr__C': [1.0, 3.0, 4.0],\
       'select__k': [12000, 14000, 16000, 18000]}
#pg = {'classifier__svm__C': [0.1], 'classifier__lr__C': [1.0],\
#      'classifier__rfc__n_estimators': [20, 30], 'classifier__sgd__alpha': [0.001, 0.002],\
#      'select__k': [1000, 2000, 3000, 4000]}
grid = GridSearchCV(pipeline, param_grid=pg, cv=5, n_jobs=4, verbose=5)
#grid = pipeline
#grid = RandomizedSearchCV(pipeline, param_distributions=pg, n_iter=10)

grid.fit(train_comments, train_y)
print grid.best_params_
print grid.best_score_
print "Linear svm - grid: ", grid.score((test_comments, test_time), test_y)
predictions = grid.predict_proba((test_comments, test_time))
predict_ans = grid.predict((test_comments, test_time))
grid.classes_
print grid.classes_
predictions = predictions[:, 1]
false_positive_rate, true_positive_rate, thresholds = roc_curve(test_y, predictions)
print "Test data auc(roc curve) : ", auc(false_positive_rate, true_positive_rate)
print "Test data roc auc : ", roc_auc_score(test_y, predictions)
precision, recall, thresholds = precision_recall_curve(test_y, predictions)
print "Test data auc(PR curve) : ", auc(recall, precision)
print "(PRF)macro : ", precision_recall_fscore_support(test_y, predict_ans, average='macro')
print "(PRF)micro : ", precision_recall_fscore_support(test_y, predict_ans, average='micro')
print "(PRF)weighted : ", precision_recall_fscore_support(test_y, predict_ans, average='weighted')
print "(PRF) : ", precision_recall_fscore_support(test_y, predict_ans)

logging.info(" ----------- End Detector ------------")
total_time = time.time() - start_time
m, s = divmod(total_time, 60)
h, m = divmod(m, 60)
print "Program run time: %d:%02d:%02d" % (h, m, s)


