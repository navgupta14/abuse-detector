import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from custom_features import BadWordCounter, Preprocessing, Preprocessing_without_stemming, UpperCaseLetters,\
    LikelyAbusePhrase, DayAndTime, CommentLength, AverageWordLength, Punctuations, Misspelling
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve, precision_recall_fscore_support
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
import logging
import time
import scipy.sparse as sp


fileName = "classifier.joblib.pkl"

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
start_time = time.time()
logging.info(" ----------- Start Detector ------------")
train_data = pd.read_csv('../data/train_sentences.csv')
test_data = pd.read_csv('../data/test_with_solutions.csv')

f = open("results.log", "a")
f.write("\n\n ---------------------------- BEGIN EXECUTION ---------------------------- ")

train_y = np.array(train_data.Insult)
train_time = np.array(train_data.Date)
test_y = np.array(test_data.Insult)
train_comments = np.array(train_data.Comment)
#train_comments = preprocessing(train_comments)
test_comments = np.array(test_data.Comment)
test_time = np.array(test_data.Date)
#test_comments = preprocessing(test_comments)

# word n grams - count vectors
#word_cv = CountVectorizer(ngram_range=(1, 3), analyzer='word')
# char n grams - count vectors
#char_cv = CountVectorizer(ngram_range=(3, 5), analyzer='char_wb')
# word n grams - TfIdf
word_tfidf_1 = TfidfVectorizer(ngram_range=(1, 1), analyzer='word', sublinear_tf=True, max_df=0.5, stop_words='english')
word_tfidf_2 = TfidfVectorizer(ngram_range=(2, 2), analyzer='word', sublinear_tf=True, max_df=0.5, stop_words='english')
word_tfidf_3 = TfidfVectorizer(ngram_range=(3, 3), analyzer='word', sublinear_tf=True, max_df=0.5, stop_words='english')
# char n grams - TfIdf
char_tfidf_1 = TfidfVectorizer(ngram_range=(3, 3), analyzer='char_wb', sublinear_tf=True, stop_words='english')
char_tfidf_2 = TfidfVectorizer(ngram_range=(4, 4), analyzer='char_wb', sublinear_tf=True, stop_words='english')
char_tfidf_3 = TfidfVectorizer(ngram_range=(5, 5), analyzer='char_wb', sublinear_tf=True, stop_words='english')
preprocessing = Preprocessing()
preprocessing_without_stemming = Preprocessing_without_stemming()
badwords = BadWordCounter()
n_caps = UpperCaseLetters()
likely_abuse = LikelyAbusePhrase()
day_and_time = DayAndTime()
comment_length = CommentLength()
avg_word_length = AverageWordLength()
punctuations = Punctuations()
misspelling = Misspelling()

'''
combined_features = FeatureUnion([
    #('punctuations', punctuations),
    #('average_word_length', avg_word_length),
    #('comment_length', comment_length),
    ('misspelling', misspelling),
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
'''

features_list = [(word_tfidf_1, 2000), (word_tfidf_2, 2000), (word_tfidf_3, 2000), (char_tfidf_2, 2000),\
                 (char_tfidf_3, 2000), (char_tfidf_1, 2000), (n_caps, 2), (day_and_time, 4), (misspelling, 1)]


preprocessed_train_comments = preprocessing.fit_transform((train_comments, train_time))
preprocessed_test_comments = preprocessing.transform((test_comments, test_time))

training_data_features_list = []
test_data_features_list = []
for feature in features_list:
    feat = feature[0]
    f.write("\n" + str(feat))
    k_new = feature[1]
    if k_new > 1000:
        feat.fit(preprocessed_train_comments)
        train_x = feat.transform(preprocessed_train_comments)
        test_x = feat.transform(preprocessed_test_comments)
        select_k = SelectKBest(score_func=chi2, k=k_new)
        train_xx = select_k.fit_transform(train_x, train_y)
        test_xx = select_k.transform(test_x)
    else:
        train_xx = feat.fit_transform((train_comments, train_time))
        test_xx = feat.transform((test_comments, test_time))
    training_data_features_list.append(train_xx)
    test_data_features_list.append(test_xx)


final_training_data = sp.hstack(training_data_features_list)
final_test_data = sp.hstack(test_data_features_list)


#fitting a svm
svm = svm.SVC(C=0.3,kernel='linear',probability=True, verbose=True)
lr = LogisticRegression(random_state=1, verbose=True)
rfc = RandomForestClassifier(random_state=1)
gnb = GaussianNB()
sgd = SGDClassifier(n_iter=15000)

eclf = VotingClassifier(estimators=[
    ('svm', svm), ('lr', lr)
], voting='soft', weights=[0.6, 0.4])



df = pd.DataFrame(columns=('w1', 'w2', 'mean', 'std'))

i = 0
for w1 in []:
    for w2 in []:

            if len(set((w1,w2))) == 1: # skip if all weights are equal
                continue

            eclf = VotingClassifier(estimators=[
                ('svm', svm), ('lr', lr)
            ], voting='soft', weights=[w1, w2])
            scores = cross_val_score(
                                            estimator=eclf,
                                            X=final_training_data,
                                            y=train_y,
                                            cv=3,
                                            scoring='accuracy',
                                            n_jobs=-1)

            df.loc[i] = [w1, w2, scores.mean(), scores.std()]
            i += 1

print df.sort(columns=['mean', 'std'], ascending=False)





'''
pipeline = Pipeline([
    ("features", combined_features),
    ("select", SelectKBest(score_func=chi2, k=20000)),
    ("classifier", eclf)
])
'''

pg = {'classifier__svm__C': [0.1, 0.3], 'classifier__lr__C': [1.0, 3.0],\
       'select__k': [10000, 12000, 14000, 16000]}
#pg = {'classifier__svm__C': [0.1], 'classifier__lr__C': [1.0],\
#      'classifier__rfc__n_estimators': [20, 30], 'classifier__sgd__alpha': [0.001, 0.002],\
#      'select__k': [1000, 2000, 3000, 4000]}
#grid = GridSearchCV(pipeline, param_grid=pg, cv=5, n_jobs=4)
grid = eclf

#grid.fit((train_comments, train_time), train_y)
grid.fit(final_training_data, train_y)
#print grid.best_params_
#print grid.best_score_
score = grid.score(final_test_data, test_y)
joblib.dump(grid, fileName)

strss = "Score = %s" % score
f.write("\n" + strss)
print "Linear svm - grid: ", score
predictions = grid.predict_proba(final_test_data)
predict_ans = grid.predict(final_test_data)
grid.classes_
print grid.classes_
f = open('results_analysis.txt', 'w')
predictions = predictions[:, 1]
results = zip(predict_ans, predictions)
for item in results:
    f.write('\n' + str(item))
false_positive_rate, true_positive_rate, thresholds = roc_curve(test_y, predictions)
auc_roc = auc(false_positive_rate, true_positive_rate)
strss = "AUC_ROC = %s" % auc_roc
f.write("\n" + strss)
print "Test data auc(roc curve) : ", auc_roc
print "Test data roc auc : ", roc_auc_score(test_y, predictions)
precision, recall, thresholds = precision_recall_curve(test_y, predictions)
print "Test data auc(PR curve) : ", auc(recall, precision)
print "(PRF)macro : ", precision_recall_fscore_support(test_y, predict_ans, average='macro')
print "(PRF)micro : ", precision_recall_fscore_support(test_y, predict_ans, average='micro')
print "(PRF) : ", precision_recall_fscore_support(test_y, predict_ans, labels=[0, 1])
f.write("\n\n ---------------------------- END EXECUTION ---------------------------- \n\n")
f.close()
logging.info(" ----------- End Detector ------------")
total_time = time.time() - start_time
m, s = divmod(total_time, 60)
h, m = divmod(m, 60)
print "Program run time: %d:%02d:%02d" % (h, m, s)


