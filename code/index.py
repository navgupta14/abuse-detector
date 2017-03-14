import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

train_data = pd.read_csv('../data/train_sentences.csv')
test_data = pd.read_csv('../data/test_with_solutions.csv')

train_y = np.array(train_data.Insult)
train_comments = np.array(train_data.Comment)

cv = CountVectorizer()
cv.fit(train_comments)
train_x = cv.transform(train_comments)

#fitting a svm
svm = LinearSVC()
svm.fit(train_x, train_y)

#Test data evaluation
test_comments = np.array(test_data.Comment)
test_x = cv.transform(test_comments)
test_y = np.array(test_data.Insult)
print svm.score(test_x, test_y)