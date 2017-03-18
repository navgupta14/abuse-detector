from sklearn.base import BaseEstimator, TransformerMixin

class SampleExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, vars):
        self.vars = vars

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        return do_something() # actual feature extraction happens here