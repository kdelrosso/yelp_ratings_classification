from __future__ import division
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


class NaiveBayesTwoStageClassifier(object):
    """A two stage classifier for text data.

    Stage 1: Multinomial Naive Bayes classifier after tfidf transformation.
        The output of stage one are predicted probabilities of each class which
        become a feature for stage two.
    Stage 2: A traditional ML classifier (logistic, rf, boost, svm, knn, etc.)
        The output of stage two is a class prediction.

    Parameters
    ----------
    stage_two_model: sklearn classification object
    """

    def __init__(self, stage_two_model):
        self.stage_two_model = stage_two_model

    def text_column(self, X, column_index):
        """Return text column from a feature matrix X. Also return X with the text column removed."""

        text_array = X[:, column_index].flatten()
        X = np.delete(X, column_index, axis=1).astype(float)
        return text_array, X

    def fit_naive_bayes_with_tfidf(self, text_array, y_train, tok=None):
        """Return fit tfidf and MultinomialNB models.

        Parameters
        ----------
        text_array: 1d array, array containing text from training data
        y_train: 1d array, training target array
        tok: tokenizer, optional
        """

        self.tfidf = TfidfVectorizer(stop_words='english', tokenizer=tok)
        X_train = self.tfidf.fit_transform(text_array)

        # parameters found using grid search
        self.mnb = MultinomialNB(alpha=0.5, fit_prior=False)
        self.mnb.fit(X_train, y_train)

    def add_naive_bayes_probs(self, X_train, probs):
        """Add predicted probabilities from naive bayes model to X_train."""

        return np.c_[probs, X_train]

    def predict_with_naive_bayes(self, text_array):
        """Return array of predictions using naive bayer model.

        Parameters
        ----------
        text_array: 1d array, array containing text
        """

        return self.mnb.predict( self.tfidf.transform(text_array) )

    def predict_with_two_stage(self, text_array, X_test):
        """Return array of predictions using the stage two model.

        Parameters
        ----------
        text_array: 1d array, array containing text
        X_test: 2d array, test feature matrix
        """

        tfidf_vectorized = self.tfidf.transform(text_array)
        probs = self.mnb.predict_proba(tfidf_vectorized)

        # add predicted probabilities from naive bayes model a feature to model 2
        X_test = self.add_naive_bayes_probs(X_test, probs)

        return self.stage_two_model.predict(X_test)

    def fit_stage_one(self, X_train, y_train, text_column_index=0, tok=None):
        """Return updated training data after fitting the stage one model."""

        text_array, X_train = self.text_column(X_train, text_column_index)
        self.fit_naive_bayes_with_tfidf(text_array, y_train)
        mnb_probs = self.mnb.predict_proba(self.tfidf.transform(text_array))

        return self.add_naive_bayes_probs(X_train, mnb_probs)

    def fit(self, X_train, y_train, text_column_index=0, tok=None):
        """Fit stage one and two models.

        Parameters
        ----------
        X_train: 2d array: training feature matrix
        y_train: 1d array: training target
        text_column_index: int, index of column in X_train containing text for tfidf
        tok: function (optional), function to use to tokenize documents
        """

        X_train = self.fit_stage_one(X_train, y_train, text_column_index, tok)
        self.stage_two_model.fit(X_train, y_train)

    def predict(self, X_test, text_column_index=0, stage_one=False):
        """Return a prediction for each row in X_test.

        Parameters
        ----------
        X_test: 2d array, data to use to make predictions
        text_column_index: int, index of column in X_test containing text for tfidf
        stage_one: bool, use True to make prediction using stage 1 only, False for stage 2
        """

        text_array, X_test = self.text_column(X_test, text_column_index)

        if stage_one:
            return self.predict_with_naive_bayes(text_array)
        else:
            return self.predict_with_two_stage(text_array, X_test)
