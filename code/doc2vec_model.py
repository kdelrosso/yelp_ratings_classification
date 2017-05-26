from __future__ import division
from os import path
import multiprocessing
import numpy as np

from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
import gensim.models.doc2vec

from utils.preprocessing_utils import text_tokenizer
from utils.model_utils import most_common_element

PROJECT_DIR = path.dirname(path.dirname(path.abspath(__file__)))

class LabeledLineSentence(object):
    """A class which permutes the order of documents / labels. Used as a iterator
    that returns a LabeledSentence.

    Parameters
    ----------
    doc_list: a list of documnet strings
    labels_list: a list of label strings

    References
    ----------
    https://rare-technologies.com/doc2vec-tutorial/
    """

    def __init__(self, doc_list, labels_list):
       self.doc_list = doc_list
       self.labels_list = labels_list

    def __iter__(self):
        """Return a LabeledSentence after permuting the order of the documents."""

        perms = np.random.permutation( zip(self.doc_list, self.labels_list) )
        for text, label in perms:
            all_words, without_stop = text_tokenizer(text)
            yield LabeledSentence(all_words, [label])

class Doc2VecClassifier(object):

    def fit(self, text_array, labels):
        """Train a Doc2Vec model. In order to use the model for classification, we need all
        documents to be included.

        Parameters
        ----------
        text_array: list of strings, all documents in the corpus
        labels: list, containing unique identifier for every document in test_array
        """

        if gensim.models.doc2vec.FAST_VERSION == -1:
            raise Exception('Need to add c compiler to increase Doc2Vec speed')

        # initialize Doc2Vec object
        cores = multiprocessing.cpu_count()
        self.model = Doc2Vec(min_count=1, window=10, size=50, sample=1e-4, negative=5, workers=cores)

        # build model vocabulary
        itt = LabeledLineSentence(text_array, labels)
        self.model.build_vocab(itt)

        # train the model
        for epoch in range(10):
            print 'epoch', epoch
            self.model.train(itt)
            self.model.alpha -= 0.002
            self.model.min_alpha = self.model.alpha

    def predict(self, train_dict, test_dict, k):
        """Return arrays for true values and predicted values.

        Parameters
        ----------
        train_dict: dictionary, training data mapping labels to stars
        test_dict: dictionary, test data mapping labels to stars
        k: number of nearest vectors to use for classification
        """

        y_pred = []
        for label in test_dict.iterkeys():
            # since the model contains both training and test data we get the k * 3
            # nearest vectors and keep the k nearest vectors in the training data
            most_sim = self.model.docvecs.most_similar(label, topn=k*3)

            votes = []
            for lab, sim in most_sim:
                # only keep vectors in the training data
                if lab in train_dict:
                    votes.append(train_dict[lab])
                    # keep k nearest vectors
                    if len(votes) == k:
                        break

            # the most common value from the k nearest vectors is the prediction
            y_pred.append(most_common_element(votes))

        return y_pred
