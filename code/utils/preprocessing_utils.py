from __future__ import division
from os import path
from collections import Counter
import re
import string
import cPickle
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

PROJECT_DIR = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
STOPWORDS = stopwords.words('english')

class ReviewPreprocessing(object):
    """For preprocessing the raw Yelp review data. Features added for
    review / word sentiment and word / punctuation counts. Rating stars
    modified to fall in [1, 3, 5].

    Parameters
    ----------
    df: DataFrame, containing reviews

    Notes
    -----
    Sentiment data from:
    https://www.quora.com/Is-there-a-downloadable-database-of-positive-and-negative-words
    http://sentiwordnet.isti.cnr.it/
    https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html
    """

    def __init__(self, df):
        self.df = df

    def pos_neg_words(self, value, filename):
        """Create and save a dictionary of positive / negative words. Positive words
        receive a value of +1 and while negative words get -1. Dictionaries are saved
        as Pickle objects to pos_dict.pkl / neg_dict.pkl.

        Parameters
        ----------
        pos_words: bool, True to load the positive words, False for negative words
        """

        words = {}
        with open(filename) as f:
            for line in f:
                if len(line) == 0: continue
                if (line == '\n') or (line[0] == ';'): continue
                words[line.replace('\n', '')] = value

        return words

    def sentiment_words(self, filename):
        """Create and save a dictionary of sentiment scores for words. Words can
        have a positive score, negative score, or both. Words are accompanied by
        a number, with #1 representing the most common meaning of the word. The
        sentiment dictionary is saved to sent_dict.pkl as a Pickle object.
        """

        df = pd.read_table(filename, skiprows=26)
        df['score'] = df['PosScore'] - df['NegScore']
        df = df[['SynsetTerms', 'score']]
        df.columns = ['words', 'score']

        # remove neutral words
        mask = df['score'] != 0
        df = df[mask]

        rx1 = re.compile('#([0-9])')
        rx2 = re.compile('#[0-9] ?')
        sent_dict = {}
        for i, row in df.iterrows():
            w = row['words']
            s = row['score']

            nums = re.findall(rx1, w)
            # use -1 since last element is an empty string
            words = re.split(rx2, w)[:-1]
            for nn, ww in zip(nums, words):
                # only sentiment for the most common meaning of the word
                if nn == '1':
                    sent_dict[ww] = s

        return sent_dict

    def create_sentiment_dicts(self):
        """Create sentiment dictionaries from two sources. Dictionaries
        contain {'word': sentiment_score}.
        """

        # combine positive and negative dicts
        self.pos_neg_dict = self.pos_neg_words(1, PROJECT_DIR + '/data/positive-words.txt')
        neg_dict = self.pos_neg_words(-1, PROJECT_DIR + '/data/negative-words.txt')
        self.pos_neg_dict.update(neg_dict)

        self.sent_dict = self.sentiment_words(PROJECT_DIR + '/data/SentiWordNet.txt')

    def update_sentiment_score(self, val, senti_dict):
        """Return a dictionary with values updated based on the sign of val.

        Parameters
        ----------
        val: float, a sentiment score
        senti_dict: dictionary, e.g. {'score': 0, 'pos_cnt': 0, 'neg_cnt': 0}
        """

        senti_dict['score'] += val
        if val > 0:
            senti_dict['pos_cnt'] += 1
        elif val < 0:
            senti_dict['neg_cnt'] += 1

        return senti_dict

    def text_features(self, row):
        """Return a DataFrame row with featured added based on words, character,
        and punctuation counts and sentiment.

        Parameters
        ----------
        row: Series, a row in a DataFrame
        """

        # review text as a string
        text = row['text'].lower()

        # count punctuation, characters, exclamation points, and question marks
        rx = re.compile( '[%s]' % re.escape(string.punctuation) )
        punct_count = Counter(re.findall(rx, text))
        row['e_point'] = punct_count['!']
        row['q_mark'] = punct_count['?']
        row['punct'] = np.sum(punct_count.values())
        row['chars'] = len(re.sub('\s+', '', text))

        # remove all punctuation
        text = re.sub("'", '', text)
        text = re.sub(rx, ' ', text).strip()

        # review word count
        words = re.split(r'\s+', text)
        row['words'] = len(words)

        # add sentiment features
        senti_1_dict = {'score': 0, 'pos_cnt': 0, 'neg_cnt': 0}
        senti_2_dict = senti_1_dict.copy()
        for w in words:
            # use 0 as sentiment default value
            senti_1_val = self.sent_dict.get(w, 0)
            senti_2_val = self.pos_neg_dict.get(w, 0)

            senti_1_dict = self.update_sentiment_score(senti_1_val, senti_1_dict)
            senti_2_dict = self.update_sentiment_score(senti_2_val, senti_2_dict)

        # sentiment rate is the sentiment score divided by the number of words
        row['senti_1_score'] = senti_1_dict['score']
        row['senti_1_rate'] = senti_1_dict['score'] / len(words)
        row['senti_1_pos_cnt'] = senti_1_dict['pos_cnt']
        row['senti_1_neg_cnt'] = senti_1_dict['neg_cnt']
        row['senti_2_score'] = senti_2_dict['score']
        row['senti_2_rate'] = senti_2_dict['score'] / len(words)
        row['senti_2_pos_cnt'] = senti_2_dict['pos_cnt']
        row['senti_2_neg_cnt'] = senti_2_dict['neg_cnt']

        return row

    def update_array(self, arr):
        """Update the values in arr, with 2 -> 1 and 4 -> 5."""

        replace_dict = {2: 1, 4: 5,}
        return np.array([ replace_dict[a] if a in replace_dict else a for a in arr ])

    def update_features(self):
        """Main method, update DataFrame features and update rating stars."""

        self.create_sentiment_dicts()
        self.df = self.df.apply(self.text_features, axis = 1)
        self.df['stars'] = self.update_array(self.df['stars'].values)

    def save_df(self, filename=PROJECT_DIR + '/data/saved_df.pkl'):
        """Save the DataFrame to disk as pickle object."""

        self.df.to_pickle(filename)

    def load_df(self, filename=PROJECT_DIR + '/data/saved_df.pkl'):
        """Return a saved DataFrame."""

        with open(filename, 'r') as f:
            return cPickle.load(f)

def text_tokenizer(text):
    """Return two array of tokenized lower case words with and without
    stopwords removed.

    Parameters
    ----------
    text: string
    """

    tokenizer = RegexpTokenizer(r'\w+')
    text = text.lower().replace("'", '')

    all_tokens = tokenizer.tokenize(text)
    without_stopwords = [word for word in all_tokens if not word in STOPWORDS]

    return np.array(all_tokens), np.array(without_stopwords)
