from __future__ import division
from collections import Counter
import random
import numpy as np
import pandas as pd
import scipy
import warnings
warnings.filterwarnings("ignore")

from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

RANDOM_SEED_SPLIT_1 = 8859
RANDOM_SEED_SPLIT_2 = 3259
TRAIN_TEST_VAL_PCT = [0.6, 0.2, 0.2]

def split_sizes(split_pct):
    """Return dictionary of percentages for splits 1 and 2 of train / test / val."""

    train_size, test_size, val_size = split_pct
    return {'split_1': test_size, 'split_2': val_size / (1 - test_size)}

def df_train_test_val_split(df, split_pct=TRAIN_TEST_VAL_PCT):
    """Return DataFrame split into train, test, and validation DataFrames.

    Parameters
    ----------
    df: DataFrame
    split_pct: list of length 2, pct of df to use for train, test, and validation sets
    """

    splits = split_sizes(split_pct)
    df_tmp, df_test = df_train_test_split(df, test_size=splits['split_1'], random_seed=RANDOM_SEED_SPLIT_1)
    df_train, df_val = df_train_test_split(df_tmp, test_size=splits['split_2'], random_seed=RANDOM_SEED_SPLIT_2)

    return df_train, df_test, df_val

def df_train_test_split(df, test_size, random_seed=None):
    """Return training and test DataFrames.

    Parameters
    ----------
    df: DataFrame
    test_size: float in the interval (0, 1), percent of df to use as test set
    random_seed: int (optional), sets the random seed for consistent data splits
    """

    # set the seed for consistent train / test data split
    np.random.seed(random_seed)

    # shuffle all rows numbers in df
    nrows = df.shape[0]
    all_rows = np.arange(nrows)
    np.random.shuffle(all_rows)

    # reset random seed to a random value
    scipy.random.seed()

    # split into two sets for training and test data
    test_rows, train_rows = np.split(all_rows, [int(nrows * test_size)])
    df_test = df.ix[test_rows]
    df_train = df.ix[train_rows]

    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)

def get_Xy(df, target_column_name):
    """Return the feature matrix X and target values y from a DataFrame

    Parameters
    ----------
    df: DataFrame, either the training or test data
    target_column_name: string, name of a column in df to use as y

    Returns
    -------
    X: 2d array, feature matrix
    y: 1d array, target
    """

    y = df[target_column_name].values
    df = df.drop(target_column_name, axis=1)
    X = df.values

    return X, y

def train_test_val_split(df, target_column_name):
    """Return train, test, and validation feature matrices and target arrays. Original
    data split using 60% for training and 20% each for testing and validation.

    Returns
    -------
    X_train, y_train, X_test, y_test, X_val, y_val
    """

    splits = []
    for frame in df_train_test_val_split(df):
        splits.extend(get_Xy(frame, target_column_name))

    return splits

def arrays_to_dict(array_keys, array_values):
    """Return dictionary mapping array_keys to array_values."""

    return dict(np.c_[array_keys, array_values])

def train_test_val_dicts(df, key_column, value_column):
    """Return an array of dictionaries for training, test, and validations. Dictionaries
    map key_column to value_column.

    Parameters
    ----------
    df: DataFrame
    key_column: string, column in df to use as dictionary keys
    value_column: string, column in df to use as dictionary values
    """

    arr = []
    for frame in df_train_test_val_split(df):
        key = frame[key_column]
        val = frame[value_column]
        arr.append(arrays_to_dict(key, val))

    return arr

def tokenizer(doc, stem_lem_type='porter'):
    """Return tokenized document after performing the specified stemming or lemmatization.

    Parameters
    ----------
    doc: string, a document to tokenize
    stem_lem_type: string, the type of stemming or lemmatization ('porter', 'snowball',
        or 'lemmatizer')
    """

    words = word_tokenize(doc.lower())
    if stem_lem_type == 'porter':
        porter = PorterStemmer()
        fun = porter.stem
    elif stem_lem_type == 'snowball':
        snowball = SnowballStemmer('english')
        fun = snowball.stem
    else:
        wordnet = WordNetLemmatizer()
        fun = wordnet.lemmatize

    return [fun(w) for w in words]

# wrappers for each tokenizer type
tokenizer_p = lambda doc: tokenizer(doc, stem_lem_type='porter')
tokenizer_s = lambda doc: tokenizer(doc, stem_lem_type='snowball')
tokenizer_l = lambda doc: tokenizer(doc, stem_lem_type='lemmatizer')

def grid_search_naive_bayes_with_tfidf(X_text, y_train, tok=None):
    """Return fit tfidf and MultinomialNB models after grid search to find
    optimal parameters.

    Parameters
    ----------
    X_text: 1d array, array containing text from training data
    y_train: 1d array, training target array
    tok: tokenizer, optional
    """

    tfidf = TfidfVectorizer(stop_words='english', tokenizer=tok)
    X_train = tfidf.fit_transform(X_text)

    NB_grid = {
        'alpha': np.linspace(0, 1, 5),
        'fit_prior': [True, False]
    }

    NB_gridsearch = GridSearchCV(
        MultinomialNB(),
        NB_grid,
        cv=3,
        n_jobs=-1,
        verbose=False,
        scoring='accuracy'
    )

    NB_gridsearch.fit(X_train, y_train)

    print 'Best Naive Bayes Parameters:', NB_gridsearch.best_params_, '\nScore:', NB_gridsearch.best_score_
    return tfidf, NB_gridsearch.best_estimator_

def subsample(X_train, y_train, n=1000):
    """Return X_train, y_train after sampling n rows."""

    rows, _ = X_train.shape
    indices = np.random.choice(rows, n, replace=False)

    return X_train[indices], y_train[indices]

def grid_search_classification(X, y, scaler, cv=3, n=None):
    """Perform grid search using the classifiers:
        - Logistic Regression
        - Random Forest
        - Ada Boost
        - Gradient Boost
        - Support Vector Machine
        - kNN

    Parameters
    ----------
    X: 2d array, feature matrix
    y: 1d array, target
    scaler: StandardScaler, fit scaler object
    cv: int, number of cross validation folds
    n: int (optional), number of rows to subsample from X and y for faster training

    Returns
    -------
    list of best model parameters
    """

    if n:
        X, y = subsample(X, y, n)

    X = scaler.transform(X)

    logistic_grid = {
        'penalty': ['l1', 'l2'],
        'C': [0.5, 1, 2],
        'fit_intercept': [True, False]
    }
    random_forest_grid = {
        'max_depth': [3, None],
        'max_features': ['sqrt', 'log2', None],
        'min_samples_split': [1, 4],
        'min_samples_leaf': [1, 2, 4],
        'n_estimators': [100]
    }
    gradient_boost_grid = {
        'learning_rate': [0.05, 0.1, 0.5, 1],
        'max_depth': [3, 4, 5],
        'min_samples_leaf': [1, 2, 3],
        'max_features': [None, 'sqrt', 'log2'],
        'n_estimators': [50, 100, 200]
    }
    ada_boost_grid = {
        'learning_rate': [0.5, 0.4, 0.1],
        'n_estimators': [50, 100, 200]
    }
    svc_grid = {
        'C': [0.01, .1, 1, 5],
        'kernel': ['poly', 'rbf'],
        'gamma': [0.0, 0.05, .1, .15],
        'degree': [2, 3, 4, 5]
    }
    linear_svc_grid = {
        'C': [0.5, 1, 2],
        'fit_intercept': [True, False]
    }
    knn_grid = {
        'n_neighbors': [3, 5, 7],
        'leaf_size': [20, 30, 40]
    }

    classifier_list = [
        LogisticRegression(),
        RandomForestClassifier(n_jobs=-1),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        SVC(),
        LinearSVC(),
        KNeighborsClassifier()
    ]

    classifier_params = [
        logistic_grid,
        random_forest_grid,
        ada_boost_grid,
        gradient_boost_grid,
        svc_grid,
        linear_svc_grid,
        knn_grid
    ]

    best_model_grid = []
    for model, model_grid in zip(classifier_list, classifier_params):
        print '\n{0}'.format(model.__class__.__name__)
        print "Mean Cross Val Score:", np.mean( cross_val_score(model, X, y, cv=cv, scoring='accuracy') )

        model_gridsearch = GridSearchCV(
            model,
            model_grid,
            cv=cv,
            n_jobs=-1,
            verbose=False,
            scoring='accuracy'
        )
        model_gridsearch.fit(X, y)

        print "Best Model Parameters:", model_gridsearch.best_params_, '\nScore:', model_gridsearch.best_score_
        best_model_grid.append( model_gridsearch.best_estimator_ )

    return best_model_grid

def most_common_element(arr):
    """Returns the most common element in arr, breaking ties randomly."""

    if len(arr) == 0:
        return np.nan

    list_of_tuples = Counter(arr).most_common()
    max_count = list_of_tuples[0][1]
    max_results = [value for value, count in list_of_tuples if count == max_count]

    return random.choice(max_results)

def ensemble_votes(y_pred_list):
    """Return most popular vote from a list of predictions. Ties are broken randomly."""

    return np.apply_along_axis(most_common_element, 0, np.asarray(y_pred_list))
