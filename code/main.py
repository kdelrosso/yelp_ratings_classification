from __future__ import division
from os import path
from utils.db_utils import run_query
from utils.preprocessing_utils import ReviewPreprocessing
from utils.eda_utils import show_all_eda
from utils.model_utils import *
from utils.results_utils import classification_errors
from two_stage_tfidf_model import NaiveBayesTwoStageClassifier
from doc2vec_model import Doc2VecClassifier


PROJECT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
YELP_REVIEWS_QUERY = """
    SELECT
        review_id,
        text,
        stars
    FROM reviews
    ;
    """
KEEP_COLS = [
    'stars',
    'text',
    'words',
    'q_mark',
    'e_point',
    'senti_1_rate',
    'senti_1_neg_cnt',
    'senti_2_rate',
    'senti_2_neg_cnt'
]

def get_reviews_data(filename, use_saved):
    """Return DataFrame of reviews data, either load the data from disk
    or process the raw data and saving results.

    Parameters
    ----------
    filename: string, location to load or save the DataFrame
    use_saved: bool, use True to load data from filename, False
        to process the raw data and saving results to filename.
    """

    print 'Loading the data...'
    if use_saved:
        preprocess = ReviewPreprocessing(None)
        df = preprocess.load_df(filename)
    else:
        df = run_query(YELP_REVIEWS_QUERY)
        preprocess = ReviewPreprocessing(df)
        preprocess.update_features()
        preprocess.save_df(filename)
        df = preprocess.df

    return df

def get_all_models():
    """Return list of classification models: random forest, adaboost, linear SVM and kNN,
    with parameters found using grid search.
    """

    rf = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=4,
        n_jobs=-1,
        min_samples_split=1,
        max_features='sqrt',
        max_depth=None
    )
    aboost = AdaBoostClassifier(
        n_estimators=200,
        learning_rate=0.1
    )
    gboost = GradientBoostingClassifier(
        max_features='sqrt',
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=1
    )
    lin_svm = LinearSVC(
        C=1,
        fit_intercept=True
    )
    knn = KNeighborsClassifier(
        n_neighbors=7,
        leaf_size=20,
        n_jobs=-1
    )

    return [rf, aboost, gboost, lin_svm, knn]

def grid_search_stage_two(X_train, y_train, text_column_index=0):
    """Perform grid search to determine the optimal stage two model."""

    nb2 = NaiveBayesTwoStageClassifier(None)
    X_train = nb2.fit_stage_one(X_train, y_train, text_column_index)
    scaler = StandardScaler().fit(X_train)

    return grid_search_classification(X_train, y_train, scaler)

def two_stage_model(X_train, y_train, X_test, y_test):
    """Fit, predict, and show classification results for the two stage model."""

    print 'Fitting the two stage models...'
    # fit all models using the training data
    all_models = get_all_models()
    fit_models = []
    for model in all_models:
        nb2 = NaiveBayesTwoStageClassifier(model)
        nb2.fit(X_train, y_train)
        fit_models.append(nb2)

    # predict all models using the test data
    predictions = []
    for model in fit_models:
        y_pred = model.predict(X_test)
        stage_two_model_name = model.stage_two_model.__class__.__name__
        classification_errors(y_test, y_pred, stage_two_model_name)
        predictions.append(y_pred)

    # classification results for the ensemble model
    classification_errors(y_test, ensemble_votes(predictions), 'Ensemble')

def doc2vec_model(df, train_dict, test_dict, k):
    """Fit, predict, and show classification results for the Doc2Vec model.

    Parameters
    ----------
    df: DataFrame, full dataset
    train_dict: dictionary, training data mapping labels to stars
    test_dict: dictionary, test data mapping labels to stars
    k: number of nearest vectors to use for classification

    Returns
    -------
    trained Doc2Vec model
    """

    print 'Fitting the Doc2Vec model...'
    d2v = Doc2VecClassifier()
    d2v.fit(df['text'].values, df['review_id'].values)
    y_pred = d2v.predict(train_dict, test_dict, k)
    y_test = test_dict.values()

    classification_errors(y_test, y_pred, 'Doc2Vec with k = {0}'.format(k))

    return d2v.model

def doc2vec_examples(yelp_model):
    """Print examples using the Doc2Vec model trained with yelp reviews data."""

    print '\nDoc2Vec examples:'
    print yelp_model.most_similar('pizza')
    print yelp_model.doesnt_match('burrito taco nachos pasta'.split())
    print yelp_model.doesnt_match('waiter server bartender napkin'.split())
    print yelp_model.most_similar(positive=['bar', 'food'], negative=['alcohol'])
    print yelp_model.most_similar(positive=['drink', 'hot', 'caffeine'])


if __name__ == '__main__':

    # load and preprocess the yelp reviews data
    df = get_reviews_data(PROJECT_DIR + '/data/saved_df.pkl', use_saved=True)

    # exploratory data analysis plots
    show_all_eda(df)

    # split data into training, test, and validation sets
    df_train, df_test, df_val = df_train_test_val_split(df[KEEP_COLS])
    X_train, y_train, X_test, y_test, X_val, y_val = train_test_val_split(df[KEEP_COLS], 'stars')
    train_dict, test_dict, val_dict = train_test_val_dicts(df, 'review_id', 'stars')

    # use grid search to optimize stage 1 & 2 model parameters
    grid_search_naive_bayes_with_tfidf(df_train['text'].values, df_train['stars'].values)
    grid_search_stage_two(X_train, y_train)

    # fit and print classification results for the two stage tfidf model
    two_stage_model(X_train, y_train, X_test, y_test)

    # fit and print classification results for the Doc2Vec model
    yelp_model = doc2vec_model(df, train_dict, test_dict, k=25)

    # use the Doc2Vec model
    doc2vec_examples(yelp_model)
