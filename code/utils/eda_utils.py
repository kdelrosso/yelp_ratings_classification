from __future__ import division
import re
import string
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from preprocessing_utils import text_tokenizer

def plot_sentiment(df):
    """Plot a bar chart of the sentiment rate (sentiment score / length of review)
    from each sentiment source segmented by the number of review stars.
    """

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
    cols = ['stars', 'senti_1_rate', 'senti_2_rate']
    df[cols].groupby('stars').mean()['senti_1_rate'].plot(kind='bar', ax=ax[0])
    df[cols].groupby('stars').mean()['senti_2_rate'].plot(kind='bar', ax=ax[1])

def plot_hist(df, ax, plot_col):
    """Plot 3 histograms for star rating on a single figure.

    Parameters
    ----------
    df: DataFrame, containing columns for 'stars' and plot_col
    ax: Axes, location to plot histogram
    plot_col: string, column in df with data to use for histogram
    """

    mask_1 = df['stars'] == 1
    mask_3 = df['stars'] == 3
    mask_5 = df['stars'] == 5
    df.loc[mask_5, plot_col].plot(kind='hist', bins=100, alpha=0.5, ax=ax)
    df.loc[mask_3, plot_col].plot(kind='hist', bins=100, alpha=0.5, ax=ax)
    df.loc[mask_1, plot_col].plot(kind='hist', bins=100, alpha=0.5, ax=ax)

    # plot a vertical dashed line at the mean
    mean_dict = df.groupby('stars').mean()[plot_col].to_dict()
    for val in mean_dict.itervalues():
        ax.axvline(val, ls = '--', color = 'black')

    if plot_col == 'senti_1_rate':
        ax.set_xlim([-0.15, 0.15])
    else:
        ax.set_xlim([-0.3, 0.3])

def plot_sentiment_hists(df):
    """Plot histograms for sentiment rates for each star rating (1, 3, and 5)."""

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    cols = ['stars', 'senti_1_rate', 'senti_2_rate']
    for col, a in zip(['senti_1_rate', 'senti_2_rate'], ax):
        plot_hist(df[cols], ax=a, plot_col=col)

def plot_punctuation(df):
    """Plot a bar chart of the mean number of questions marks and exclamation points
    segmented by the number of review stars.
    """

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    cols = ['stars', 'q_mark', 'e_point']
    df[cols].groupby('stars').mean()['q_mark'].plot(kind='bar', ax=ax[0])
    df[cols].groupby('stars').mean()['e_point'].plot(kind='bar', ax=ax[1])

def get_unique_word_counts(text):
    """Print count of unique words (with and without stop words) for the input text string."""

    all_words, without_stop = text_tokenizer(text)
    print 'Number of unique words in all reviews: {0}'.format(len(set(all_words)))
    print 'Number of unique words (excluding stop words): {0}'.format(len(set(without_stop)))

def preprocess_review_text(df):
    """Return a string after removing all punctuation and extra white space from a column
    of text in a DataFrame.
    """

    text = df['text'].str.lower().str.replace("'", '')
    for punct in string.punctuation:
        text = text.str.replace(punct, ' ')

    # make all white space a single space
    text = text.str.replace(r'\s+', ' ')
    return ' '.join(text.values)

def unique_work_count(df):
    return get_unique_word_counts(preprocess_review_text(df))

def show_all_eda(df):
    plot_sentiment(df)
    plot_sentiment_hists(df)
    plot_punctuation(df)
    unique_work_count(df)
    plt.show()
