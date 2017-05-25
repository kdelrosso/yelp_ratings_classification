from __future__ import division
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error


def regression_errors(y_true, y_pred, model_name=''):
    """Print MSE, sqrt(MSE), mean absolute error, and median absolute error"""

    print "\n{0}".format(model_name)
    print "MSE:", mean_squared_error(y_true, y_pred)
    print "Sqrt MSE:", np.sqrt(mean_squared_error(y_true, y_pred))
    print "Mean absolute error:", mean_absolute_error(y_true, y_pred)
    print "Median absolute error:", median_absolute_error(y_true, y_pred)

def classification_errors(y_true, y_pred, model_name=''):
    """Print confusion matrix, accuracy, precision, recall, and F1 scores"""

    # convert decimals to rounded percentages
    perc = lambda x: round(x * 100, 2)

    print "\n{0}".format(model_name)
    print "Confusion Matrix:\n", confusion_matrix(y_true, y_pred)
    print "Accuracy:", perc( accuracy_score(y_true, y_pred) )
    print "Precision:", perc( precision_score(y_true, y_pred, average='weighted') )
    print "Recall:", perc( recall_score(y_true, y_pred, average='weighted') )
    print "F1:", perc( f1_score(y_true, y_pred, average='weighted') )
