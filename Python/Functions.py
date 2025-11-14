import numpy as np


# Perform a z-transformation on dataset x with given column means and column
# standard deviations.
def do_scale(x, colmeans, colsds):
    """
    Perform z-transformation on dataset `x` with given column means and standard deviations.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        Data to scale.
    colmeans : array-like, shape (n_features,)
        Column means.
    colsds : array-like, shape (n_features,)
        Column standard deviations.

    Returns
    -------
    numpy.ndarray
        Scaled dataset with same shape as `x` (dtype=float).
    """
    x = np.asarray(x, dtype=float)
    colmeans = np.asarray(colmeans, dtype=float)
    colsds = np.asarray(colsds, dtype=float)

    if x.ndim != 2:
        raise ValueError("`x` must be 2-dimensional")
    if x.shape[1] != colmeans.size or x.shape[1] != colsds.size:
        raise ValueError("Number of columns and length of mean and sd vector must be equal")

    # vectorized scaling using broadcasting
    return (x - colmeans) / colsds


# Calculates the euclidean distance between two instances/objects.
def euclid_dist(o1, o2):
    """
    Calculate Euclidean distance between two 1-D arrays.

    Parameters
    ----------
    o1, o2 : array-like
        Input vectors of the same length.

    Returns
    -------
    float
        Euclidean distance.
    """
    o1 = np.asarray(o1, dtype=float)
    o2 = np.asarray(o2, dtype=float)
    if o1.shape != o2.shape:
        raise ValueError("Objects must have the same shape")
    return float(np.linalg.norm(o1 - o2))


# Predict the labels for a given test set (x_test) based on training data (x_train)
# and the corresponding training labels (y_train). Uses 1-NN (nearest neighbour).
def knn_predictions(x_train, x_test, y_train):
    """
    1-NN predictions for `x_test` using `x_train` and `y_train`.

    Parameters
    ----------
    x_train : array-like, shape (n_train, n_features)
    x_test  : array-like, shape (n_test, n_features)
    y_train : array-like, shape (n_train,)

    Returns
    -------
    numpy.ndarray
        Predicted labels for `x_test` (shape (n_test,)).
    """
    x_train = np.asarray(x_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    y_train = np.asarray(y_train)

    if x_train.ndim != 2:
        raise ValueError("`x_train` must be 2-dimensional")
    if x_test.ndim == 1:
        x_test = x_test.reshape(1, -1)
    if x_test.ndim != 2:
        raise ValueError("`x_test` must be 2-dimensional")
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("Length of training labels must be equal to number of rows")
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError("Training and test feature dimension must match")

    ntest = x_test.shape[0]
    y_pred = np.empty(ntest, dtype=y_train.dtype)

    for i in range(ntest):
        tmp = x_test[i]
        # vectorized distance computation to all training points
        dists = np.linalg.norm(x_train - tmp, axis=1)
        min_idx = int(np.argmin(dists))
        y_pred[i] = y_train[min_idx]

    return y_pred


# Calculates the accuracy using the predicted (y_predict) and actual labels (y_test)
def acc(y_predict, y_test):
    """
    Calculate accuracy using predicted and actual labels.

    Parameters
    ----------
    y_predict : array-like
        Predicted labels.
    y_test : array-like
        True labels.

    Returns
    -------
    float
        Accuracy (fraction correct) as a float in [0, 1].
    """
    y_predict = np.asarray(y_predict)
    y_test = np.asarray(y_test)
    if y_predict.shape != y_test.shape:
        raise ValueError("`y_predict` and `y_test` must have the same shape")
    return float(np.mean(y_predict == y_test))
