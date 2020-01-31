# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals


def answer_01(data):
    """
    Write a code which standardizes the column of given data EXCEPT THE LAST ONE !

    Args
    ----
        data [pandas.DataFrame]: the data which should be standardized

    Returns
    -------
        scaled_data [pandas.DataFrame]: Rescaled data except last column

    """
    # TODO : standardize dat
    scaled_data = data
    scaled_data.iloc[:, :2] = (data.iloc[:,:2]-data.iloc[:,:2].mean())/data.iloc[:,:2].std()
    return scaled_data



def answer_02():
    """
    Import KNeighborsClassifier from the scikit-learn library 

    Returns
    -------
        nearest_neighbors class
    """
    # Wrong classifier
    from sklearn.neighbors import KNeighborsClassifier as KNei

    return sklearn_model = KNei()


def answer_03():
    """
    Import balanced_accuracy_score from scikit-learn

    Args
    ----

    Returns
    -------
    """
    from sklearn.metrics import balanced_accuracy_score as sklearn_metric
    return sklearn_metric


def answer_04():
    """
    In this context :
    Does re-scaling variables always help ?

    Returns
    -------
        YES or NO

    """
    YES = 'YES'
    NO = 'NO'
    # Return YES or NO
    return NO


def answer_05():
    """
    In which case does it help most?

    Returns
    -------
     CASE1, CASE2, CASE3 or CASE4
    """
    CASE1 = "RAW"
    CASE2 = "RE"
    CASE3 = "CS"
    CASE4 = "CROP"
    # Return CASE1, CASE2, CASE3 or CASE4
    return CASE3
    #Sur le graphique obtenu dans le notebook TP2, la colonne 1 (basée sur 0) est la bonne. Notre professeur m'ayant dit que mes résultats étant valides, la mention "CORRECT" ne m'est pas nécessaire.


def answer_06():
    """
    If the test performance is bad but the training performance is good,
     is the model under-fitting or over-fitting? 
    """
    under_fitting = "under-fitting"
    over_fitting  = "over-fitting"
    # Return under_fitting or over_fitting
    return over_fitting


def answer_07():
    """
    If both are bad, is the model is under-fitting or over-fitting ? 
    """
    under_fitting = "under-fitting"
    over_fitting  = "over-fitting"
    # Return under_fitting or over_fitting
    return under_fitting


def answer_08():
    """
    Which models are over-fitting ?
    """
    model_name = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

    overfitting_models = ['RBF SVM', 'Decision Tree', 'Random Forest' ]
    return overfitting_models


def answer_09():
    """
    Which models are are under-fitting ?
    """
    model_name = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

    underfitting_models = ['Linear SVM', 'Naive Bayes']
    return underfitting_models





