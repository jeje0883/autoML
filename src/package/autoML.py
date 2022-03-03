import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression


def comparescoreclassifier(X_train, X_test, y_train, y_test):
    names = [
    "RBF SVM",
    "Linear SVM",
    "SGDClassifier",
    "KNeighborsClassifier",
    "GaussianProcessClassifier",
    "GaussianNaiveBayes",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoostClassifier",
    "QuadraticDiscriminantAnalysis",
    "LogisticRegression",
    ]

    classifiers = [
        SVC(gamma=2, C=1),
        LinearSVC( C=0.025),
        SGDClassifier(loss="hinge", penalty="l2", max_iter=10),
        KNeighborsClassifier(3),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        GaussianNB(),
        DecisionTreeClassifier(max_depth=2),
        RandomForestClassifier(max_depth=2, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        QuadraticDiscriminantAnalysis(),
        LogisticRegression(),
    ]

    scoredf = pd.DataFrame()

    for name, classifier in zip(names, classifiers):
        classifier.fit(X_train, y_train)
        testscore = classifier.score(X_test, y_test)
        trainscore = classifier.score(X_train, y_train)
        scoredf2 = pd.DataFrame([[name, testscore, trainscore]])
        scoredf = pd.concat([scoredf, scoredf2], ignore_index=True)

    scoredf.columns=['model','test score','train score']
    print(scoredf)

#if __name__ == '__main__':
    #pass
    #comparescoreclassifier(X_train, X_test, y_train, y_test)
