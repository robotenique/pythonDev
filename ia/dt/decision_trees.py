import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, roc_curve

def main():
    df = pd.read_csv("bill_authentication.csv")
    """ print(df.head())
    print(df.describe()) # -> The max (Y) is 8 hours
    # Show a whisker boxplot
    df.boxplot()
    # Histogram of every column
    df.hist()
    # Scatter matrix using Kernel Density Estimator
    # -> In this case we see a good separation in the data, no need for machine learning
    scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')
    plt.show() """
    X, y = df.iloc[:, :4], df.iloc[:, 4]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    tree_clf = DecisionTreeClassifier()
    tree_clf.fit(X_train, y_train)
    y_pred = tree_clf.predict(X_test)
    gnb_clf = GaussianNB()
    gnb_clf.fit(X_train, y_train)
    y_pred_gnb = gnb_clf.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr)
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)
    print(confusion_matrix(y_test, y_pred_gnb))
    print(accuracy_score(y_test, y_pred_gnb))
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_gnb)
    plt.plot(fpr, tpr)
    plt.title('ROC curve GAUSSIAN')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)
    plt.show()


    # metrics


if __name__ == '__main__':
    main()