import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pprint import pprint
from time import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

def main():
    filename = "training.json"
    #visualize(filename)
    #new_cases = proc_input()
    X_train, X_test, y_train, y_test, classifier = train(filename)
    evaluate_model(X_test, y_test, classifier)
    #new_cases_predicted = classifier.predict(new_cases)
    #print(new_cases_predicted[:10])

def train(filename):
    X, y = proc_data(filename)
    """Split our data in train and test (33%)"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
    """ Create a pipeline to process data and predict in a more readable manner """
    text_clf = Pipeline([
        ('vect', CountVectorizer(max_df=.5, ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(max_iter=50, penalty='l2', alpha=1e-05)),
    ])
    text_clf = text_clf.fit(X_train.ravel(), y_train)
    return X_train, X_test, y_train, y_test, text_clf

def proc_input():
    concatenate = lambda el: el["question"].replace("\n", "")+" "+el["excerpt"].replace("\n", "")
    npred = int(input(""))
    json_questions = [json.loads(input("")) for _ in range(npred)]
    new_cases = list(map(concatenate, json_questions))
    return new_cases

def proc_data(filename):
    """ Process the data in json and return the X and y in numpy format"""
    fp = open(filename, "r")
    nrows = int(fp.readline())
    json_dt = fp.readlines()
    json_txt = "["+",".join([l for l in json_dt[:len(json_dt) - 1]])
    json_txt += "]"
    y = []
    x = []
    json_data = json.loads(json_txt)
    for el in json_data:
        # I'm concatenating the question and the excerpt together
        x.append([el["question"].replace("\n","")+" "+el["excerpt"].replace("\n","")])
        y.append(el["topic"])
    X = np.array(x, dtype=str)
    y = np.array(y, dtype=str)
    return X, y


def evaluate_model(X_test, y_test, classifier):
    """ Model evaluation metrics """
    # In this case we need to flatten the array
    y_pred = classifier.predict(X_test.ravel())
    # Accuracy score
    print('True:', y_test[0:5])
    print('Pred:', y_pred[0:5])
    # The classes in our classifier
    print(f"Classes = {classifier.classes_}")
    # The confusion matrix
    print(f" Confusion Matrix = \n{metrics.confusion_matrix(y_test, y_pred)}\n\n")
    # Classification Accuracy: how often is the classifier correct
    print(f" Classification Accuracy = {metrics.accuracy_score(y_test, y_pred)}")
    # Classification Error: how often is the classifier incorrect
    print(f" Classification Error = {1 - metrics.accuracy_score(y_test, y_pred)}")
    # RECALL/Sensitivity: When the actual value is positive, how often is the prediction correct?
    # print(f" Recall = {metrics.recall_score(y_test, y_pred)}")
    # This will return the precision scores for each class
    print(f"\nRECALL (per class) = \n{metrics.recall_score(y_test, y_pred, average=None)}")
    # will return the total ratio of tp/(tp + fp)
    print(f"RECALL (overall) = {metrics.recall_score(y_test, y_pred, average='micro')}")
    # PRECISION: When a positive value is predicted, how often is the prediction correct?
    # print(f"PRECISION = {metrics.precision_score(y_test, y_pred)}") # Can't use default because we have more than 2 classes
    print(f"PRECISION (per class) = {metrics.precision_score(y_test, y_pred, average=None)}")
    print(f"PRECISION (overall) = {metrics.precision_score(y_test, y_pred, average='micro')}")
    print("---------------------------------------------------------------------")
    print(f"{metrics.classification_report(y_test, y_pred)}")




def hyperparameter_search():
    # multiprocessing requires the fork to happen in a __main__ protected
    # block
    X, y = proc_data("training.json")
    """Split our data in train and test (33%)"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # #############################################################################
    # Define a pipeline combining a text feature extractor with a simple
    # classifier
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier()),
    ])

    # uncommenting more parameters will give better exploring power but will
    # increase processing time in a combinatorial way
    parameters = {
        'vect__max_df': (0.5, ),
        #'vect__max_features': (None, 5000, 10000, 50000),
        # 'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        #'clf__max_iter': (5,),
        #'clf__alpha': (0.00001, 0.000001),
        #'clf__penalty': ('l2', 'elasticnet'),
        'clf__max_iter': (10, 50, 80),
    }
    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, cv=5,
                               n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(X.ravel(), y)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


def visualize(filename):
    X, y = proc_data(filename)
    """Split our data in train and test (33%)"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    """Create a sparse matrix representation of a Bag of Words"""
    # Also, removes stop words in english
    #vectorizer = CountVectorizer(stop_words='english')
    #X_train_bagw = vectorizer.fit_transform(X_train.ravel())

    # The shape of our bag of words
    # print(X_train_bagw.shape)
    # Show what is the feature array like.
    # Since X_train_bagw is a sparse data structure, this can be quite costly
    # print(X_train_bagw.toarray()[:10, :10])
    # Print some of the words name
    # print(np.random.choice(vectorizer.get_feature_names(), size=10))
    # Get what index is the word 'recharging' mapped to
    # print(vectorizer.vocabulary_.get(u'recharging'))
    """ Create a pipeline to process data and predict in a more readable manner """
    text_clf = Pipeline([
        ('vect', CountVectorizer(max_df=.5, ngram_range=(1, 2), stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(max_iter=50, penalty='elasticnet', alpha=1e-05)),
    ])
    text_clf = text_clf.fit(X_train.ravel(), y_train)
    #print(np.mean(y_pred == y_test))


if __name__ == '__main__':
    main()
