from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import pandas as pd

#from decision_tree import DecisionTree

from ren_algorithm import DecisionTree

data = datasets.load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = DecisionTree(max_depth=10)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

custom_acc = accuracy(y_test, predictions)


def built_in_model(model):
    model.fit(X,y)
    predictions = model.predict(X)
    return accuracy(y, predictions)

sk_decision_tree = DecisionTreeClassifier(min_samples_split=2, max_depth=100, max_features=None )

builtin_acc = built_in_model(sk_decision_tree)


results = [
    ["Built-in decision tree classification ", builtin_acc],
    ["Custom decision tree classification ", custom_acc]
]

df = pd.DataFrame(results, columns=["Model", "accuracy"])
print("\n",df,"\n")