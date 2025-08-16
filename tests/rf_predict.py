from random import random
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.ensemble import RandomForestClassifier

#from random_forest import RandomForest
from ren_algorithm import RandomForest

data = datasets.load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

clf = RandomForest(n_trees=20)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

custom_acc =  accuracy(y_test, predictions)

def built_in_model(X_train,X_test, y_train, y_test):
    # Create a Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=20, random_state=42)
    # Train the model
    rf_model.fit(X_train, y_train)
    # Make predictions
    y_pred = rf_model.predict(X_test)
    return accuracy(y_test, y_pred)


builtin_acc = built_in_model(X_train,X_test, y_train, y_test)

print("\n Built-in random forest model accuracy: ", builtin_acc)
print("\n Custom Random forest model accuracy: ", custom_acc)
