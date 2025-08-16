import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

#from logistic_regression import LogisticRegression
from ren_algorithm import LogisticRegression

#loading dataset diabetes
ds = datasets.load_diabetes()
X, y = ds.data, ds.target

# View the features in the dataset
print(ds.feature_names, "\n")

# Convert y to binary value if the diabetes progression is above average, classify as 1
#y = (y > y.mean()).astype(int)
y = (y >= 0.5).astype(int)

#spliting the datast into train and test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 1234)


cls = LogisticRegression()
cls.learn(x_train,y_train)
y_pred = cls.predict(x_test)

#calculating the acccuracy
def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)


ac = accuracy(y_pred, y_test)
print("\n",ac)

