from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from DecisionTrees import DecisionTrees

data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234)


clf = DecisionTrees()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)


def accuracy(pred, y_test):
    return (pred == y_test).mean()


acc = accuracy(pred, y_test)
print("Accuracy: {:.2f}%".format(acc * 100))
