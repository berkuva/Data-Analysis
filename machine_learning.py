import random

class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train


    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = random.choice(self.y_train)
            predictions.append(label)

        return predictions

#import a dataset
from sklearn import datasets
iris = datasets.load_iris()
# print iris

X = iris.data
# print X.shape
y = iris.target
# print len(y)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

# from sklearn.neighbors import KNeighborsClassifier
my_classifier = ScrappyKNN()
# my_classifier = KNeighborsClassifier()

# from sklearn import tree
# my_classifier = tree.DecisionTreeClassifier()
my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)
# print predictions

from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)



'''
4. Training and testing data, pipelining

#import a dataset
from sklearn import datasets
iris = datasets.load_iris()
# print iris

X = iris.data
# print X.shape
y = iris.target
# print len(y)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()
my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)
# print predictions

from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)
'''

'''
3. features

import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500

grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

plt.hist([grey_height, lab_height], stacked = True, color = ['r', 'b'])
plt.show()
'''

'''
# 2. Train and test real dataset (Iris)

import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

# from IPython.display import Image

iris = load_iris()
# print iris.feature_names
# print iris.target_names
# print iris.data[0]
# print iris.target[0] #labels

test_idx = [0, 50, 100]

print iris.target
# print iris.data

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

print test_target
print clf.predict(test_data)
'''

'''
visualization

from sklearn.externals.six import StringIO
import pydot

dot_data = StringIO()
tree.export_graphviz(clf, out_file = dot_data,
                     feature_names = iris.feature_names,
                     class_names = iris.target_names,
                     filled = True,
                     rounded = True,
                     impurity = False)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")
'''

# for i in range(len(iris.target)):
#     print "Example %d: labels %s, features %s" % (i, iris.target[i], iris.data[i])


"""
1. Hello World

from sklearn import tree

features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

print clf.predict([[150, 0]])

"""