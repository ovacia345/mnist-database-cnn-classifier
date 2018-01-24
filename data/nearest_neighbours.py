# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:27:04 2018

@author: Niels-laptop
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.neighbors as nbrs
import sklearn.metrics as mtc

training_data = np.load("training_set.npz")
X_train = training_data["images"]
y_train = training_data["labels"]

test_data = np.load("test_set.npz")
X_test = test_data["images"]
y_test = test_data["labels"]

accuracies = []
for K in range(25,41):
    neigh = nbrs.KNeighborsClassifier(n_neighbors = K, p = 2)
    neigh.fit(X_train, y_train)
    
    accuracies.append(neigh.score(X_test, y_test))
    print(str(K) + ": " + str(accuracies[len(accuracies) - 1]))
    
    
# plt.scatter(range(1, 41), accuracies)
# plt.title("Scatterplot of classification accuracy against number of nearest neighbours")
# plt.xlabel("Number of nearest neighbours")
# plt.ylabel("Accuracy")
# plt.show()
    

