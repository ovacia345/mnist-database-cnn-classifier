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
for K in range(1,31):
    neigh = nbrs.KNeighborsClassifier(n_neighbors = K, p = 2)
    neigh.fit(X_train, y_train)
    
    accuracies.append(neigh.score(X_test, y_test))
    print(str(K) + ": " + str(accuracies[-1]))
    
    
#plt.scatter(range(1, 31), accuracies)
plt.scatter(range(1, 31), [0.9691, 0.9627, 0.9705, 0.9682, 0.9688, 0.9677, 0.9694, 0.967, 0.9659, 0.9665, 0.9668, 0.9661, 0.9653, 0.964, 0.9633,
            0.9632, 0.963, 0.9633, 0.9632, 0.9625, 0.963, 0.9618, 0.9619, 0.9608, 0.9609, 0.9612, 0.9604, 0.9602, 0.9593, 0.9596])
plt.title("Scatterplot of classification accuracy against number of nearest neighbours")
plt.xlabel("Number of nearest neighbours")
plt.ylabel("Accuracy")
plt.show()
    

