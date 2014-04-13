# Author: Kien Trinh

import time
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn import decomposition
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation as cv

def load_decompose_data(K):
    '''Fit the model with X and apply the dimensionality reduction on X.'''
    rawtrain = pd.read_csv('CSVs/train.csv', header=None)
    rawtest = pd.read_csv('CSVs/test.csv', header=None)
    label = pd.read_csv('CSVs/trainLabels.csv', header=None)    
    
    pca = decomposition.PCA(n_components=K, whiten=True)
    train = pca.fit_transform(rawtrain)            
    test = pca.transform(rawtest)
    return train, label.T.values[0], test

def classifier(train, label, test):
    cvk = cv.StratifiedKFold(label, n_folds = 3)
    params = {'gamma': [0.1, 1./3, 2./3, 1.],'C': [1e+1, 1e+2, 1e+4, 1e+6]}

    clf = GridSearchCV(SVC(), param_grid=params, cv=cvk, n_jobs=-1)
    clf.fit(train, label)
    
    # Estimate score
    scores = cv.cross_val_score(clf.best_estimator_, train, label, cv=60)
    print('Estimated score: %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))
    
    # Predict and save
    result = clf.best_estimator_.predict(test)
    return result

if __name__ == '__main__':
    t0 = time.clock()  
    
    predTest = pd.read_csv('CSVs/testLabels.csv')
    train, label, test = load_decompose_data(12)
    
    predTest['Solution'] = classifier(train, label, test)
    
    predTest.to_csv("CSVs/submission.csv",index=False)
    
    print 'Running time = ' + str(time.clock() - t0)

#=====================================#
# Study affect of dimension reduction #
#=====================================#

KK = range(2,16,1) + range(16,42,2)
data = []
for k in KK:
    train, label, _ = load_decompose_data(k)
    tuned_parameters = {'kernel': ['rbf'], 'gamma': [1, 1e-1, 1e-2], 'C': [1e-1, 1./3, 1, 10./3, 10]}
    svm1 = GridSearchCV( SVC(), tuned_parameters, cv=4, n_jobs=-1, verbose=2 ).fit(train, label)
    data += [svm1.best_score_]    

import matplotlib.pyplot as plt
plt.plot(KK, data, 'ro')
plt.axis([0, 40, 0, 1])
plt.xlabel('K features')
plt.ylabel('Score')
plt.title('Score vs. # of K features')
plt.show()
