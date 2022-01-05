import sklearn
from fuzzyCmeans import FuzzyCmeans
from mcfcmPhase2 import MCFCMeansPhase2
from evaluationCriteria import EvaluationCriteria
import numpy as np
import pandas as pd

silhouette = list()
bouldin = list()
alpha = list()
randindex = list()
alpha_arr = np.arange(0.5, 1, 0.5)
# print(len(alpha_arr))
# exit()

for i in alpha_arr:
    X = MCFCMeansPhase2()
    X.read_file("Ecoli.csv")
    X.set_param(8, 2, 1.1, 9.1, 0.5)
    labels, centers, acc = X.MCFCM_phase2()
    df = X.df
    X1 = EvaluationCriteria(df, labels)
    tmp, a = X1.ASWC()
    a = round(a, 3)
    b = sklearn.metrics.davies_bouldin_score(X.df, labels)
    b = round(b, 3)
    r = sklearn.metrics.rand_score(labels, X.class_labels)
    r = round(r, 3)
    silhouette.append(a)
    bouldin.append(b)
    randindex.append(r)
    del X
    del X1
    print("------", i, "------")


print(silhouette)
print(bouldin)
print(randindex)
