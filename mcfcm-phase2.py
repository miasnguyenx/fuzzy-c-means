from collections import Counter
from sklearn import metrics
import sklearn
from sklearn.metrics import davies_bouldin_score
import pandas as pd  # reading all required header files
import numpy as np
import random
import operator
import math
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.stats import multivariate_normal  # for generating pdf
import csv
from evaluationCriteria import EvaluationCriteria


class MCFCMeansPhase2():
    def __init__(self):
        self.maxIter = 100
        self.eps = math.pow(10, -6)
        # self.e = 0.01

    def read_file(self, name):

        df_full = pd.read_csv(name)  # iris data
        df_full.head()

        # Drop a column
        df_full = df_full.drop(["Id"], axis=1)
        df_full.shape

        columns = list(df_full.columns)  # column now is an array variable

        # features is column without species attribute
        features = columns[: len(columns) - 1]
        # print(features)
        self.class_labels = list(df_full[columns[-1]])
        # print(class_labels)
        self.df = df_full[features]
        self.n = len(self.df)

    def set_param(self, k, m, mL, mU, alpha):
        self.m = m
        self.k = k
        self.mL = mL
        self.mU = mU
        self.alpha = alpha

    def accuracy(self, cluster_labels, class_labels):
        df = self.df
        correct_pred = 0
        # find max of a set base on key value as the element has the most occurences
        seto = max(set(cluster_labels[0:50]), key=cluster_labels[0:50].count)
        vers = max(set(cluster_labels[50:100]),
                   key=cluster_labels[50:100].count)
        virg = max(set(cluster_labels[100:]), key=cluster_labels[100:].count)

        for i in range(len(df)):
            if cluster_labels[i] == seto and class_labels[i] == "Iris-setosa":
                correct_pred = correct_pred + 1
            if (
                cluster_labels[i] == vers
                and class_labels[i] == "Iris-versicolor"
                and vers != seto
            ):
                correct_pred = correct_pred + 1
            if (
                cluster_labels[i] == virg
                and class_labels[i] == "Iris-virginica"
                and virg != seto
                and virg != vers
            ):
                correct_pred = correct_pred + 1

        accuracy = correct_pred / len(df) * 100
        return accuracy

    def initDistancesMatrix(self):
        n = self.n
        df = self.df
        rows, cols = (n, n)
        dst = [[0 for i in range(cols)] for j in range(rows)]
        for i in range(n):
            x = list(df.iloc[i])
            for j in range(n):
                y = list(df.iloc[j])
                dst[i][j] = distance.euclidean(x, y)
        return dst

    def initializeMembershipMatrix(self):  # initializing the membership matrix
        n = self.n
        k = self.k
        membership_mat = []
        # initialize the membership maxtrix for each objects
        for i in range(n):
            random_num_list = [random.random()
                               for i in range(k)]  # random a k-values array

            summation = sum(random_num_list)  # sum of all values in array
            temp_list = [x / summation for x in random_num_list]

            # attach the object to a cluster (0,1,2) - respectively to three iris categories
            flag = temp_list.index(max(temp_list))
            for j in range(0, len(temp_list)):
                if j == flag:
                    temp_list[j] = 1
                else:
                    temp_list[j] = 0

            membership_mat.append(temp_list)
        return membership_mat

    def calculateClusterCenter_phase1(self, membership_mat):
        n = self.n
        k = self.k
        df = self.df
        m = self.m
        cluster_mem_val = list(zip(*membership_mat))
        cluster_centers = []
        for j in range(k):
            x = list(cluster_mem_val[j])  # Uik
            xraised = [p ** m
                       for f, p in enumerate(x)]  # Uik power by m
            denominator = sum(xraised)
            temp_num = []
            for i in range(n):
                data_point = list(df.iloc[i])
                prod = [xraised[i] * val for val in data_point]  # Uik^m * Xi
                temp_num.append(prod)
            # sum of list(zip(*temp_num)) is numerator of v[k]
            numerator = map(sum, list(zip(*temp_num)))
            center = [z / denominator for z in numerator]
            cluster_centers.append(center)

        return cluster_centers

    def updateMembershipValue_phase1(self, membership_mat, cluster_centers):
        n = self.n
        k = self.k
        df = self.df
        m = self.m
        p = 2 / (m-1)
        for i in range(n):
            x = list(df.iloc[i])
            # D(i, k)
            distances = [
                np.linalg.norm(
                    np.array(list(map(operator.sub, x, cluster_centers[j]))))
                for j in range(k)
            ]
            #
            for j in range(k):
                den = sum(
                    [math.pow(float(distances[j] / distances[c]), p)
                     for c in range(k)]
                )
                membership_mat[i][j] = float(1 / den)
        return membership_mat

    # calculating the cluster center
    def calculateClusterCenter_phase2(self, membership_mat, MCFCMCoeff):
        n = self.n
        k = self.k
        df = self.df
        cluster_mem_val = list(zip(*membership_mat))
        cluster_centers = []
        for j in range(k):
            x = list(cluster_mem_val[j])  # Uik
            xraised = [p ** MCFCMCoeff[f]
                       for f, p in enumerate(x)]  # Uik power by m
            denominator = sum(xraised)
            temp_num = []
            for i in range(n):
                data_point = list(df.iloc[i])
                prod = [xraised[i] * val for val in data_point]  # Uik^m * Xi
                temp_num.append(prod)
            # sum of list(zip(*temp_num)) is numerator of v[k]
            numerator = map(sum, list(zip(*temp_num)))
            center = [z / denominator for z in numerator]
            cluster_centers.append(center)

        return cluster_centers

    def updateMembershipValue_phase2(self, membership_mat, cluster_centers, MCFCMCoeff):
        n = self.n
        k = self.k
        df = self.df
        for i in range(n):
            x = list(df.iloc[i])
            # D(i, k)
            distances = [
                np.linalg.norm(
                    np.array(list(map(operator.sub, x, cluster_centers[j]))))
                for j in range(k)
            ]
            #
            for j in range(k):
                coeff = 2 / (MCFCMCoeff[i] - 1)
                den = sum(
                    [math.pow(float(distances[j] / distances[c]), coeff)
                     for c in range(k)]
                )
                membership_mat[i][j] = float(1 / den)
        return membership_mat

    def getClusters(self, membership_mat):  # getting the clusters
        n = self.n
        cluster_labels = list()
        for i in range(n):
            # index of x - val is the value of enumerate list
            max_val, idx = max((val, idx)
                               for (idx, val) in enumerate(membership_mat[i]))
            cluster_labels.append(idx)
        return cluster_labels

    def FCM(self):
        # Membership Matrix
        membership_mat = self.initializeMembershipMatrix()
        curr = 0
        acc = []
        while curr < self.maxIter:
            cluster_centers = self.calculateClusterCenter_phase1(
                membership_mat)
            membership_mat = self.updateMembershipValue_phase1(
                membership_mat, cluster_centers)
            cluster_labels = self.getClusters(membership_mat)

            acc.append(cluster_labels)
            if curr == 0:
                print("Cluster Centers:")
                print(np.array(cluster_centers))
            curr += 1
        self.labels = cluster_labels
        print("---------------------------")
        print("Partition matrix:")
        for i in range(0, 150):
            print(i, np.max(membership_mat[i]), np.array(
                membership_mat)[i], cluster_labels[i])
        print(np.array(cluster_labels))
        print(np.array(cluster_labels).shape)

        a = self.accuracy(cluster_labels, self.class_labels)
        print("Accuracy = ", a)
        print("DB_score: ", sklearn.metrics.davies_bouldin_score(
            self.df, cluster_labels))
        return cluster_labels, cluster_centers, acc

    def ASWCoefficientMatrix(self):
        self.labels, centers, acc = self.FCM()
        mU = self.mU
        mL = self.mL
        alpha = self.alpha
        tmp = EvaluationCriteria(self.df, self.labels)
        ASWC_matrix = tmp.ASWC()

        FuzzyCoeff = [0]*150
        max = np.max(ASWC_matrix)
        min = np.min(ASWC_matrix)
        for i in range(150):
            X_std = math.pow((ASWC_matrix[i] - min) / (max - min), 2.5)
            FuzzyCoeff[i] = X_std * (mU - mL) + mL
        return FuzzyCoeff

    def MCFCM_phase2(self):
        # Membership Matrix
        membership_mat = self.initializeMembershipMatrix()
        curr = 0
        acc = []
        MCFCMCoeff = self.ASWCoefficientMatrix()
        print(np.array(MCFCMCoeff))
        while curr < self.maxIter:
            cluster_centers = self.calculateClusterCenter_phase2(
                membership_mat, MCFCMCoeff)
            membership_mat = self.updateMembershipValue_phase2(
                membership_mat, cluster_centers, MCFCMCoeff)
            cluster_labels = self.getClusters(membership_mat)
            acc.append(cluster_labels)

            if curr == 0:
                print("Cluster Centers:")
                print(np.array(cluster_centers))
            curr += 1
        for i in range(0, 150):
            print(i, np.max(membership_mat[i]), np.array(
                membership_mat)[i], cluster_labels[i])
        print(np.array(cluster_labels))
        a = self.accuracy(cluster_labels, self.class_labels)
        print("Accuracy = ", a)
        return cluster_labels, cluster_centers, acc


X = MCFCMeansPhase2()
X.read_file('Iris.csv')
X.set_param(3, 2, 1.1, 9.1, 2.5)
labels, centers, acc = X.MCFCM_phase2()
df = X.df
X1 = EvaluationCriteria(df, labels)
X1.ASWC()
print("DB_score: ", sklearn.metrics.davies_bouldin_score(df, labels))
print(np.array(labels))

print(Counter(labels))
