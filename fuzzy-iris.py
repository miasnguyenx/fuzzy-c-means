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

# Show data
df_full = pd.read_csv("Iris.csv")  # iris data
df_full.head()

# Drop a column
df_full = df_full.drop(["Id"], axis=1)
df_full.shape

columns = list(df_full.columns)  # column now is an array variable

# features is column without species attribute
features = columns[: len(columns) - 1]
# print(features)
class_labels = list(df_full[columns[-1]])
# print(class_labels)
df = df_full[features]
print(df.head())
# Number of Clusters
#
#
# ---------------- Parameters -------------------
#
#
# Number of Clusters
k = 3
# Maximum number of iterations untill termination
MAX_ITER = 100
# Number of data points
n = len(df)
# Fuzzy parameter
m = 2  # Select a value greater than 1 else it will be knn
# terminated creterion for FCM clustering
e = 0.01
mL = 1.1
mU = 9.1
alpha = 2.5


def accuracy(cluster_labels, class_labels):
    correct_pred = 0
    # find max of a set base on key value as the element has the most occurences
    seto = max(set(cluster_labels[0:50]), key=cluster_labels[0:50].count)
    vers = max(set(cluster_labels[50:100]), key=cluster_labels[50:100].count)
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


def initializeMembershipMatrix():  # initializing the membership matrix
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


membership_mat = initializeMembershipMatrix()
print(membership_mat)
print(list(zip(*membership_mat)))

# Param: membership matrix


def calculateClusterCenter(membership_mat):  # calculating the cluster center
    cluster_mem_val = list(zip(*membership_mat))

    cluster_centers = []
    for j in range(k):
        x = list(cluster_mem_val[j])  # Uik
        xraised = [p ** m for p in x]  # Uik power by m
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


cluster_centers = calculateClusterCenter(membership_mat)


def initDistancesMatrix():
    rows, cols = (n, n)
    dst = [[0 for i in range(cols)] for j in range(rows)]
    for i in range(n):
        x = list(df.iloc[i])
        for j in range(n):
            y = list(df.iloc[j])
            dst[i][j] = distance.euclidean(x, y)
    return dst


def fuzzyCoefficientMatrix():
    distances = initDistancesMatrix()
    delta = [0] * n
    dist = [0] * n
    fuzzyCoeff = [0] * n
    for i in range(n):
        distances[i].sort()
        for j in range(int(n / k)):
            delta[i] += distances[i][j]
    # print(delta[0])
    deltaMin = min(delta)
    deltaMax = max(delta)

    # print(deltaMax)
    # print(deltaMin)
    for i in range(n):
        fuzzyCoeff[i] = mL + (mU - mL) * (
            math.pow((delta[i] - deltaMin) / (deltaMax - deltaMin), alpha)
        )
    print(np.array(fuzzyCoeff))
    return fuzzyCoeff


def updateMembershipValue(membership_mat, cluster_centers):
    p = float(2 / (m - 1))
    for i in range(n):
        x = list(df.iloc[i])
        distances = [
            np.linalg.norm(
                np.array(list(map(operator.sub, x, cluster_centers[j]))))
            for j in range(k)
        ]
        for j in range(k):
            den = sum(
                [math.pow(float(distances[j] / distances[c]), p)
                 for c in range(k)]
            )
            membership_mat[i][j] = float(1 / den)
    return membership_mat


MCFCMCoeff = fuzzyCoefficientMatrix()


def updateMembershipValue2(membership_mat, cluster_centers):

    for i in range(n):
        x = list(df.iloc[i])
        distances = [
            np.linalg.norm(
                np.array(list(map(operator.sub, x, cluster_centers[j]))))
            for j in range(k)
        ]
        for j in range(k):
            den = sum(
                [math.pow(float(distances[j] / distances[c]), MCFCMCoeff[i])
                 for c in range(k)]
            )
            membership_mat[i][j] = float(1 / den)
    return membership_mat


def getClusters(membership_mat):  # getting the clusters
    cluster_labels = list()
    for i in range(n):
        # index of x - val is the value of enumerate list
        max_val, idx = max((val, idx)
                           for (idx, val) in enumerate(membership_mat[i]))
        cluster_labels.append(idx)
    return cluster_labels


def fuzzyCMeansClustering():  # Third iteration Random vectors from data
    # Membership Matrix
    membership_mat = initializeMembershipMatrix()
    curr = 0
    acc = []
    while curr < MAX_ITER:
        cluster_centers = calculateClusterCenter(membership_mat)
        membership_mat = updateMembershipValue(membership_mat, cluster_centers)
        cluster_labels = getClusters(membership_mat)

        acc.append(cluster_labels)

        if curr == 0:
            print("Cluster Centers:")
            print(np.array(cluster_centers))
        curr += 1
    print("---------------------------")
    print("Partition matrix:")
    for i in range(0, 149):
        print(np.max(membership_mat[i]), np.array(
            membership_mat)[i], cluster_labels[i])

    # print(np.array(cluster_labels))
    # print(np.array(cluster_labels).shape)
    # return cluster_labels, cluster_centers
    return cluster_labels, cluster_centers, acc


def MCFCMeansClustering():
    # Membership Matrix
    membership_mat = initializeMembershipMatrix()
    curr = 0
    acc = []
    while curr < MAX_ITER:
        cluster_centers = calculateClusterCenter(membership_mat)
        membership_mat = updateMembershipValue2(
            membership_mat, cluster_centers)
        cluster_labels = getClusters(membership_mat)

        acc.append(cluster_labels)

        if curr == 0:
            print("Cluster Centers:")
            print(np.array(cluster_centers))
        curr += 1
    print("---------------------------")
    print("Partition matrix:")
    for i in range(0, 149):
        print(np.max(membership_mat[i]), np.array(
            membership_mat)[i], cluster_labels[i])

    # print(np.array(cluster_labels))
    # print(np.array(cluster_labels).shape)
    # return cluster_labels, cluster_centers
    return cluster_labels, cluster_centers, acc


labels, centers, acc = MCFCMeansClustering()

# labels, centers, acc = fuzzyCMeansClustering()


def DaviesBouldinIndex():
    occ = Counter(labels)
    djk = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            djk[i][j] = distance.euclidean(centers[i], centers[j])

    dj = [0]*3
    for i in range(k):
        center = centers[i]
        label = labels[i]
        for j in range(n):
            if labels[j] == label:
                x = df.iloc[j]
                dj[i] += distance.euclidean(x, center)
        dj[i] /= occ[i]

    DB = 0
    for i in range(k):
        Djay = 0
        for j in range(k):
            if i != j:
                x = (abs(dj[i] - dj[j])) / djk[i][j]
                if x > Djay:
                    Djay = x
        DB += Djay

    DB /= k
    print("DB_index: ", DB)
    return DB


def ASWC():
    eps = math.pow(10, -6)
    dst = initDistancesMatrix()

    rows, cols = (n, k)
    InterAVGdist = [[0 for i in range(cols)] for j in range(rows)]
    IntraAVGdist = [0]*n
    minInterAVG = [0]*n
    occ = Counter(labels)
    # print(occ[0], occ[1], occ[2])

    rows, cols = (k, n)
    cluster = [[0 for i in range(cols)] for j in range(rows)]
    count = 0

    for i in range(n):
        label = labels[i]
        for j in range(n):
            if labels[j] == label:
                IntraAVGdist[i] += dst[i][j]
                count += 1
        IntraAVGdist[i] /= count
        count = 0

    for i in range(n):
        x = list(df.iloc[i])
        y = labels[i]
        cluster[y].append(x)

    for i in range(n):
        for j in range(n):
            if labels[j] != labels[i]:
                t = labels[j]
                InterAVGdist[i][t] += dst[i][j]

    # print(np.array(InterAVGdist))
    ASWC_matrix = [0]*150
    for i in range(n):
        for j in range(k):
            if InterAVGdist[i][j] != 0:
                InterAVGdist[i][j] /= occ[j]

        InterAVGdist[i].sort()
        minInterAVG[i] = InterAVGdist[i][1]
        ASWC_matrix[i] = minInterAVG[i] / (IntraAVGdist[i] + eps)
    # print(np.array(InterAVGdist))
    # print(np.array(IntraAVGdist))

    ASWC = sum((minInterAVG[i] / (IntraAVGdist[i] + eps)) for i in range(n))
    ASWC /= n
    # print(np.array(ASWC_matrix))
    print("ASWC_index: ", ASWC)
    return ASWC_matrix, ASWC


print("DB_score: ", sklearn.metrics.davies_bouldin_score(df, labels))
DaviesBouldinIndex()
a = accuracy(labels, class_labels)
print("Accuracy = ", a)

ASWC_maxtrix, ASWC = ASWC()


def fuzzyCoefficientMatrix2():
    FuzzyCoeff = [0]*150
    max = np.max(ASWC_maxtrix)
    min = np.min(ASWC_maxtrix)
    for i in range(150):
        X_std = (max - ASWC_maxtrix[i]) / (max - min)
        FuzzyCoeff[i] = X_std * (mU - mL) + mL
    print(np.array(FuzzyCoeff))
    return FuzzyCoeff


FuzzyCoeff2 = fuzzyCoefficientMatrix2()


def updateMembershipValue_phase2(membership_mat, cluster_centers):
    for i in range(n):
        x = list(df.iloc[i])
        distances = [
            np.linalg.norm(
                np.array(list(map(operator.sub, x, cluster_centers[j]))))
            for j in range(k)
        ]
        for j in range(k):
            den = sum(
                [math.pow(float(distances[j] / distances[c]), FuzzyCoeff2[i])
                 for c in range(k)]
            )
            membership_mat[i][j] = float(1 / den)
    return membership_mat


def MCFCMeansClustering_phase2():
    # Membership Matrix
    membership_mat = initializeMembershipMatrix()
    curr = 0
    acc = []
    while curr < MAX_ITER:
        cluster_centers = calculateClusterCenter(membership_mat)
        membership_mat = updateMembershipValue_phase2(
            membership_mat, cluster_centers)
        cluster_labels = getClusters(membership_mat)

        acc.append(cluster_labels)

        if curr == 0:
            print("Cluster Centers:")
            print(np.array(cluster_centers))
        curr += 1
    print("---------------------------")
    print("Partition matrix:")
    for i in range(0, 149):
        print(np.max(membership_mat[i]), np.array(
            membership_mat)[i], cluster_labels[i])

    # print(np.array(cluster_labels))
    # print(np.array(cluster_labels).shape)
    return cluster_labels, cluster_centers, acc


labels2, centers2, acc2 = MCFCMeansClustering_phase2()

print("DB_score: ", sklearn.metrics.davies_bouldin_score(df, labels2))
# ASWCValidationCriteria()
print(Counter(labels))
print(Counter(labels2))
a = accuracy(labels2, class_labels)
print("Accuracy = ", a)

print("Cluster center final:")  # final cluster centers
print(np.array(centers))
