from collections import Counter
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
# Number of Clusters
#
#
# ---------------- Parameters -------------------
#
#
# Number of Clusters
k = 3
# Maximum number of iterations untill termination
MAX_ITER = 50
# Number of data points
n = len(df)
# Fuzzy parameter
m = 2  # Select a value greater than 1 else it will be knn
# terminated creterion for FCM clustering 
e = 0.01

#
#
# ----------------End Parameters -------------------
#
#

# temp_list = [x for x in random_num_list]
# print(temp_list)

# 
#
# --------------- Graph show --------------
#
#
plt.figure(figsize=(10, 10))  # scatter plot of sepal length vs sepal width
plt.scatter(list(df.iloc[:, 0]), list(df.iloc[:, 1]), marker="o")
plt.axis("equal")
plt.xlabel("Sepal Length", fontsize=16)
plt.ylabel("Sepal Width", fontsize=16)
plt.title("Sepal Plot", fontsize=22)
plt.grid()
plt.show()

plt.figure(figsize=(10, 10))  # scatter plot of petal length vs sepal width
plt.scatter(list(df.iloc[:, 2]), list(df.iloc[:, 3]), marker="o")
plt.axis("equal")
plt.xlabel("Petal Length", fontsize=16)
plt.ylabel("Petal Width", fontsize=16)
plt.title("Petal Plot", fontsize=22)
plt.grid()
plt.show()
# 
#
# ---------------End Graph show --------------
#
#

#Calculate accuracy for a specify dataset
def accuracy(cluster_labels, class_labels):
    correct_pred = 0
    # find max of a set base on key value as the element has the most occurences
    seto = max(set(labels[0:50]), key=labels[0:50].count)
    vers = max(set(labels[50:100]), key=labels[50:100].count)
    virg = max(set(labels[100:]), key=labels[100:].count)

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
# print(membership_mat)


def calculateClusterCenter(membership_mat):  # calculating the cluster center
    cluster_mem_val = list(zip(*membership_mat))
    cluster_centers = []
    for j in range(k):
        x = list(cluster_mem_val[j])
        xraised = [p ** m for p in x]
        denominator = sum(xraised)
        temp_num = []
        for i in range(n):
            data_point = list(df.iloc[i])
            prod = [xraised[i] * val for val in data_point]
            temp_num.append(prod)
        # sum of list(zip(*temp_num)) is numerator of v[i]
        numerator = map(sum, list(zip(*temp_num)))
        center = [z / denominator for z in numerator]
        cluster_centers.append(center)
    return cluster_centers


cluster_centers = calculateClusterCenter(membership_mat)
# print(cluster_centers)
# calculateClusterCenter(membership_mat)

# print(list(df.iloc[1]))
# x = list(df.iloc[1])
# distances = [np.linalg.norm(
#             np.array(list(map(operator.sub, x, cluster_centers[j])))) for j in range(k)]
# print(distances)
# Updating the membership value
#


def initDistancesMatrix():
    rows, cols = (150, 150)
    dst = [[0 for i in range(cols)] for j in range(rows)]
    for i in range(n):
        x = list(df.iloc[i])
        for j in range(n):
            y = list(df.iloc[j])
            dst[i][j] = distance.euclidean(x, y)
    return dst


def fuzzyCoefficientMatrix():
    distances = initDistancesMatrix()
    mL = 1
    mU = 10
    alpha = 2
    delta = [0] * 150
    dist = [0] * 150
    fuzzyCoeff = [0] * 150
    for i in range(n):
        distances[i].sort()
        for j in range(int(n / k)):
            delta[i] += distances[i][j]
    # print(delta[0])
    delta.sort()
    deltaMin = min(delta)
    deltaMax = max(delta)

    # print(deltaMax)
    # print(deltaMin)
    for i in range(n):
        fuzzyCoeff[i] = mL + (mU - mL) * (
            math.pow((delta[i] - deltaMin) / (deltaMax - deltaMin), alpha)
        )

    return fuzzyCoeff


MCFCMCoeff = fuzzyCoefficientMatrix()


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
    # for i in range (0,149):
    # print(np.max(membership_mat[i]), np.array(membership_mat)[i], cluster_labels[i])

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
    # print("---------------------------")
    # print("Partition matrix:")
    # for i in range(0, 149):
    #     print(np.max(membership_mat[i]), np.array(membership_mat)[i], cluster_labels[i])

    # print(np.array(cluster_labels))
    # print(np.array(cluster_labels).shape)
    # return cluster_labels, cluster_centers
    return cluster_labels, cluster_centers, acc


labels, centers, acc = MCFCMeansClustering()
# labels, centers, acc = fuzzyCMeansClustering()

def ASWCValidationCriteria():
    eps = math.pow(10, -6)
    dst = initDistancesMatrix()
    rows, cols = (150, 3)
    InterAVGdist = [[0 for i in range(cols)] for j in range(rows)]
    IntraAVGdist = [0]*150
    minInterAVG = [0]*150
    occ = Counter(labels)
    print(occ[0], occ[1], occ[2])

    rows, cols = (3, 150)
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

    print(np.array(InterAVGdist))

    for i in range(n):
        for j in range(k):
            if InterAVGdist[i][j] != 0:
                InterAVGdist[i][j] /= occ[j]

        InterAVGdist[i].sort()
        minInterAVG[i] = InterAVGdist[i][-2]
    
    print(np.array(InterAVGdist))
    print(np.array(IntraAVGdist))

    ASWC = sum((minInterAVG[i] / (IntraAVGdist[i] + eps)) for i in range (n))
    ASWC /= n
    return ASWC

print(ASWCValidationCriteria())

# labels, centers, acc = fuzzyCMeansClustering()
a = accuracy(labels, class_labels)
# print(labels[1])
# print(class_labels[0])
# P.S. The accuracy calculation is for iris data only

# acc_lis = []
# for i in range(0, len(acc)):
#     val = accuracy(acc[i], class_labels)
#     acc_lis.append(val)
# acc_lis = np.array(acc_lis)  # calculating accuracy and std deviation 100 times
# print("mean=", np.mean(acc_lis))
# print("Std dev=", np.std(acc_lis))
print("Accuracy = ", a)
# accuracy = accuracy(labels, class_labels)

# accuracy = accuracy(labels, class_labels)
# print("---------------------------")
# print("Accuracy: " ,accuracy)
print("Cluster center final:")  # final cluster centers
print(np.array(centers))
# for i in range (n):
#     print(i, labels[i])

# print(list(df.iloc[1]))
# sepal_df = df_full.iloc[:,0:2]
# sepal_df = np.array(sepal_df)
# #m1 = [0,0]
# #m2 = [0,0]
# #m3 = [0,0]
# #Second initialization
# #m1 = [-0.47534495, -0.16392118]
# #m2 = [0.89019389, -1.19440781]
# #m3 = [1.29107135, 0.48248487]
# #Third initialization
# m1 = random.choice(sepal_df)
# m2 = random.choice(sepal_df)
# m3 = random.choice(sepal_df)

# cov1 = np.cov(np.transpose(sepal_df))
# cov2 = np.cov(np.transpose(sepal_df))
# cov3 = np.cov(np.transpose(sepal_df))

# x1 = np.linspace(4,8,150)
# x2 = np.linspace(1.5,4.5,150)
# #x1 = np.linspace(-1,8,150)
# #x2 = np.linspace(-1,4.5,150)
# X, Y = np.meshgrid(x1,x2)

# Z1 = multivariate_normal(m1, cov1)
# Z2 = multivariate_normal(m2, cov2)
# Z3 = multivariate_normal(m3, cov3)

# pos = np.empty(X.shape + (2,)) ## a new array of given shape and type, without initializing entries

# pos[:, :, 0] = X; pos[:, :, 1] = Y

# plt.figure(figsize=(10,10)) # creating the figure and assigning the size
# plt.scatter(sepal_df[:,0], sepal_df[:,1], marker='o')
# plt.contour(X, Y, Z1.pdf(pos), colors="r" ,alpha = 0.5)
# plt.contour(X, Y, Z2.pdf(pos), colors="b" ,alpha = 0.5)
# plt.contour(X, Y, Z3.pdf(pos), colors="g" ,alpha = 0.5)
# plt.axis('equal') # making both the axis equal
# plt.xlabel('Sepal Length', fontsize=16) # X-Axis
# plt.ylabel('Sepal Width', fontsize=16)  # Y-Axis
# plt.title('Initial Random Clusters(Sepal)', fontsize=22) # Title of the plot
# plt.grid() # displaying gridlines
# plt.show()
