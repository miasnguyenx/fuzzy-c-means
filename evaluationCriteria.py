from collections import Counter
import math
from scipy.spatial import distance


class EvaluationCriteria:
    def __init__(self, df, labels):
        self.df = df
        self.n = len(df)
        self.k = len(set(labels))
        self.labels = labels

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

    def ASWC(self):
        n = self.n
        k = self.k
        labels = self.labels
        df = self.df
        eps = math.pow(10, -6)
        dst = self.initDistancesMatrix()

        rows, cols = (n, k)
        InterAVGdist = [[0 for i in range(cols)] for j in range(rows)]
        IntraAVGdist = [0]*n
        minInterAVG = [0]*n
        occ = Counter(labels)

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

        ASWC = sum((minInterAVG[i] / (IntraAVGdist[i] + eps))
                   for i in range(n))
        ASWC /= n
        # print(np.array(ASWC_matrix))
        print("ASWC_index: ", ASWC)
        return ASWC_matrix
