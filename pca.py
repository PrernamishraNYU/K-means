import pandas as pd
import numpy as np

class PCA:
    def __init__(self, matrix):
        self.matrix = matrix

    def calculate(self):
        features = self.matrix.T
        self.cov_matrix = np.cov(features)

        values, vectors = np.linalg.eig(self.cov_matrix)
        max_num = (-values).argsort()[:2]
        max1 = max_num[0]
        max2 = max_num[1]

        po1 = self.matrix.dot(vectors.T[max1])
        po2 = self.matrix.dot(vectors.T[max2])

        pdata = pd.DataFrame(po1, columns=['pc1'])
        pdata['pc2'] = po2

        return pdata