import numpy as np

from Projection import Projection
from PCA import PCA
from LDA import LDA

class PCALDA(Projection):
    def __init__(self, pca_components=30, n_components=2):
        super().__init__(n_components)
        self.pca_components = pca_components
        self.pca = None
        self.lda = None

    def fit(self, X, y):
        self.pca = PCA(n_components=self.pca_components).fit(X)
        pca_projected = self.pca.project(X)

        self.lda = LDA(n_components=self.n_components).fit(pca_projected, y)

        self.subspace = np.dot(self.pca.pro_subspace, self.lda.pro_subspace)

        return self

    def project(self, X):
        self.check_fitted()
        return np.dot(X - self.pca.X_mean, self.subspace)

    def reconstruct(self, X):
        return X.dot(self.lda.pro_subspace.T).dot(self.pca.pro_subspace.T) + self.pca.X_mean