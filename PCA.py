from Projection import Projection

class PCA(Projection):
     def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__X_mean = None
        self.__eigenvalues = None

    def fit(self, X, y=None):
        assert X.ndim == 2, 'X can only be a 2-d matrix'

        # Center the data
        self.__X_mean = np.mean(X, axis=0)
        X = X - self.__X_mean
        n, d = X.shape

        # If the d >> n then use dual PCA for efficiency
        use_dual_pca = d > n

        if use_dual_pca:
            X = X.T
            n, d = d, n

        # calculate the covariance matrix
        C = np.dot(X.T, X) / (n - 1)

        U, S, V = np.linalg.svd(C)

        if use_dual_pca:
            U = X.dot(U).dot(np.diag(1 / np.sqrt(S * (n - 1))))

        self.__subspace = U
        self.__eigenvalues = S

        return self

    def project(self, X):
        self.__check_fitted()
        X = X - self.__X_mean
        return np.dot(X, self.pro_subspace)

    def reconstruct(self, X):
        self._check_fitted()
        return np.dot(X, self.pro_subspace.T) + self.__X_mean