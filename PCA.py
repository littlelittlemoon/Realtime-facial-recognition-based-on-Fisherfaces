from Projection import Projection

class PCA(Projection):
     def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X_mean = None
        self.eigenvalues = None

    def fit(self, X, y=None):
        assert X.ndim == 2, 'X can only be a 2-d matrix'

        # Center the data
        self.X_mean = np.mean(X, axis=0)
        X = X - self.X_mean
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

        self.subspace = U
        self.eigenvalues = S

        return self

    def project(self, X):
        self.check_fitted()
        X = X - self.X_mean
        return np.dot(X, self.pro_subspace)

    def reconstruct(self, X):
        self.check_fitted()
        return np.dot(X, self.pro_subspace.T) + self.X_mean