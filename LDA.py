from Projection import Projection

class LDA(Projection):
    def __init__(self, auto_components=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eigenvalues = None
        self.class_means = None
        self.auto_components = auto_components
    
    def compute_Sw_Sb(X, y, class_means, full_mean, n_classes, n_samples):
        Sw, Sb = 0, 0

        for i in range(n_classes):
            # Compute the within class scatter matrix: Sw
            for j in X[y == i]:
                tmp = np.atleast_2d(j - class_means[i])
                Sw += np.dot(tmp.T, tmp)

            # Compute the between class scatter matrix: Sb
            tmp = np.atleast_2d(class_means[i] - full_mean)
            Sb += n_samples * np.dot(tmp.T, tmp)

        return Sw, Sb

    def compute_eigenvalues_and_eigenvectors(Sw, Sb):
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(np.dot(np.linalg.inv(Sw), Sb))

        # Ascending order by eigenvalues
        sorted_idx = np.argsort(eigenvalues)[::-1]

        return eigenvalues[sorted_idx], eigenvectors[:, sorted_idx]

    def fit(self, X, y):
        assert X.shape[0] == y.shape[0], 'X and y dimensions do not match.'

        n_classes = np.max(y) + 1
        n_samples, n_features = X.shape

        if self.auto_components:
            self.n_fisherfaces = n_classes - 1
        else:
            assert self.n_fisherfaces <= n_classes, \
                'LDA has (c - 1) non-zero eigenvalues. ' \
                'Please change n_fisherfaces to <= '

        # Compute the class means
        class_means = np.zeros((n_classes, n_features))

        for i in range(n_classes):
            class_means[i, :] = np.mean(X[y == i], axis=0)

        full_mean = np.mean(class_means, axis=0)

        Sw, Sb = compute_Sw_Sb(X, y, class_means, full_mean, n_classes, n_samples)
        # print(Sw, Sb)
       
        # Get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = compute_eigenvalues_and_eigenvectors(Sw, Sb)

        self.subspace = eigenvectors.astype(np.float64)
        self.eigenvalues = eigenvalues

        self.class_means = np.dot(class_means, self.pro_subspace)

        return self

    def project(self, X):
        self.check_fitted()
        return np.dot(X, self.pro_subspace)

    def reconstruct(self, X):
        self.check_fitted()
        return np.dot(X, self.pro_subspace.T)