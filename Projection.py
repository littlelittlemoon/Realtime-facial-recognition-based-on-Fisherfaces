class Projection:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.subspace = None

    def fit(self, X, y):
        """
        Description: Fit the projection onto the training data.
        Input:
        X: images array
        y: labels
        """

    def project(self, X):
        """
        Description: Project the new data using the fitted projection matrices.
        Input:
        X: images array
        """

    def reconstruct(self, X):
        """
        Description: Reconstruct the projected data back into the original space.
        Input:
        X: projected array
        """

    def check_fitted(self):
        """
        Description: Check that the projector has been fitted.
        """
        assert self.subspace is not None, \
            'You must fit %s before you can project' % self.__class__.__name__

    @property
    def pro_subspace(self):
        self.check_fitted()
        return self.subspace[:, :self.n_components]