from sklearn.linear_model import MultiTaskLasso

__author__ = 'Kern'


class ScikitMultiTaskLasso:
    def __init__(self):
        self.clf = MultiTaskLasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
                                  normalize=False, tol=0.0001, warm_start=False)

    def train(self, x_data, y_data):
        self.clf.fit(x_data, y_data)

    def predict(self, x_data):
        return self.clf.predict(x_data)