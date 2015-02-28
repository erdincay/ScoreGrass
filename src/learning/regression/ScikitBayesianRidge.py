from sklearn.linear_model import BayesianRidge

__author__ = 'Kern'


class ScikitBayesianRidge:
    def __init__(self):
        self.clf = BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False,
                                 copy_X=True, fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06,
                                 n_iter=300, normalize=False, tol=0.001, verbose=False)

    def train(self, x_data, y_data):
        self.clf.fit(x_data, y_data)

    def predict(self, x_data):
        return self.clf.predict(x_data)