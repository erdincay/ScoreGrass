from sklearn.linear_model import BayesianRidge

from src.learning.regression.ScikitLearning import ScikitModel

__author__ = 'Kern'


class ScikitBayesianRidge(ScikitModel):
    def __init__(self, s_id):
        super().__init__(
            BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False, copy_X=True, fit_intercept=True,
                          lambda_1=1e-06, lambda_2=1e-06, n_iter=300, normalize=False, tol=0.001, verbose=False),
            s_id)
