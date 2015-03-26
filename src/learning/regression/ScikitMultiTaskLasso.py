from sklearn.linear_model import MultiTaskLasso
from src.learning.regression.ScikitLearning import ScikitModel

__author__ = 'Kern'


class MultiTaskLasso(ScikitModel):
    def __init__(self):
        ScikitModel.__init__(self)
        self.model = MultiTaskLasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000, normalize=False, tol=0.0001, warm_start=False)