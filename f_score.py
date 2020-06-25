import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.utils.validation import check_X_y

class ANOVA():

    def __init__ (self, k = 5):
        self.k = k

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        return X, y

    def fit_transform(self, X, y):
        F, p = f_classif(X, y)
        idx = np.argsort(F)
        idx_t = idx[::-1]
        selected_features = X[:, idx[0:self.k]]
        return selected_features