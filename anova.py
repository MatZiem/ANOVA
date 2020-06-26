import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.utils.validation import check_X_y

class ANOVA():

    def __init__ (self, k = 5):
        self.k = k

    def fit(self, X, y):
        #sprawdza czy rozmiar X i y jest podobny
        X, y = check_X_y(X, y)
        return X, y

    def fit_transform(self, X, y):
        #sprwdzenie wartości F-score i p_value dla kazdej kolumny
        F, p = f_classif(X, y)
        #sortowanie po F-score malejaco
        idx = np.argsort(-F)
        #wybranie k pierwszy wartości z największym F-score
        selected_features = X[:, idx[0:self.k]]
        return selected_features