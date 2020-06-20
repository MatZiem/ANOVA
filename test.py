import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import feature_selection
import scipy.stats as stats
from sklearn.feature_selection import SelectKBest, f_classif

data = pd.read_csv("hepatitis.csv", header=None)
#dataset = np.genfromtxt("hepatitis.csv", delimiter=",")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

F, p = feature_selection.f_classif(X, y)
print(p<0.05)
X_new = SelectKBest(f_classif, 6).fit_transform(X, y)
#n, k = X.shape
#x_mean = X.mean(axis = 0)
#overall_mean = x_mean.mean()

#sb = n*(x_mean-overall_mean)**2
#ssb = sb.sum()
#dfb = k - 1
#msb = ssb/dfb

#sw = (X - x_mean)**2
#ssw = sw.sum()
#dfw = k*(n-1)
#msw = ssw/dfw
#F = msb/msw

#alfa = 0.05
#F_crit = stats.f.ppf(1-alfa, dfb, dfw)