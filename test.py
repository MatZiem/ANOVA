import numpy as np
import pandas as pd
from sklearn import datasets
import scipy.stats as stats
from sklearn.model_selection import train_test_split

dataset = np.genfromtxt("australian.csv", delimiter=",")
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=.3,
    random_state=42
)

n, k = X.shape
x_mean = X.mean(axis = 0)
overall_mean = x_mean.mean()

sb = n*(x_mean-overall_mean)**2
ssb = sb.sum()
dfb = k - 1
msb = ssb/dfb

sw = (X - x_mean)**2
ssw = sw.sum()
dfw = k*(n-1)
msw = ssw/dfw
F = msb/msw

alfa = 0.05
F_crit = stats.f.ppf(1-alfa, dfb, dfw)