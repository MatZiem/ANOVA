import numpy as np
import pandas as pd
from scipy import special, stats
from scipy.sparse import issparse
from sklearn.feature_selection import f_classif
from sklearn.utils import (as_float_array, check_array, check_X_y, safe_sqr,
                     safe_mask)

dataset = 'australian'
dataset = np.genfromtxt("%s.csv" % (dataset), delimiter=",")
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)

X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'])
args = [X[safe_mask(X, y == k)] for k in np.unique(y)]

n_classes = len(args)
args = [as_float_array(a) for a in args]
n_samples_per_class = np.array([a.shape[0] for a in args])
n_samples = np.sum(n_samples_per_class)
ss_alldata = sum(safe_sqr(a).sum(axis=0) for a in args)
sums_args = [np.asarray(a.sum(axis=0)) for a in args]
square_of_sums_alldata = sum(sums_args) ** 2
square_of_sums_args = [s ** 2 for s in sums_args]
sstot = ss_alldata - square_of_sums_alldata / float(n_samples)
ssbn = 0.
for k, _ in enumerate(args):
    ssbn += square_of_sums_args[k] / n_samples_per_class[k]
ssbn -= square_of_sums_alldata / float(n_samples)
sswn = sstot - ssbn
dfbn = n_classes - 1
dfwn = n_samples - n_classes
msb = ssbn / float(dfbn)
msw = sswn / float(dfwn)
constant_features_idx = np.where(msw == 0.)[0]
if (np.nonzero(msb)[0].size != msb.size and constant_features_idx.size):
    warnings.warn("Features %s are constant." % constant_features_idx,
                    UserWarning)
f = msb / msw
# flatten matrix to vector in sparse case
#f = np.asarray(f).ravel()
prob = special.fdtrc(dfbn, dfwn, f)

i = np.argsort(prob)
X = X[:,i]
X = X[:,0:5]
