import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from f_score import ANOVA
from sklearn.decomposition import PCA



# load data
data = np.genfromtxt("sonar.csv", delimiter=",")

X = data[:, :-1]
y = data[:, -1]

anova = ANOVA(k=6)
anova.fit(X, y)
X_new = anova.fit_transform(X, y)

folds = 5
# split data into 5 folds
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1410)
# perform evaluation on classification task
clf = GaussianNB()    # Gausian

for folds, (train, test) in enumerate(skf.split(X,y)):
    # train a classification model with the selected features on the training dataset
    clf.fit(X_new[train], y[train])

    # predict the class labels of test data
    y_predict = clf.predict(X_new[test])

    # obtain the classification accuracy on the test data
    results = accuracy_score(y[test], y_predict)

# output the average classification accuracy over all 10 folds
print('Accuracy:', float(results))