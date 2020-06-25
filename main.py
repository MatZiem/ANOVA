import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold
from f_score import ANOVA
from sklearn.decomposition import PCA



# load data
data = np.genfromtxt("australian.csv", delimiter=",")

X = data[:, :-1]
y = data[:, -1]

anova = ANOVA(k=6)
anova.fit(X, y)
X_new = anova.fit_transform(X, y)







#n_samples, n_features = X.shape    # number of samples and number of features

# split data into 10 folds
#ss = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=43)
# perform evaluation on classification task
#num_fea = 3  # number of selected features
#clf = GaussianNB()    # Gausian

# obtain the f-score of each feature
#score = ANOVA(X, y)

# rank features in descending order according to score
#idx = feature_ranking(score)

# obtain the dataset on the selected features
#selected_features = X[:, idx[0:num_fea]]

#for train_index, test_index in ss.split(X, y):
    # train a classification model with the selected features on the training dataset
    #clf.fit(selected_features[train_index], y[train_index])

    # predict the class labels of test data
    #y_predict = clf.predict(selected_features[test_index])

    # obtain the classification accuracy on the test data
    #acc = accuracy_score(y[test_index], y_predict)

# output the average classification accuracy over all 10 folds
#print('Accuracy:', float(acc))