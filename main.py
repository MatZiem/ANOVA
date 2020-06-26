import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import f_regression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from scipy.stats import rankdata
from scipy.stats import ranksums
from tabulate import tabulate
from anova import ANOVA

k=3
clfs = {
    'GNB': GaussianNB(),
}
redus = {
    'PCA': PCA(n_components=k),
    'SelectKBest': SelectKBest(score_func=f_regression, k=k)
}
datasets = ["australian", "ring", "shuttle", "spectfheart", "thyroid", "wine", "vehicle", "magic", "penbased", "vowel"]
n_datasets = len(datasets)
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1410)
anova = ANOVA(k)
scores = np.zeros((4, n_datasets, n_splits))

#Klasyfikacja dla ANOVA - 0
for data_id, dataset in enumerate(datasets):
    dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X)

    for fold_id, (train, test) in enumerate(skf.split(X, y)):
        X_new = anova.fit_transform(scaled_data, y)
        clf = GaussianNB()
        clf.fit(X_new[train], y[train])
        y_pred = clf.predict(X_new[test])
        scores[0, data_id, fold_id] = accuracy_score(y[test], y_pred)


#Klasyfikacja dla PCA - 1 i SelectKBast - 2
for data_id, dataset in enumerate(datasets):
    dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X)

    for redu_id, redu_name in enumerate(redus):
        redu = clone(redus[redu_name])
        for fold_id, (train, test) in enumerate(skf.split(X, y)):
            X_new = redu.fit_transform(scaled_data, y)
            clf = GaussianNB()
            clf.fit(X_new[train], y[train])
            y_pred = clf.predict(X_new[test])
            scores[redu_id + 1, data_id, fold_id] = accuracy_score(y[test], y_pred)


#Klasyfikacja dla modelu bez selekcji - 3
for data_id, dataset in enumerate(datasets):
    dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    for fold_id, (train, test) in enumerate(skf.split(X, y)):
        clf = GaussianNB()
        clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        scores[3, data_id, fold_id] = accuracy_score(y[test], y_pred)

np.save('results', scores)

scores = np.load('results.npy')

mean_scores = np.mean(scores, axis=2).T
ranks = []
for ms in mean_scores:
    ranks.append(rankdata(ms).tolist())
ranks = np.array(ranks)

mean_ranks = np.mean(ranks, axis=0)

alfa = .05
w_statistic = np.zeros((4, 4))
p_value = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])

#uzyskane statystyki w oraz p-wartości w tabeli
headers = ["ANOVA", "PCA", "SelectKBest", "Model bazowy"]
names_column = np.array([["ANOVA"], ["PCA"], ["SelectKBest"], ["Model bazowy"]])
w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

#wyznaczenie macierzy przewagi
advantage = np.zeros((4, 4))
advantage[w_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
print("\nAdvantage:\n", advantage_table)

#wyznaczenie macierzy istotności
significance = np.zeros((4, 4))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
print("\nStatistical significance (alpha = 0.05):\n", significance_table)

#wyznaczenie macierzy obserwacji końcowych
stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate(
    (names_column, stat_better), axis=1), headers)
print("\nStatistically significantly better:\n", stat_better_table)

