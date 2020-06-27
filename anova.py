import numpy as np
from sklearn.utils.validation import check_X_y

class ANOVA():

    def __init__ (self, k = 5):
        self.k = k

    def fit(self, X, y):
        #sprawdza czy rozmiar X i y jest podobny
        X, y = check_X_y(X, y)
        return X, y

    def fit_transform(self, X, y):
        #połączenie X oraz y (do obliczen)
        data = np.concatenate((X, y[:, np.newaxis]), axis=1)
        #zdobycie liczby wierszy
        alldata = len(data)
        #zdobycie liczby klas problemu
        n_unique = len(np.unique(y))
        #srednia każdej z kolumn
        overall_mean = np.mean(data[:, :-1], axis=0)

        #utworzenie tablicy na tablice klas problemu
        class_specific = []
        self.i = 0
        while self.i < n_unique:
            temp = data[np.where(data[:, -1] == self.i)]
            temp = temp[:, :-1]
            class_specific.append(temp)
            self.i += 1

        #utworzenie tabliczy na wyliczenie liczebności dla każdej z klas
        number_of_classes = []
        self.i = 0
        while self.i < n_unique:
            temp = len(class_specific[self.i])
            number_of_classes.append(temp)
            self.i += 1

        #utworzenie tablicy na wyliczenie sredniej dla kazdej z kolumn dla wartosci nalezacych do danej klasy
        mean_of_classes = []
        self.i = 0
        while self.i < n_unique:
            temp = np.mean(class_specific[self.i], axis=0)
            mean_of_classes.append(temp)
            self.i += 1

        #utworzenie tablicy na wyliczenie wartosci MSB dla kazdej z kolumn dla wartosci nalezacych do danej klasy
        MSB = []
        self.i = 0
        while self.i < n_unique:
            temp = (number_of_classes[self.i] * (mean_of_classes[self.i] - overall_mean) ** 2) / (n_unique - 1)
            MSB.append(temp)
            self.i += 1
        #wyliczenie sumy MSB dla kazdej kolumny pomiędzy klasami
        MSB = np.sum(MSB, axis=0)

        #utworzenie tablicy na wyliczenie wartosci MSW dla kazdej z kolumn dla wartosci nalezacych do danej klasy
        MSW = []
        for self.i in range(n_unique):
            for self.j in range(number_of_classes[self.i]):
                temp = ((class_specific[self.i][self.j] - mean_of_classes[self.i]) ** 2) / (alldata - n_unique)
                MSW.append(temp)
        #wyliczenie sumy MSW dla kazdej kolumny pomiędzy klasami
        MSW = np.sum(MSW, axis=0)

        #wyliczenie wartosci F dla kazdej z kolumn
        F = MSB / MSW
        #sortowanie po F-score malejaco
        idx = np.argsort(-F)
        #wybranie k pierwszy wartości z największym F-score
        selected_features = X[:, idx[0:self.k]]
        return selected_features