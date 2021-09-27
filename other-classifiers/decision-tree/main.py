import csv
import numpy as np
import graphviz
import pickle
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold

x_train = pickle.load(open('../../X_train.pickle', 'rb'))
y_train = pickle.load(open('../../y_train.pickle', 'rb'))
x_test = pickle.load(open('../../X_test.pickle', 'rb'))
y_test = pickle.load(open('../../y_test.pickle', 'rb'))
x = np.concatenate((x_train, x_test), axis=0)
y = np.concatenate((y_train, y_test), axis=None)

kf = StratifiedKFold(n_splits=10)
n_fold = 1
accuracies = []

for train_index, test_index in kf.split(x, y):
    model = DecisionTreeClassifier()
    model.fit(x[train_index],y[train_index])
    score = model.score(x[test_index], y[test_index])
    print("Fold: {} Accuraccy: {}".format(n_fold, score))
    accuracies.append(score)

    dot_data = tree.export_graphviz(model, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("{}-dt-result".format(n_fold))
    n_fold += 1

with open('dt-results.csv', 'w', encoding='UTF-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    fold = 1
    header = ['fold', 'precision']
    writer.writerow(header)
    for accuracy in accuracies:
        row = [fold, accuracy]
        writer.writerow(row)
        fold += 1
    avgs = ["Average", np.mean(np.array(accuracies))]
    writer.writerow(avgs)