import csv
from posixpath import join
import numpy as np
import graphviz
import pickle
import scikitplot
import os
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
precisions = []
recalls = []
f1_scores = []

cm_folder = 'confusion-matrix'
if not os.path.exists(cm_folder):
    os.makedirs(cm_folder)

for train_index, test_index in kf.split(x, y):
    model = DecisionTreeClassifier()
    model.fit(x[train_index],y[train_index])
    y_pred = model.predict(x[test_index])

    accuracy = accuracy_score(y[test_index], y_pred)
    precision = precision_score(y[test_index], y_pred, average='macro', zero_division=0)
    recall = recall_score(y[test_index], y_pred, average='macro', zero_division=0)
    f1 = f1_score(y[test_index], y_pred, average='macro', zero_division=0)

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

    print("Fold: {} Accuraccy: {}".format(n_fold, accuracy))

    scikitplot.metrics.plot_confusion_matrix(y[test_index], 
                                            y_pred,
                                            title='Decision Tree Confusion Matrix', 
                                            normalize=True, text_fontsize='small')
    plt.savefig(join(cm_folder, "{}-cm".format(n_fold)))
    plt.clf()
    plt.close()
    
    dot_data = tree.export_graphviz(model, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("{}-dt-result".format(n_fold))
    n_fold += 1

with open('dt-results.csv', 'w', encoding='UTF-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    fold = 1
    header = ['fold', 'accuracy', 'precision', 'recall', 'f1_score']
    writer.writerow(header)
    for accuracy, precision, recall, f1 in zip(accuracies, precisions, recalls, f1_scores):
        row = [fold, accuracy, precision, recall, f1]
        writer.writerow(row)
        fold += 1
    avgs = ["Average", 
            np.mean(np.array(accuracies)),
            np.mean(np.array(precisions)),
            np.mean(np.array(recalls)),
            np.mean(np.array(f1_scores))]
    writer.writerow(avgs)