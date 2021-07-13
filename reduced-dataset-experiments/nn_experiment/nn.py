from matplotlib import pyplot as plt
import torch
from torch.functional import split
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import pickle
import os
import scikitplot as skplt
import csv  

x_datasets = ['CFS-x.pickle', 'FCBF-x.pickle', 
            'MRMR-x.pickle', 'ReliefF-x.pickle']

y_datasets = ['CFS-y.pickle', 'FCBF-y.pickle', 
            'MRMR-y.pickle', 'ReliefF-y.pickle']

for x_dataset, y_dataset in zip(x_datasets, y_datasets):

    dirname = x_dataset.split('-')
    dirname = dirname[0]

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # Import datasets

    x_data = pickle.load(open("../datasets/" + x_dataset, 'rb'))
    y_data = pickle.load(open("../datasets/" + y_dataset, 'rb'))

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    y = []

    for o in y_data:
        if o == 0:
            y.append([1, 0, 0, 0, 0, 0, 0])
        if o == 1:
            y.append([0, 1, 0, 0, 0, 0, 0])
        if o == 2:
            y.append([0, 0, 1, 0, 0, 0, 0])
        if o == 3:
            y.append([0, 0, 0, 1, 0, 0, 0])
        if o == 4:
            y.append([0, 0, 0, 0, 1, 0, 0])
        if o == 5:
            y.append([0, 0, 0, 0, 0, 1, 0])
        if o == 6:
            y.append([0, 0, 0, 0, 0, 0, 1])

    x = torch.from_numpy(x_data)
    y = np.array(y)
    y = torch.from_numpy(y)

    criterion = nn.MSELoss()
    epochs = 5000
    kf = StratifiedKFold(n_splits=10)
    lrate = 0.0006

    accuracies_train = []
    accuracies_test = []
    balanced_accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    # Validation in k folds

    n_fold = 1

    print("Classifying ", dirname, " dataset")

    for train_index, test_index in kf.split(x, y_data):
        
        model = nn.Sequential(nn.Linear(len(x[1]), 25), nn.Tanh(), nn.Linear(25, 7), nn.Tanh())

        # Stochastic Grdient Descent optimizer
        optimizer = optim.SGD(model.parameters(), lr=lrate)

        accuracy_train_epochs = []
        accuracy_test_epochs = []
        balanced_accuracy_epochs = []
        precision_epochs = []
        recall_epochs = []
        f1_epochs = []

        for e in range(epochs):
            
            # Neural network training

            y_true_train = []
            y_clas_train = []

            for xi, yo in zip(x[train_index], y[train_index]):
                optimizer.zero_grad()
                output = model(xi.float())
                y_true_train.append(torch.argmax(yo).numpy())
                y_clas_train.append(torch.argmax(output).numpy())
                loss = criterion(output, yo.float())
                loss.backward()
                optimizer.step()

            # Neural network test

            y_true = []
            y_clas = []

            for xi, yo in zip(x[test_index], y[test_index]):
                output = model(xi.float())
                y_true.append(torch.argmax(yo).numpy())
                y_clas.append(torch.argmax(output).numpy())


            # Saving confusion matrix

            if (e == epochs-1):
                skplt.metrics.plot_confusion_matrix(y_true, y_clas, title="Fold " + str(n_fold) + " Confusion Matrix" , normalize=True, text_fontsize="small")
                plt.savefig(os.path.join(dirname, "confusion-matrix" + str(n_fold) + ".png"))
                plt.clf()
                
            # Metrics calculation per epoch

            accuracy_train_epochs.append(accuracy_score(y_true_train, y_clas_train))
            accuracy_test_epochs.append(accuracy_score(y_true, y_clas))
            balanced_accuracy_epochs.append(balanced_accuracy_score(y_true, y_clas))
            precision_epochs.append(precision_score(y_true, y_clas, average='macro', zero_division=0))
            recall_epochs.append(recall_score(y_true, y_clas, average='macro', zero_division=0))
            f1_epochs.append(f1_score(y_true, y_clas, average='macro', zero_division=0))

        # Metrics calculation per fold

        accuracies_train.append(accuracy_train_epochs[-1])
        accuracies_test.append(accuracy_test_epochs[-1])
        balanced_accuracies.append(balanced_accuracy_epochs[-1])
        precisions.append(precision_epochs[-1])
        recalls.append(recall_epochs[-1])
        f1_scores.append(f1_epochs[-1])

        # Saving training and test accuracies plot

        plt.title("Training and test accuracy in fold " + str(n_fold))
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.plot(range(epochs), accuracy_train_epochs, label='Train accuracy')
        plt.plot(range(epochs), accuracy_test_epochs, label='Test accuracy')
        plt.legend(title='Metrics')
        plt.savefig(os.path.join(dirname, "accuracies" + str(n_fold) + ".png"))
        plt.clf()
        plt.close()

        n_fold += 1
    
    # Saving experiment results in csv

    with open(dirname + "results.csv", 'w', encoding='UTF8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        fold = 1
        header = ['fold', 'accuracy train', 'accuracy test', 'balanced accuracy', 'precision', 'recall', 'f1_score']
        writer.writerow(header)
        for atrain, atest, ba, p, r, f1 in zip(accuracies_train,
                                            accuracies_test, 
                                            balanced_accuracies,
                                            precisions,
                                            recalls,
                                            f1_scores):
            row = [fold, atrain, atest, ba, p, r, f1]
            writer.writerow(row)
            fold += 1

        avgs = ["Average",
                np.mean(np.array(accuracies_train)),
                np.mean(np.array(accuracies_test)),
                np.mean(np.array(balanced_accuracies)),
                np.mean(np.array(precisions)),
                np.mean(np.array(recalls)),
                np.mean(np.array(f1_scores)) ]
        writer.writerow(avgs)