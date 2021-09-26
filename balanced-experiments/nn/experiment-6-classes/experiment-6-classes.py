# Experiment without fault tolerance class, six classes.

import csv
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import scikitplot as sklpt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

x = pickle.load(open('../../datasets/x-no-1', 'rb'))
y = pickle.load(open('../../datasets/y-no-1', 'rb'))

Y = []
for o in y:
    if o == 0:
        Y.append([1, 0, 0, 0, 0, 0, 0])
    if o == 1:
        Y.append([0, 1, 0, 0, 0, 0, 0])
    if o == 2:
        Y.append([0, 0, 1, 0, 0, 0, 0])
    if o == 3:
        Y.append([0, 0, 0, 1, 0, 0, 0])
    if o == 4:
        Y.append([0, 0, 0, 0, 1, 0, 0])
    if o == 5:
        Y.append([0, 0, 0, 0, 0, 1, 0])
    if o == 6:
        Y.append([0, 0, 0, 0, 0, 0, 1])

x = torch.from_numpy(x)
Y = np.array(Y)
Y = torch.from_numpy(Y)

criterion = nn.MSELoss()
epochs = 5000

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x = x.to(device)
Y = Y.to(device)

# Defining folds
kf = StratifiedKFold(n_splits=10)
lrate = 0.0006

accuracies_train = []
accuracies_test = []
precisions = []
recalls = []
f1_scores = []

n_fold = 1

for train_index, test_index in kf.split(x, y):
    model = nn.Sequential(nn.Linear(148, 25), nn.Tanh(), nn.Linear(25, 7), nn.Tanh())
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lrate)

    accuracy_train_epochs = []
    accuracy_test_epochs = []
    precision_epochs = []
    recall_epochs = []
    f1_epochs = []

    for e in range(epochs):

        y_true_train = []
        y_clas_train = []

        for xi, yo in zip(x[train_index], Y[train_index]):
            optimizer.zero_grad()
            output = model(xi.float())
            y_true_train.append(torch.argmax(yo).cpu().detach().numpy())
            y_clas_train.append(torch.argmax(output).cpu().detach().numpy())
            loss = criterion(output, yo.float())
            loss.backward()
            optimizer.step()
        
        y_true = []
        y_clas = []
        
        for xi, yo in zip(x[test_index], Y[test_index]):
            output = model(xi.float())
            y_true.append(torch.argmax(yo).cpu().detach().numpy())
            y_clas.append(torch.argmax(output).cpu().detach().numpy())
        
        if (e == epochs-1):
            sklpt.metrics.plot_confusion_matrix(y_true, y_clas, 
                title= "Six Classes Confusion Matrix: Fold " + str(n_fold), 
                normalize=True, text_fontsize='small')
            plt.savefig('cf-6-classes-' + str(n_fold) + '.png')
            plt.clf()
            plt.close()
        
        # Metrics in this fold

        accuracy_train_epochs.append(accuracy_score(y_true_train, y_clas_train))
        accuracy_test_epochs.append(accuracy_score(y_true, y_clas))
        precision_epochs.append(precision_score(y_true, y_clas, average='macro', zero_division=0))
        recall_epochs.append(recall_score(y_true, y_clas, average='macro', zero_division=0))
        f1_epochs.append(f1_score(y_true, y_clas, average='macro', zero_division=0))

        print("Fold {}, Epoch {}, Accuracy train {}, Accuracy test {}, Precision {}, Recall {}, F1 {}".format(
            n_fold,
            e, 
            accuracy_train_epochs[-1], 
            accuracy_test_epochs[-1], 
            precision_epochs[-1], 
            recall_epochs[-1],
            f1_epochs[-1]))


    plt.title('Convergence Plot in fold ' + str(n_fold))
    plt.xlabel('Epochs')
    plt.ylabel('Values')
    plt.plot(range(epochs), accuracy_train_epochs, label='Train Accuracy')
    plt.plot(range(epochs), accuracy_test_epochs, label=" Test Accuracy")
    plt.plot(range(epochs), precision_epochs, label="Test Precision")
    plt.plot(range(epochs), recall_epochs, label='Test Recall')
    plt.plot(range(epochs), f1_epochs, label='Test F1-score')
    plt.legend(title='Metrics')
    plt.savefig("convergence-" + str(n_fold))
    plt.clf()
    plt.close()

    accuracies_train.append(accuracy_train_epochs[-1])
    accuracies_test.append(accuracy_test_epochs[-1])
    precisions.append(precision_epochs[-1])
    recalls.append(recall_epochs[-1])
    f1_scores.append(f1_epochs[-1])

    n_fold = n_fold + 1

with open("results-6-classes.csv", 'w', encoding='UTF8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    fold = 1
    header = ['fold', 'accuracy train', 'accuracy test', 'precision', 'recall', 'f1_score']
    writer.writerow(header)
    for atrain, atest, p, r, f1 in zip(accuracies_train,
                                        accuracies_test, 
                                        precisions,
                                        recalls,
                                        f1_scores):
        row = [fold, atrain, atest, p, r, f1]
        writer.writerow(row)
        fold += 1

    avgs = ["Average",
            np.mean(np.array(accuracies_train)),
            np.mean(np.array(accuracies_test)),
            np.mean(np.array(precisions)),
            np.mean(np.array(recalls)),
            np.mean(np.array(f1_scores)) ]
    writer.writerow(avgs)