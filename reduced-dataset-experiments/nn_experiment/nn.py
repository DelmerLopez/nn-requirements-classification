import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import pickle

x_datasets = ['CFS-x.pickle', 'FCBF-x.pickle', 
            'MRMR-x.pickle', 'ReliefF-x.pickle']

y_datasets = ['CFS-y.pickle', 'FCBF-y.pickle', 
            'MRMR-y.pickle', 'ReliefF-y.pickle']

for x_dataset, y_dataset in zip(x_datasets, y_datasets):

    # Import datasets

    x_data = pickle.load(open("../datasets/" + x))
    y_data = pickle.load(open("../datasets/" + y))

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
    epochs = 3000
    kf = StratifiedKFold(n_splits=10)
    lrate = 0.0006

    for train_index, test_index in kf.split(x, y):
        
        model = nn.Sequential(nn.Linear(148, 25), nn.Tanh(), nn.Linear(25, 7), nn.Tanh())

        # Stochastic Grdient Descent optimizer
        optimizer = optim.SGD(model.parameters(), lr=lrate)

        accuracy_train_fold = []
        accuracy_test_fold = []

        for e in range(epochs):

            train_asserts = 0
            train_accuracy = 0
            
            # Neural network training

            for xi, yo in zip(x[train_index], y[train_index]):
                optimizer.zero_grad()
                output = model(xi.float())
                loss = criterion(output, yo.float())
                loss.backward()
                optimizer.step()

                if torch.argmax(output) == torch.argmax(yo):
                    train_asserts += 1
            
            test_asserts = 0

            # Neural network test
            for xi, yo in zip(x[test_index], y[test_index]):
                output = model(xi.float())
                
                if torch.argmax(output) == torch.argmax(yo):
                    test_asserts += 1
            
        accuracy_train_fold.append(train_asserts / len(train_index))
        accuracy_test_fold.append(test_asserts / len(test_index))

    print(x_dataset)
    print(accuracy_train_fold)
    print(accuracy_train_fold)