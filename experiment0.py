import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import pickle
import os

if not os.path.exists('plots'):
    os.makedirs('plots')

# Extract data from binaries
x_train_data = pickle.load(open("X_train.pickle", "rb"))
x_test_data = pickle.load(open("X_test.pickle", "rb"))
y_train_data = pickle.load(open("y_train.pickle", "rb"))
y_test_data = pickle.load(open("y_test.pickle", "rb"))

# Transform ndarray to array
X = []
y = []

for x in x_train_data:
    X.append(x)

for x in x_test_data:
    X.append(x)

for yo in y_train_data:
    y.append(yo)

for yo in y_test_data:
    y.append(yo)

# Array to numpy array
X = np.array(X)
y = np.array(y)

# Expected output
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

# Numpy array to tensor
X = torch.from_numpy(X)
Y = np.array(Y)
Y = torch.from_numpy(Y)

# There are 630 records to classify into 7 classes, every record has 148 neurons
model = nn.Sequential(nn.Linear(148, 147), nn.Tanh(), nn.Linear(147, 7), nn.Tanh())

criterion = nn.MSELoss()

# Stochastic Grdient Descent optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001)

epochs = 1000

# Device configuration, it changes to cuda device if it is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
X = X.to(device)
Y = Y.to(device)
model.to(device)

# Define the number of folds to cross-validate
kf = KFold(n_splits=10)

n_fold = 0
max_accuracy_train = []
max_accuracy_test = []

for train_index, test_index in kf.split(X):
    error_epochs = []
    accuracy_train_epochs = []
    accuracy_test_epochs = []
    for e in range(epochs):
        running_loss = 0
        num_asserts = 0

        for xi, yo in zip(X, Y):
            optimizer.zero_grad()
            output = model(xi.float())
            loss = criterion(output, yo.float())
            loss.backward()
            optimizer.step()

            if torch.argmax(output) == torch.argmax(yo):
                num_asserts += 1
                
            running_loss += loss.item()

        training_loss = running_loss/len(xi)
        # print("Training loss:", training_loss)
        error_epochs.append(training_loss)

        test_num_asserts = 0

        for xi, yo in zip(X[test_index], Y[test_index]):
            output = model(xi.float())

            if torch.argmax(output) == torch.argmax(yo):
                test_num_asserts += 1
        
        accuracy_train = num_asserts/567
        accuracy_train_epochs.append(accuracy_train)
        accuracy_test = test_num_asserts/63
        accuracy_test_epochs.append(accuracy_test)
        print("Epoch:", e, "Accuracy train:", accuracy_train, "Accuracy test:", accuracy_test)
    
    max_accuracy_train.append(max(accuracy_train_epochs))
    max_accuracy_test.append(max(accuracy_test_epochs))

    n_fold += 1
    filename = ""
    dir_name = 'plots'

    # Saving training loss plot
    plt.plot(range(epochs), error_epochs)
    title = "Trainig loss in fold " + str(n_fold)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Training loss')
    filename = "loss" + str(n_fold) + ".png"
    plt.savefig(os.path.join(dir_name, filename))
    plt.clf()

    # Saving training accuracy plot
    plt.plot(range(epochs), accuracy_train_epochs)
    title = "Training accuracy in fold " + str(n_fold)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    filename = "training_accuracy" + str(n_fold) + ".png"
    plt.savefig(os.path.join(dir_name, filename))
    plt.clf()

    # Saving test accuracy plot
    plt.plot(range(epochs), accuracy_test_epochs)
    title = "Test accuracy in fold " + str(n_fold)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    filename = "test_accuracy" + str(n_fold) + ".png"
    plt.savefig(os.path.join(dir_name, filename))
    plt.clf()