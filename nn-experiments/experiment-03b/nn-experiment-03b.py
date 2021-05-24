import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import pickle
import os

# Extract data from binaries
x_train_data = pickle.load(open("../X_train.pickle", "rb"))
x_test_data = pickle.load(open("../X_test.pickle", "rb"))
y_train_data = pickle.load(open("../y_train.pickle", "rb"))
y_test_data = pickle.load(open("../y_test.pickle", "rb"))

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

criterion = nn.MSELoss()

epochs = 1500

# Device configuration, it changes to cuda device if it is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
X = X.to(device)
Y = Y.to(device)

# Define the number of folds to cross-validate
kf = StratifiedKFold(n_splits=10)

hidden_neurons = [200, 300, 400, 500]

for hn in hidden_neurons:

  n_fold = 0
  max_accuracy_train = []
  last_accuracy_train = []
  max_accuracy_test = []
  last_accuracy_test = []
  folds_precision = []
  folds_recall = []
  folds_f1 = []

  dir_name = 'plots' + str(hn)
  
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)

  for train_index, test_index in kf.split(X, y):

      # There are 630 records to classify into 7 classes, every record has 148 neurons
      model = nn.Sequential(nn.Linear(148, 25), nn.Tanh(), nn.Linear(25, hn), nn.Tanh(), nn.Linear(hn,7), nn.Tanh())
      model.to(device)

      # Stochastic Grdient Descent optimizer
      optimizer = optim.SGD(model.parameters(), lr=0.001)

      error_epochs = []
      accuracy_train_epochs = []
      accuracy_test_epochs = []
      precision_epochs = []
      recall_epochs = []
      f1_epochs = []
      for e in range(epochs):
          running_loss = 0
          num_asserts = 0
          true_positives = [0 , 0, 0, 0, 0, 0, 0]

          for xi, yo in zip(X[train_index], Y[train_index]):
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
          train_true = []
          train_results = []

          for xi, yo in zip(X[test_index], Y[test_index]):
              output = model(xi.float())
              train_true.append(torch.argmax(yo).cpu().detach().numpy())
              train_results.append(torch.argmax(output).cpu().detach().numpy())

              if torch.argmax(output) == torch.argmax(yo):
                  test_num_asserts += 1

          recall_epochs.append(recall_score(train_true, train_results, average='macro'))
          precision_epochs.append(precision_score(train_true, train_results, average='macro'))
          f1_epochs.append(f1_score(train_true, train_results, average='macro'))
          accuracy_train = 0
          accuracy_train = num_asserts/len(train_index)
          accuracy_train_epochs.append(accuracy_train)
          accuracy_test = 0
          accuracy_test = test_num_asserts/len(test_index)
          accuracy_test_epochs.append(accuracy_test)
          #print("\nEpoch:", e, "Accuracy train:", accuracy_train, "Accuracy test:", accuracy_test, "Precision:", precision_epochs[-1],  "Recall:", recall_epochs[-1], "f1-score:", f1_epochs[-1])
      
      max_accuracy_train.append(max(accuracy_train_epochs))
      last_accuracy_train.append(accuracy_train_epochs[-1])
      max_accuracy_test.append(max(accuracy_test_epochs))
      last_accuracy_test.append(accuracy_test_epochs[-1])
      folds_precision.append(precision_epochs[-1])
      folds_recall.append(recall_epochs[-1])
      folds_f1.append(f1_epochs[-1])

      n_fold += 1
      filename = ""

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
      title = "Training and test accuracy in fold " + str(n_fold)
      plt.title(title)
      plt.xlabel('Epochs')
      plt.ylabel('Accuracy')
      plt.plot(range(epochs), accuracy_train_epochs, label='Train accuracy')
      plt.plot(range(epochs), accuracy_test_epochs, label='Test accuracy')
      filename = "accuracy" + str(n_fold) + ".png"
      plt.savefig(os.path.join(dir_name, filename))
      plt.clf()

  print("Number of hidden neurons: ", hn)
  print("Max accuracy train: ")
  print(max_accuracy_train)
  print("Average:", sum(max_accuracy_train) / len(max_accuracy_train))
  print("Max accuracy test:")
  print(max_accuracy_test)
  print("Average:", sum(max_accuracy_test) / len(max_accuracy_test))
  print("Last accuracy train per epoch")
  print(last_accuracy_train)
  print("Average:", sum(last_accuracy_train) / len(last_accuracy_train))
  print("Last accuracy test per epoch")
  print(last_accuracy_test)
  print("Average:", sum(last_accuracy_test) / len(last_accuracy_test))
  print("Precision by fold:", folds_precision)
  print("Average:", sum(folds_precision) / len(folds_precision))
  print("Recall by fold:", folds_precision, "\nAverage:", sum(folds_recall) / len(folds_recall))
  print("f1_score by fold:", folds_f1, "\nAverage:", sum(folds_f1) / len(folds_recall))
  print("-----------------------------------------\n")