import numpy as np
import pickle
from numpy.core.fromnumeric import shape

from numpy.lib import delete

x_train = pickle.load(open('../../X_train.pickle', 'rb'))
x_test = pickle.load(open("../../X_test.pickle", "rb"))
y_train = pickle.load(open("../../y_train.pickle", "rb"))
y_test = pickle.load(open("../../y_test.pickle", "rb"))

x = []
y = []

for xi in x_train:
    x.append(xi)

for xi in x_test:
    x.append(xi)

for yo in y_train:
    y.append(yo)

for yo in y_test:
    y.append(yo)

x = np.array(x)
y = np.array(y)

imbalanced_data = [1, 4, 2, 0]

for imb_data in imbalanced_data:

    index = 0
    list_index = []

    for index in range(len(y)):

        if (y[index] == imb_data):
            list_index.append(index)

    y = np.delete(y, list_index, None)
    x = np.delete(x, list_index, 0)

    print(shape(x))
    print(shape(y))

    with open('x-no-' + str(imb_data), 'wb') as f:
        pickle.dump(x, f)
    
    with open('y-no-' + str(imb_data), 'wb') as f:
        pickle.dump(y, f)