import pickle

x_train_data = pickle.load(open("X_train.pickle", "rb"))
y_train_data = pickle.load(open("y_train.pickle", "rb"))
y_test_data = pickle.load(open("y_test.pickle", "rb"))

y = []

for yo in y_train_data:
    y.append(yo)

for yo in y_test_data:
    y.append(yo)

Y = []
availability = 0
tolerance = 0
maintainability = 0
performance = 0
scalability = 0
security = 0
usability = 0

for o in y:
    if o == 0:
        availability += 1
    if o == 1:
        tolerance += 1
    if o == 2:
        maintainability += 1
    if o == 3:
        performance += 1
    if o == 4:
        scalability += 1
    if o == 5:
        security += 1
    if o == 6:
        usability += 1

print("0:", availability)
print("1: ", tolerance)
print("2:", maintainability)
print("3:", performance)
print("4:", scalability)
print("5:", security)
print("6:", usability)