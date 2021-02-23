import pickle

y_train_data = pickle.load(open("y_train.pickle", "rb"))
y_test_data = pickle.load(open("y_test.pickle", "rb"))

y = []

for yo in y_train_data:
    y.append(yo)

for yo in y_test_data:
    y.append(yo)

Y = []
disponibilidad = 0
tolerancia = 0
mantenibilidad = 0
rendimiento = 0
escalabilidad = 0
seguridad = 0
usabilidad = 0

for o in y:
    if o == 0:
        disponibilidad += 1
    if o == 1:
        tolerancia += 1
    if o == 2:
        mantenibilidad += 1
    if o == 3:
        rendimiento += 1
    if o == 4:
        escalabilidad += 1
    if o == 5:
        seguridad += 1
    if o == 6:
        usabilidad += 1

print("disponibilidad:", disponibilidad)
print("tolerancia: ", tolerancia)
print("mantenibilidad:", mantenibilidad)
print("rendimiento:", rendimiento)
print("escalabilidad:", escalabilidad)
print("seguridad:", seguridad)
print("usabilidad:", usabilidad)