import csv
import pickle

datasets = ['CFS-reduced-dataset.csv', 'FCBF-reduced-dataset.csv', 
            'MRMR-reduced-dataset.csv', 'ReliefF-reduced-dataset.csv']

for dataset in datasets:

    # Extracting data into lists

    x = []
    y = []

    with open(dataset) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            x.append(row[0:-1])
            y.append(row[-1])

    # Parsing string elements in x to float numbers in x_numeric

    x_numeric = []

    for xi in x:
        row = []
        for xj in xi:
            xj = float(xj)
            row.append(xj)
        x_numeric.append(row)

    # Parsing string elements in y to float numbers in y_numeric

    y_numeric = [] 

    for yi in y:
        if (yi == 'availability'):
            y_numeric.append(0)
        
        if (yi == 'fault tolerance'):
            y_numeric.append(1)
        
        if (yi == 'maintainability'):
            y_numeric.append(2)
        
        if (yi == 'performance'):
            y_numeric.append(3)

        if (yi == 'scalability'):
            y_numeric.append(4)
        
        if (yi == 'security'):
            y_numeric.append(5)

        if (yi == 'usability'):
            y_numeric.append(6)

    # Exporting numeric datasets to binaries

    dataset_namesplitted = dataset.split('-')
    preffix_name = dataset_namesplitted[0]

    with open(preffix_name + '-x.pickle', 'wb') as f:
        pickle.dump(x_numeric, f)

    with open(preffix_name + '-y.pickle', 'wb') as f:
        pickle.dump(y_numeric, f)