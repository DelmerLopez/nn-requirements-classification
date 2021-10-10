import csv
import re

def main():
    file = open('7-classes-results-8', 'r')
    lines = file.readlines()

    del lines[0:9]
    
    data = []
    
    for line in lines:
        if(line.find('True') > -1):
            row = re.findall(r'\d+', line)
            data.append(row[0:2])
    
    with open('fs-results-7-classes-8.csv', 'w', encoding='UTF8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['characteristic', 'node']
        writer.writerow(header)
        for row in data:
            writer.writerow(row)
    

if __name__=='__main__':
    main()