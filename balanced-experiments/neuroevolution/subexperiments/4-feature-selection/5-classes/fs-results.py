import csv
import re

def main():
    file = open('5-classes-results-9', 'r')
    lines = file.readlines()

    del lines[0:9]
    
    data = []
    
    for line in lines:
        if(line.find('True') > -1):
            row = re.findall(r'\d+', line)
            row.pop()
            row.pop()
            data.append(row)
    
    with open('fs-results-5-classes-9.csv', 'w', encoding='UTF8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['characteristic', 'node']
        writer.writerow(header)
        for row in data:
            writer.writerow(row)
    

if __name__=='__main__':
    main()