import itertools
import csv

with open('test_islice.csv', 'r') as csv_file:
    datareader = csv.reader(csv_file)
    #next(datareader)    # skip header row
    # counter = 0
    for row in itertools.islice(datareader, 0, None):
        while True:
            #next_n_lines = list(itertools.islice(csv_file, 100))
            next_n_lines = [x.rstrip('\n') for x in itertools.islice(csv_file, 100)]
            if not next_n_lines:
                break
            # process next_n_lines
            print(next_n_lines)
            print("\n Separate Groups \n")