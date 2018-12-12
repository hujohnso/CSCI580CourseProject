import numpy as np
import os
import time

start = time.clock()
arr = []
with open(os.path.join('HeatEqnData', str(1000) + '.csv')) as f:
    reader = f.readlines()
    timestamp = []
    for row in reader:
        split = row.split(',')
        for s in split:
            timestamp.append(float(s))
    arr.append(timestamp)

arr = np.array(arr)
print("Total time for IO is: ", time.clock() - start)

