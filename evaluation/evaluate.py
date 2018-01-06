#IMPORT
import time, math, random, subprocess, sys
import numpy as np
import cPickle as pickle
import csv
from collections import defaultdict
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial

script = sys.argv[1]
interactions_file = "evaluation/interactions_train.csv"
targets_file = "evaluation/target_test.csv"
output = "evaluation/output" + str(time.time()).split('.')[0] + '.csv'
if len(sys.argv) > 2:
    output = sys.argv[2]
validation = pickle.load(open("evaluation/validation.fini", 'rb'))
if len(sys.argv) > 3:
    with open(sys.argv[3], 'rb') as f:
        reader = csv.reader(f, delimiter='\n')
        validation_raw = list(reader)[1:]
    validation = defaultdict(set)
    for v in validation_raw:
        entry = v[0].split(",")
        validation[int(entry[0])] = set(int(x) for x in entry[1].split(" "))

command = "python " + " ".join([script, interactions_file, targets_file, output])
# Run script
if script != "pass":
    print "Computing again"
    recommender = subprocess.Popen(command, shell = True)
    recommender.wait()
    print "Ended subprogram, evaluate"

# Evaluate results
with open(output, 'rb') as f:
    reader = csv.reader(f, delimiter='\n')
    output_raw = list(reader)[1:]

results = defaultdict(list)
for x in output_raw:
    entry = x[0].split(",")
    results[int(entry[0])] = map(int, entry[1].split(" "))

# Compute score
score = float(0)
for user in validation.keys():
    #print "User " + str(user)
    #print validation[user]
    #print results[user]
    minl = min(5, len(validation[user]))
    for i in range(1,minl+1):
        #print "Values at cutoff " + str(i)
        l = len(set.intersection(validation[user],set(results[user][:i]))) / float(i * minl)
        #print l
        score += l

fscore = score / float(len(validation.keys()))
print "Final score: "
print fscore
