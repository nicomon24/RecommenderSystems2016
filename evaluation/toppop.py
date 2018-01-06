import time, math, sys
import numpy as np
import cPickle as pickle
import csv
from collections import defaultdict
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial

interactions_file = sys.argv[1]
targets_file = sys.argv[2]
output_file = sys.argv[3]

def intify(str):
    try:
        return int(str)
    except ValueError, ex:
        return int(-1)

def convertInter(inter):
    newInter = []
    newInter.append(intify(inter[0]))
    newInter.append(intify(inter[1]))
    newInter.append(intify(inter[2]))
    newInter.append(intify(inter[3]))
    return newInter

def convertTarget(target):
    newTarget = []
    newTarget.append(intify(target[0]))
    return newTarget

with open(interactions_file, 'rb') as f:
    reader = csv.reader(f, delimiter='\t')
    interactions_raw = list(reader)[1:]

with open(targets_file, 'rb') as f:
    reader = csv.reader(f, delimiter='\t')
    targets_raw = list(reader)[1:]

interactions = []
for i in interactions_raw:
    interactions.append(convertInter(i))

targets = []
for i in targets_raw:
    targets.append(convertTarget(i)[0])

top = defaultdict(int)
for interaction in interactions:
    top[interaction[1]] += 1

order = sorted(top, key=top.get,reverse = True)[:5]

f = open(output_file, 'w')
f.write("user_id,recommended_items\n")
for t in targets:
    f.write(str(t) + "," + " ".join(map(str, order[:5])) + "\n")

print "Written file " + output_file
