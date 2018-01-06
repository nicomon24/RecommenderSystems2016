
#IMPORT
import time, math, random
import numpy as np
import cPickle as pickle
import csv
from collections import defaultdict
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial

# -------- TUNING -----------
MAX_TS = 1447023571 #(Sun, 08 Nov 2015 22:59:31 GMT)
MIN_TS = 1440021654 #(Wed, 19 Aug 2015 22:00:54 GMT)
PIVOT_TS = 1445904000 #(Sat, 27 Oct 2015 00:00:00 GMT)

CLUSTER1_SIZE = 1100
CLUSTER2_SIZE = 2200
CLUSTER3_SIZE = 1700

# ---------------------------

# ---------- NOTES ----------
''''
    SINGLE:
    0-1: 2221 (22%)  (1.1k on 5k)
    2-10: 4310 (43%) (2.2k on 5k)
    11-N: 3469 (35%) (1.7k on 5k)
    MULTIPLE:
    0-1: 2067
    2-10: 3723
    11-N: 4210
'''
# ---------------------------

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

with open('../Dataset/interactions.csv', 'rb') as f:
    reader = csv.reader(f, delimiter='\t')
    interactions_raw = list(reader)[1:]

with open('../Dataset/target_users.csv', 'rb') as f:
    reader = csv.reader(f, delimiter='\t')
    targets_raw = list(reader)[1:]

interactions = []
for i in interactions_raw:
    interactions.append(convertInter(i))

targets = []
for i in targets_raw:
    targets.append(convertTarget(i)[0])

# Build indexes

def user_item_index(temp_interactions):
    u_i = defaultdict(lambda : defaultdict(int))
    for interact in temp_interactions:
        u_i[interact[0]][interact[1]] += 1
    return u_i

def item_user_index(temp_interactions):
    i_u = defaultdict(lambda : defaultdict(int))
    for interact in temp_interactions:
        i_u[interact[1]][interact[0]] += 1
    return i_u

user_item = user_item_index(interactions)
item_user = item_user_index(interactions)

def analyze_frequencies(temp_user_item, temp_targets):
    scount1 = 0 #0-1 interactions
    scount2 = 0 #2-10 interactions
    scount3 = 0 #11-N interactions
    mcount1 = 0 #0-1 interactions
    mcount2 = 0 #2-10 interactions
    mcount3 = 0 #11-N interactions
    for t in temp_targets:
        single = len(temp_user_item[t].keys())
        multiple = sum(x for x in temp_user_item[t].values())
        if single < 2:
            scount1 += 1
        elif single < 11:
            scount2 += 1
        else:
            scount3 += 1
        if multiple < 2:
            mcount1 += 1
        elif multiple < 11:
            mcount2 += 1
        else:
            mcount3 += 1
    # Print results
    print "SINGLE: "
    print scount1, scount2, scount3
    print "MULTIPLE: "
    print mcount1, mcount2, mcount3

# Dataset splitting
interactions_train = []
interactions_test = []
interactions_test_temp = []
for interaction in interactions:
    if interaction[3] < PIVOT_TS:
        interactions_train.append(interaction)
    else:
        interactions_test_temp.append(interaction)

# Move users with less then 5 interactions from the test set to the train set
user_item_test_temp = user_item_index(interactions_test_temp)
for interaction in interactions_test_temp:
    if len(user_item_test_temp[interaction[0]].keys()) < 5:
        interactions_train.append(interaction)
    else:
        interactions_test.append(interaction)

# Extract users inside test set
user_item_train = user_item_index(interactions_train)
user_item_test = user_item_index(interactions_test)

#analyze_frequencies(user_item_train, user_item_test.keys())

cluster1 = []
cluster2 = []
cluster3 = []

for user in user_item_test:
    single = len(user_item_train[user].keys())
    if single < 2:
        cluster1.append(user)
    elif single < 11:
        cluster2.append(user)
    else:
        cluster3.append(user)

random.shuffle(cluster1)
random.shuffle(cluster2)
random.shuffle(cluster3)

targets_final = []
targets_final.extend(cluster1[:CLUSTER1_SIZE])
targets_final.extend(cluster2[:CLUSTER2_SIZE])
targets_final.extend(cluster3[:CLUSTER3_SIZE])

interactions_train_final = []
interactions_test_final = []
interactions_train_final.extend(interactions_train)

count = 0
for interaction in interactions_test:
    count += 1
    if count % 10000 == 0:
        print count
    # Add and condition to discard already seen items
    if interaction[0] in targets_final and user_item_train[interaction[0]][interaction[1]] == 0:
        interactions_test_final.append(interaction)
    else:
        interactions_train_final.append(interaction)

print "Extracted " + str(len(targets_final))
print "Train set size: " + str(len(interactions_train_final))
print "Test set size: " + str(len(interactions_test_final))

# Validation set dictionary
validation = defaultdict(set)
for interaction in interactions_test_final:
    validation[interaction[0]].add(interaction[1])
pickle.dump(validation, open("validation.fini", 'wb'))

# Print CSVs
inter = open('interactions_train.csv', 'w')
inter.write("user_id item_id interaction_type created_at\n")
for interaction in interactions_train_final:
    inter.write("\t".join(map(str, interaction)) + "\n")

targf = open('target_test.csv', 'w')
targf.write("user_id\n")
for target in targets_final:
    targf.write(str(target) + "\n")
