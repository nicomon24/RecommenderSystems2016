"Just a bunch of functions to build indexes and save them"


import cPickle as pickle
import csv
import time
import numpy as np
from collections import defaultdict
from functools import partial

#First load all datasets and convert them
with open('./Dataset/interactions.csv', 'rb') as f:
    reader = csv.reader(f, delimiter='\t')
    interactions = list(reader)[1:]
with open('./Dataset/item_profile.csv', 'rb') as f:
    reader = csv.reader(f, delimiter='\t')
    items = list(reader)
    item_headers = items[0]
    items = items[1:]
with open('./Dataset/user_profile.csv', 'rb') as f:
    reader = csv.reader(f, delimiter='\t')
    users = list(reader)[1:]
with open('./Dataset/target_users.csv', 'rb') as f:
    reader = csv.reader(f, delimiter='\t')
    targets = list(reader)[1:]

def intify(str):
    try:
        return int(str)
    except ValueError, ex:
        return int(-1)

def floatify(str):
    try:
        return float(str)
    except ValueError, ex:
        return float(-1)

def convertItem(item):
    newItem = []
    newItem.append(intify(item[0]))
    newItem.append(item[1].split(","))
    for x in range(0,len(newItem[1])):
        newItem[1][x] = intify(newItem[1][x])
    newItem.append(intify(item[2]))
    newItem.append(intify(item[3]))
    newItem.append(intify(item[4]))
    if item[5] == 'de':
        newItem.append(1)
    elif item[5] == 'at':
        newItem.append(2)
    elif item[5] == 'ch':
        newItem.append(3)
    elif item[5] == 'non_dach':
        newItem.append(4)
    else:
        newItem.append(0)
    newItem.append(intify(item[6]))
    newItem.append(floatify(item[7]))
    newItem.append(floatify(item[8]))
    newItem.append(intify(item[9]))
    newItem.append(item[10].split(","))
    for x in range(0,len(newItem[10])):
        newItem[10][x] = intify(newItem[10][x])
    newItem.append(intify(item[11]))
    newItem.append(intify(item[12]))
    return newItem

def convertInteraction(interact):
    newInteract = []
    newInteract.append(intify(interact[0]))
    newInteract.append(intify(interact[1]))
    newInteract.append(intify(interact[2]))
    newInteract.append(intify(interact[3]))
    return newInteract

def convertUser(user):
    newUser = []
    newUser.append(intify(user[0]))
    newUser.append(user[1].split(","))
    for x in range(0,len(newUser[1])):
        newUser[1][x] = intify(newUser[1][x])
    newUser.append(intify(user[2]))
    newUser.append(intify(user[3]))
    newUser.append(intify(user[4]))
    if user[5] == 'de':
        newUser.append(1)
    elif user[5] == 'at':
        newUser.append(2)
    elif user[5] == 'ch':
        newUser.append(3)
    elif user[5] == 'non_dach':
        newUser.append(4)
    else:
        newUser.append(0)
    newUser.append(intify(user[6]))
    newUser.append(intify(user[7]))
    newUser.append(intify(user[8]))
    newUser.append(intify(user[9]))
    newUser.append(intify(user[10]))
    newUser.append(user[11].split(","))
    for x in range(0,len(newUser[11])):
        newUser[11][x] = intify(newUser[11][x])
    return newUser

def convertTarget(target):
    newTarget = intify(target[0])
    return newTarget

for i in range(0,len(items)):
    items[i] = convertItem(items[i])

for i in range(0,len(interactions)):
    interactions[i] = convertInteraction(interactions[i])

for i in range(0,len(users)):
    users[i] = convertUser(users[i])

for i in range(0,len(targets)):
    targets[i] = convertTarget(targets[i])

print "Loaded datasets"

#Done loading and converting, now build the indexes we want
#Item index { "iid" : {rows}}
item_idx_file = "indexes/item_idx.pkl"
item_active_idx_file = "indexes/item_active_idx.pkl"
def item_idx(filename):
    index = {}
    for item in items:
        index[item[0]] = item
    pickle.dump(index, open(filename, 'wb'))
    return index

#User index { "uid" : {rows}}
user_idx_file = "indexes/user_idx.pkl"
def user_idx():
    index = {}
    for user in users:
        index[user[0]] = user
    pickle.dump(index, open(user_idx_file, 'wb'))
    return index

#Interactions by user ( "uid" : [item, item, ...])
interaction_by_user_idx_file = "indexes/interaction_by_user_idx.pkl"
interaction_active_by_user_idx_file = "indexes/interaction_active_by_user_idx.pkl"
def interaction_by_user_idx(filename):
    index = {}
    start = time.time()
    for i in range(0, len(interactions)):
        interaction = interactions[i]
        if not interaction[0] in index.keys():
            index[interaction[0]] = set()
        index[interaction[0]].add(interaction[1])
        if i % 10000 == 1:
            rem = ((time.time() - start) / float(i)) * (len(interactions) - i)
            print "Remaining time: " + str(rem) + "s [" + str(i) + "/" + str(len(interactions)) + "]"
    pickle.dump(index, open(filename, 'wb'))
    return index

#Interactions by user ( "iid" : [uid, uid, ...])
interaction_by_item_idx_file = "indexes/interaction_by_item_idx.pkl"
interaction_active_by_item_idx_file = "indexes/interaction_active_by_item_idx.pkl"
interaction_tuple_by_item_idx_file = "indexes/interaction_tuple_by_item_idx.pkl"
def interaction_by_item_idx(filename):
    index = {}
    start = time.time()
    for i in range(0, len(interactions)):
        interaction = interactions[i]
        if not interaction[1] in index.keys():
            index[interaction[1]] = set()
        index[interaction[1]].add(interaction[0])
        if i % 10000 == 1:
            rem = ((time.time() - start) / float(i)) * (len(interactions) - i)
            print "Remaining time: " + str(rem) + "s [" + str(i) + "/" + str(len(interactions)) + "]"
    pickle.dump(index, open(filename, 'wb'))
    return index

def interaction_tuple_by_item_idx():
    kweights = [0.5, 0.8, 1.0, 1.2]
    total = {}
    index = {}
    #Compute user total interactions
    for i in range(0,len(interactions)):
        interaction = interactions[i]
        if interaction[0] not in total.keys():
            total[interaction[0]] = [0,0,0]
        total[interaction[0]][interaction[2]-1] += 1
    print "End users"
    #Compute item interactions
    for i in range(0, len(interactions)):
        interaction = interactions[i]
        if interaction[1] not in index.keys():
            index[interaction[1]] = dict()
        if interaction[0] not in index[interaction[1]].keys():
            index[interaction[1]][interaction[0]] = [0,0,0]
        index[interaction[1]][interaction[0]][interaction[2]-1] += 1
    print "End items"
    #Now mix the 2 things
    for k in index.keys():
        for y in index[k].keys():
            #k is the item, y is the user
            t = total[y]
            index[k][y] = (kweights[1] * index[k][y][0] + kweights[2] * index[k][y][1] + kweights[3] * index[k][y][2]) / float(kweights[0] + t[0] * kweights[1] + t[1] * kweights[2] + t[2] * kweights[3])
    print "End all"
    pickle.dump(index, open("indexes/interaction_tuple_by_item_idx.pkl", 'wb'))
    return index

#Item feature clustering: array of dictionary per feature. List of features:
    #1: title <array>
    #2: career_level <int>
    #3: discipline <int>
    #4: industry <int>
    #5: country <int>
    #6: region <int>
    #7: latitude <float>        [NOT-INDEXED]
    #8: longitude <float>       [NOT-INDEXED]
    #9: employment <int>
    #10: tags <array>
    #11: createdAt <ts>         [NOT-INDEXED]
    #12: active <int>           [NOT-INDEXED]
item_feature_idx_file = "indexes/item_feature_idx.pkl"
def item_feature_idx():
    featureindex = [{},{},{},{},{},{},{},{},{},{},{}]
    start = time.time()
    for i in range(0,len(items)):
        for x in [1,2,3,4,5,6,9,10]:
            if isinstance(items[i][x], int) or isinstance(items[i][x], float):
                if not items[i][x] in featureindex[x].keys():
                    featureindex[x][items[i][x]] = set()
                featureindex[x][items[i][x]].add(items[i][0])
            else:
                for value in items[i][x]:
                    if not value in featureindex[x].keys():
                        featureindex[x][value] = set()
                    featureindex[x][value].add(items[i][0])
        if i % 10000 == 1:
            rem = ((time.time() - start) / float(i)) * (len(items) - i)
            print "Remaining time: " + str(rem) + "s [" + str(i) + "/" + str(len(items)) + "]"
    pickle.dump(featureindex, open(item_feature_idx_file, 'wb'))
    return featureindex

#User feature clustering: array of dictionary per feature. List of features:
    #1: jobroles <array>
    #2: career_level <int>
    #3: discipline <int>
    #4: industry <int>
    #5: country <int>
    #6: region <int>
    #7: experience_n_entries <int>
    #8: experience_years <int>
    #9: experience_years_current <int>
    #10: edu_degree <int>
    #11: edu_fields <array>
user_feature_idx_file = "indexes/user_feature_idx.pkl"
def user_feature_idx():
    featureindex = [{},{},{},{},{},{},{},{},{},{},{},{}]
    start = time.time()
    for i in range(0,len(users)):
        for x in [1,2,3,4,5,6,7,8,9,10,11]:
            if isinstance(users[i][x], int) or isinstance(users[i][x], float):
                if not users[i][x] in featureindex[x].keys():
                    featureindex[x][users[i][x]] = set()
                featureindex[x][users[i][x]].add(users[i][0])
            else:
                for value in users[i][x]:
                    if not value in featureindex[x].keys():
                        featureindex[x][value] = set()
                    featureindex[x][value].add(users[i][0])
        if i % 10000 == 1:
            rem = ((time.time() - start) / float(i)) * (len(users) - i)
            print "Remaining time: " + str(rem) + "s [" + str(i) + "/" + str(len(users)) + "]"
    pickle.dump(featureindex, open(user_feature_idx_file, 'wb'))
    return featureindex

#Now compute for real
#Activeness of items enabling or disabling
itemidx = pickle.load( open("indexes/item_idx.pkl", 'rb'))
print "Starting"
print interaction_tuple_by_item_idx()

items = [x for x in items if x[12] == 1]
interactions = [x for x in interactions if itemidx[x[1]][12]==1]

# ----- HOWTO ------
# To load a single index file do this
# index = pickle.load( open(filename, 'rb'))
