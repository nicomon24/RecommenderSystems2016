#IMPORT
import time, math, random, subprocess, sys
import numpy as np
import cPickle as pickle
import csv
from collections import defaultdict
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial

script = sys.argv[1]
INTERACTIONS_FILE = "evaluation/interactions_train.csv"
TARGETS_FILE = "evaluation/target_test.csv"
USERS_FILE = './Dataset/user_profile.csv'
ITEMS_FILE = './Dataset/item_profile.csv'
OUTPUT = "evaluation/output" + str(time.time()).split('.')[0] + '.csv'
if len(sys.argv) > 2:
    OUTPUT = sys.argv[2]
validation = pickle.load(open("evaluation/validation.fini", 'rb'))
if len(sys.argv) > 3:
    with open(sys.argv[3], 'rb') as f:
        reader = csv.reader(f, delimiter='\n')
        validation_raw = list(reader)[1:]
    validation = defaultdict(set)
    for v in validation_raw:
        entry = v[0].split(",")
        validation[int(entry[0])] = set(int(x) for x in entry[1].split(" "))

command = "python " + " ".join([script, INTERACTIONS_FILE, TARGETS_FILE, OUTPUT])
# Run script
if script != "pass":
    print "Computing again"
    recommender = subprocess.Popen(command, shell = True)
    recommender.wait()
    print "Ended subprogram, evaluate"

# Load datasets
with open(INTERACTIONS_FILE, 'rb') as f:
    reader = csv.reader(f, delimiter='\t')
    interactions_raw = list(reader)[1:]
with open(ITEMS_FILE, 'rb') as f:
    reader = csv.reader(f, delimiter='\t')
    items_raw = list(reader)[1:]
with open(USERS_FILE, 'rb') as f:
    reader = csv.reader(f, delimiter='\t')
    users_raw = list(reader)[1:]

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

def convertInter(inter):
    newInter = []
    newInter.append(intify(inter[0]))
    newInter.append(intify(inter[1]))
    newInter.append(intify(inter[2]))
    newInter.append(intify(inter[3]))
    return newInter

def convertUser(user):
    newUser = []
    newUser.append(intify(user[0]))
    newUser.append(user[1].split(","))
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
    #Check meaning
    if len(newUser[1]) == 1 and newUser[1][0] == 0:
        newUser[1] = []
    if len(newUser[11]) == 1 and newUser[11][0] == '':
        newUser[11] = []
    return newUser

def convertItem(item):
    newItem = []
    newItem.append(intify(item[0]))
    newItem.append(item[1].split(","))
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
    newItem.append(intify(item[11]))
    newItem.append(intify(item[12]))
    return newItem

interactions = []
for i in interactions_raw:
    interactions.append(convertInter(i))

items = defaultdict(list)
for i in items_raw:
    it = convertItem(i)
    items[it[0]] = it

users = defaultdict(list)
for u in users_raw:
    us = convertUser(u)
    users[us[0]] = us

#--------- INDEXING ------

# Build item user
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

def user_item_typed_index(temp_interactions):
    u_i = defaultdict(lambda : defaultdict(lambda : [0,0,0]))
    for interact in temp_interactions:
        u_i[interact[0]][interact[1]][interact[2]-1] += 1
    return u_i

def user_item_typing_index(temp_interactions):
    weights = [1, 2, 2.4]
    u_i = defaultdict(lambda : defaultdict(int))
    for interact in temp_interactions:
        u_i[interact[0]][interact[1]] += weights[interact[2]-1]
    return u_i

def user_item_last_index(temp_interactions):
    u_i = defaultdict(lambda : defaultdict(int))
    # Get the last timestamp for every user-item
    for interact in temp_interactions:
        if interact[3] > u_i[interact[0]][interact[1]]:
            u_i[interact[0]][interact[1]] = interact[3]
    # Find the last activity of the user
    for user in u_i:
        maxi = 0
        for item in u_i[user]:
            if u_i[user][item] > maxi:
                maxi = u_i[user][item]
        for item in u_i[user]:
            # Recompute difference in days
            u_i[user][item] = (maxi - u_i[user][item]) / (24 * 60 * 60)
    return u_i

user_item = user_item_index(interactions)
user_item_typed = user_item_typed_index(interactions)
user_item_typing = user_item_typing_index(interactions)
user_item_last = user_item_last_index(interactions)
item_user = item_user_index(interactions)

confidence_user = defaultdict(lambda : defaultdict(float))
def confidence_index(temp_user_item_typing):
    steeps = [0.2, 4, 4]
    temp_confidence = defaultdict(lambda : defaultdict(float))
    for user in temp_user_item_typing:
        total = sum([x for x in temp_user_item_typing[user].values()])
        mean = total / float(len(temp_user_item_typing[user].keys()))
        for item in temp_user_item_typing[user]:
            score = 1.0
            temp_confidence[item][user] = score
            confidence_user[user][item] = score
    return temp_confidence

confidence = confidence_index(user_item_typing)

# Evaluate results
with open(OUTPUT, 'rb') as f:
    reader = csv.reader(f, delimiter='\n')
    output_raw = list(reader)[1:]

results = defaultdict(list)
for x in output_raw:
    entry = x[0].split(",")
    results[int(entry[0])] = map(int, entry[1].split(" "))

def displayUser(user, tabs = ""):
    print tabs + "USER: " + str(users[user][0])
    print tabs + "\tROLES: " + str(users[user][1])
    print tabs + "\tCAREER: " + str(users[user][2])
    print tabs + "\tDISCIPLINE: " + str(users[user][3])
    print tabs + "\tINDUSTRY: " + str(users[user][4])
    print tabs + "\tCOUNTRY: " + str(users[user][5])
    print tabs + "\tREGION: " + str(users[user][6])
    print tabs + "\tEXPCLASS: " + str(users[user][7])
    print tabs + "\tEXPYOLD: " + str(users[user][8])
    print tabs + "\tEXPYNOW: " + str(users[user][9])
    print tabs + "\tDEGREE: " + str(users[user][10])
    print tabs + "\tFIELD: " + str(users[user][11])

def displayItem(item, tabs = ""):
    print tabs + "ITEM: " + str(items[item][0])
    print tabs + "\tTITLE: " + str(items[item][1])
    print tabs + "\tCAREER: " + str(items[item][2])
    print tabs + "\tDISCIPLINE: " + str(items[item][3])
    print tabs + "\tINDUSTRY: " + str(items[item][4])
    print tabs + "\tCOUNTRY: " + str(items[item][5])
    print tabs + "\tREGION: " + str(items[item][6])
    print tabs + "\tLAT: " + str(items[item][7])
    print tabs + "\tLNG: " + str(items[item][8])
    print tabs + "\tEMPLOY: " + str(items[item][9])
    print tabs + "\tTAGS: " + str(items[item][10])
    print tabs + "\tDAYSAGO: " + str((1446937200 - items[item][11])/ (24 * 60 * 60))
    print tabs + "\tACTIVE: " + str(items[item][12])

# print "CAREER | DISCIPLINE | INDUSTRY | COUNTRY | REGION | EXPCLASS | EXPYOLD | EXPYNOW | DEGREE"
def usergrid(user):
    msg = "\t"
    for i in [2,3,4,5,6,7,8,9,10]:
        msg += str(users[user][i]) + "\t"
    print msg

# print "CAREER | DISCIPLINE | INDUSTRY | COUNTRY | REGION | LAT | LNG | EMPLOY | TS | ACTIVE"
def itemgrid(itemlist):
    print "CAREER | DISCIPLINE | INDUSTRY | COUNTRY | REGION | LAT | LNG | EMPLOY | TS | ACTIVE"
    for item in itemlist:
        msg = "\t"
        for i in [2,3,4,5,6,7,8,9,11,12]:
            msg += str(items[item][i]) + "\t"
        print msg
    print "== TITLES =="
    titles = defaultdict(int)
    for item in itemlist:
        for title in items[item][1]:
            #Check correctness
            if len(title)>0:
                titles[title] += 1
    ordertitle = sorted(titles, key=titles.get,reverse = True)
    for title in ordertitle[:10]:
        print "TITLE: " + str(title) + " => " + str(titles[title])
    print "== TAGS =="
    tags = defaultdict(int)
    for item in itemlist:
        for tag in items[item][10]:
            #Check correctness
            if len(tag)>0:
                tags[tag] += 1
    ordertag = sorted(tags, key=tags.get,reverse = True)
    for tag in ordertag[:10]:
        print "TAG: " + str(tag) + " => " + str(tags[tag])

# Compute score
score = float(0)
for user in validation.keys()[2500:]:
    print "----------- USER ------------"
    print users[user]
    print "Interactions: " + str(len(user_item[user].keys()))
    displayUser(user)
    itemgrid(user_item[user].keys())
    print "----- VALID -----"
    print list(validation[user])
    itemgrid(validation[user])
    print "----- RECOMMENDED -----"
    print results[user]
    itemgrid(results[user])
    minl = min(5, len(validation[user]))
    for i in range(1,minl+1):
        l = len(set.intersection(validation[user],set(results[user][:i]))) / float(i * minl)
        print "CUTOFF: " + str(i) + " => INTERSECTION: " + str(len(set.intersection(validation[user],set(results[user][:i]))))
        score += l
    sys.stdin.read(1)

fscore = score / float(len(validation.keys()))
print "Final score: "
print fscore
