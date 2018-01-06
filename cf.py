
#IMPORT
import time
import numpy as np
import csv
from collections import defaultdict
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial

#MODEL DEFINITION
from model.interaction import Interaction as interact
from model.userprofile import Userprofile as user
from model.itemprofile import Itemprofile as item
#E.g.   xyz = item(item_profiles[0])
#       print xyz.career_level

#LOADING DATASETS
#Loading interactions
pool = ThreadPool(4)

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
print "Loaded datasets"

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

#Reshape array, convert to int
for i in range(0,len(items)):
    items[i] = convertItem(items[i])

#DO THING
itemindex = {}
for i in items:
    itemindex[i[0]] = i
userindex = {}
for u in users:
    userindex[u[0]] = u
superneigh = {}

#Clear unactive items
items = [x for x in items if x[12] == 1]

#Scores dictionary
topscores = defaultdict(int)
#Computing scores
for row in interactions:
    topscores[row[1]] += int(row[2])
#let's sort it
toppop = sorted(topscores, key=topscores.get,reverse = True)[:5]
print "Generated toppop:"
print toppop

#Counting stars
def countStars():
    for x in range(2,7):
        scores = defaultdict(int)
        for i in items:
            scores[i[x]] += 1
        print "Column " + str(item_headers[x]) + "\n" + "Count: " + str(len(scores.keys()))
        print scores
        print "\n-------\n"

def compareArray(a1,a2):
    if isinstance(a1, int) or isinstance(a1, float):
        if a1 == a2:
            return 1
        else:
            return 0
    else:
        mlen = min (len(a1), len(a2))
        counter = 0
        for x in range(len(a1)):
            for y in range(len(a2)):
                if (a1[x] == a2[y]):
                    counter += 1
        return counter / float(mlen)

#Similarity between items in 0..1
def itemSimilarity(i1, i2):
    #magic = [title,career,discipline,indu,count,regio,lat,lng,empl,tags,crea,active]
    magic = [1.8,0.3,0.85,0.3,0.7,0.65,0,0,0.6,1.5,0,0]
    score = float(0)
    for i in [2,3,4,5,6,7,10,11]:
        score += magic[i-1] * compareArray(i1[i], i2[i])
    return (i2[0],score)

def neighbours(item):
    if item[0] in superneigh.keys():
        #print "returning cached value"
        return superneigh[item[0]]
    else:
        fixed = partial(itemSimilarity, item)
        results = dict(pool.map(fixed, items))
        sort = sorted(results, key=results.get, reverse = True)
        superneigh[item[0]] = sort[:10]
        return sort[:10]

def interests(user):
    res = []
    for interact in interactions:
        if user[0] == interact[0]:
            res.append(interact[1])
    return res

def recommend(user):
    print("Recommending user " + user[0])
    liked = interests(user)
    #print("Loaded interests " + str(len(liked)))
    giganeigh = []
    for l in liked:
        giganeigh.extend(neighbours(itemindex[int(l)]))
        #print("Loaded some neighbours " + l)
    giga = list(set(giganeigh) - set(liked))
    scores = defaultdict(float)
    for x in giga:
        sim = 0
        for lik in liked:
            sim += itemSimilarity(itemindex[int(lik)],itemindex[x])[1]
        scores[x] = sim
    sort = sorted(scores, key=scores.get, reverse = True)
    #print "done some user "  + user[0]
    return [user[0],sort[:5]]

f = open('submission' + str(time.time()).split('.')[0] +'.csv', 'w')
f.write("user_id,recommended_items\n")
counter = 0
for u in targets:
    reco = map(str,recommend(userindex[u[0]])[1])
    if len(reco) < 5:
        reco.extend(toppop)
    f.write(u[0] + "," + " ".join(map(str, reco[:5])) + "\n")
    print "Avanti un altro [" + str(counter) + "/10000]"
    counter += 1

f.close()
