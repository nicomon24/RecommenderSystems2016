
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

with open('./Dataset/interactions.csv', 'rb') as f:
    reader = csv.reader(f, delimiter='\t')
    interactions = list(reader)[1:]
with open('./Dataset/item_profile.csv', 'rb') as f:
    reader = csv.reader(f, delimiter='\t')
    items = list(reader)[1:]
with open('./Dataset/user_profile.csv', 'rb') as f:
    reader = csv.reader(f, delimiter='\t')
    users = list(reader)[1:]
with open('./Dataset/target_users.csv', 'rb') as f:
    reader = csv.reader(f, delimiter='\t')
    targets = list(reader)[1:]
print "Loaded datasets"

#DO THINGS
itemindex = {}
for i in items:
    itemindex[i[0]] = i
userindex = {}
for u in users:
    userindex[u[0]] = u
superneigh = {}
supercoll = {}

#Scores dictionary
topscores = defaultdict(int)
#Computing scores
for row in interactions:
    topscores[row[1]] += int(row[2])
#let's sort it
toppop = sorted(topscores, key=topscores.get,reverse = True)[:5]

def compareArray(a1,a2):
    mlen = min (len(a1), len(a2))
    counter = 0
    for x in range(len(a1)):
        for y in range(len(a2)):
            if (a1[x] == a2[y]):
                counter += 1
    return counter / float(mlen)

def userSimilarity(i1, i2):
    #magic = [jobroles,career_level,discipline_id,industry_id,country, region,experience_n_entries_class,experience_years_experience, experience_years_in_current, edu_degree, edu_fieldofstudies]
    magic = [1.8,0.5,1,0.7,0.9,0.8,0.7,0.6,0.6,1,1]
    score = float(0)
    for i in range(1,len(i1)):
        row1 = i1[i].split(",")
        row2 = i2[i].split(",")
        score += magic[i-1] * compareArray(row1, row2)
    return (i2[0],score)

#Similarity between items in 0..1
def itemSimilarity(i1, i2):
    #magic = [title,career,discipline,indu,count,regio,lat,lng,empl,tags,crea,active]
    magic = [1.8,0.3,0.85,0.3,0.7,0.65,0,0,0.6,1.5,0,0]
    score = float(0)
    for i in range(1,len(i1)):
        row1 = i1[i].split(",")
        row2 = i2[i].split(",")
        score += magic[i-1] * compareArray(row1, row2)
    return (i2[0],score)

def neighbours(item):
    if item[0] in superneigh.keys():
        print "returning cached value"
        return superneigh[item[0]]
    else:
        pool = ThreadPool(64)
        fixed = partial(itemSimilarity, item)
        results = dict(pool.map(fixed, items))
        sort = sorted(results, key=results.get, reverse = True)
        superneigh[item[0]] = sort[:10]
        return sort[:10]

def collegues(user):
    if user[0] in supercoll.keys():
        print "returning cached value"
        return superneigh[item[0]]
    else:
        pool = ThreadPool(64)
        fixed = partial(userSimilarity, user)
        results = dict(pool.map(fixed, users))
        sort = sorted(results, key=results.get, reverse = True)
        supercoll[user[0]] = sort[:10]
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
    print("Loaded interests " + str(len(liked)))
    giganeigh = []
    for l in liked:
        giganeigh.extend(neighbours(itemindex[l]))
        print("Loaded some neighbours " + l)
    giga = list(set(giganeigh) - set(liked))
    scores = defaultdict(float)
    print("Loaded giga")
    for x in giga:
        sim = 0
        for lik in liked:
            sim += itemSimilarity(itemindex[lik],itemindex[x])[1]
        scores[x] = sim
    sort = sorted(scores, key=scores.get, reverse = True)
    print "done some user "  + user[0]
    return sort

def loadTopPop():
    wanama = np.genfromtxt('./Dataset/interactions.csv', delimiter="\t")
    wanama = np.delete(wanama, (0), axis=0)
    #Scores dictionary
    scores = defaultdict(int)
    #Computing scores
    for row in wanama:
        scores[row[1]] += row[2]
    #let's sort it
    o = sorted(scores, key=scores.get,reverse = True)
    return o;

#MAIN CODE
for u in targets[1000:1010]:
    print u
    coll_interests = []
    col = collegues(userindex[u[0]])
    for i in col:
        coll_interests.extend(interests(userindex[i]))
    coll_interests = set(coll_interests)
    cluster = []
    for w in recommend(userindex[u[0]]):
        cluster.append(w)
    final = []
    for i in cluster:
        for j in coll_interests:
            if i==j :
                print "Relevant Jobs:"
                final.append(i)
for lol in final:
    print lol





'''
toppop = loadTopPop()
for i in toppop:
    print int(i)
'''

'''
for u in targets[:5]:
    for z in targets[:5]:
        print userSimilarity(userindex[u[0]], userindex[z[0]])
'''

'''
f = open('submission' + str(time.time()).split('.')[0] +'.csv', 'w')
f.write("user_id,recommended_items\n")
for u in targets[2:3]:
    reco = recommend(userindex[u[0]])
    f.write(u[0] + "," + " ".join(reco[1]) + "\n")
f.close()
'''
