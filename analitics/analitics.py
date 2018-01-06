"Exploring the dataset"


import cPickle as pickle
import csv
import time
import numpy as np
from collections import defaultdict
from functools import partial

#First load all datasets and convert them
with open('../Dataset/interactions.csv', 'rb') as f:
    reader = csv.reader(f, delimiter='\t')
    interactions = list(reader)[1:]
with open('../Dataset/item_profile.csv', 'rb') as f:
    reader = csv.reader(f, delimiter='\t')
    items = list(reader)
    item_headers = items[0]
    items = items[1:]
with open('../Dataset/user_profile.csv', 'rb') as f:
    reader = csv.reader(f, delimiter='\t')
    users = list(reader)[1:]
with open('../Dataset/target_users.csv', 'rb') as f:
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
def convertItem(item):
    newItem = {}
    newItem['id'] = intify(item[0])
    newItem['title'] = item[1].split(",")
    for x in range(0,len(newItem['title'])):
        newItem['title'][x] = intify(newItem['title'][x])
    newItem['career_level'] = intify(item[2])
    newItem['discipline'] = intify(item[3])
    newItem['industry'] = intify(item[4])
    if item[5] == 'de':
        newItem['country'] = 1
    elif item[5] == 'at':
        newItem['country'] = 2
    elif item[5] == 'ch':
        newItem['country'] = 3
    elif item[5] == 'non_dach':
        newItem['country'] = 4
    else:
        newItem['country'] = 0
    newItem['region'] = intify(item[6])
    newItem['latitude'] = floatify(item[7])
    newItem['longitude'] = floatify(item[8])
    newItem['employment'] = intify(item[9])
    newItem['tags'] = item[10].split(",")
    for x in range(0,len(newItem['tags'])):
        newItem['tags'][x] = intify(newItem['tags'][x])
    newItem['created'] = intify(item[11])
    newItem['active'] = intify(item[12])
    return newItem

def convertInteraction(interact):
    newInteract = {}
    newInteract['userid'] = intify(interact[0])
    newInteract['itemid'] = intify(interact[1])
    newInteract['type'] = intify(interact[2])
    newInteract['ts'] = intify(interact[3])
    return newInteract

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
def convertUser(user):
    newUser = {}
    newUser['id'] = intify(user[0])
    newUser['jobroles'] = user[1].split(",")
    for x in range(0,len(newUser['jobroles'])):
        newUser['jobroles'][x] = intify(newUser['jobroles'][x])
    newUser['career_level'] = intify(user[2])
    newUser['discipline'] = intify(user[3])
    newUser['industry'] = intify(user[4])
    if user[5] == 'de':
        newUser['country'] = 1
    elif user[5] == 'at':
        newUser['country'] = 2
    elif user[5] == 'ch':
        newUser['country'] = 3
    elif user[5] == 'non_dach':
        newUser['country'] = 4
    else:
        newUser['country'] = 0
    newUser['region'] = intify(user[6])
    newUser['experience_n_entries'] = intify(user[7])
    newUser['experience_years'] = intify(user[8])
    newUser['experience_years_current'] = intify(user[9])
    newUser['edu_degree'] = intify(user[10])
    newUser['edu_fields'] = user[11].split(",")
    for x in range(0,len(newUser['edu_fields'])):
        newUser['edu_fields'][x] = intify(newUser['edu_fields'][x])
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

item_active = [x for x in items if x['active'] == 1]
item_active = set([i['id'] for i in item_active])

interacts_active = []
for i in interactions:
    if i['itemid'] in item_active:
        interacts_active.append(i)
print "Done with converting and activating"

'''
STARTING REAL COMPUTATION
'''

#----- BUILD USER_ITEM INTERACTIONS ------
user_item = defaultdict(set)

counter = 0

for i in users[:10]:
    counter += 1
    user_item[i['id']] = []
    for x in interactions:
        if x['userid'] == i['id']:
            entry = {}
            entry['itemid'] = x['itemid']
            entry['ts'] = x['ts']
            entry['type'] = x['type']
            user_item[i['id']].append(entry)
    if counter % 1000 == 0:
        print counter
pickle.dump(user_item, open('user_item.pkl', 'wb'))
print user_item

# ------- GENERIC -------

print "Done"
