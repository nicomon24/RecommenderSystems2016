
#--------- IMPORT --------
import time, math, sys, os, csv
import numpy as np
import cPickle as pickle
from collections import defaultdict
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial

#--------- CONST ---------
INTERACTIONS_FILE = './Dataset/interactions.csv'
TARGET_FILE = './Dataset/target_users.csv'
USERS_FILE = './Dataset/user_profile.csv'
ITEMS_FILE = './Dataset/item_profile.csv'
OUTPUT_FILE = str(time.time()).split('.')[0] + "CLEAN" + '.csv'
#--------- TUNING --------
#Dataset tuning
LAST_INTERACTION_DAYS = 14
#Confidence
CONFIDENCE_WEIGHTS = [3.75, 1.0, 1.25, 1.5]
#Content
CONTENT_SIMILARITY_THRESHOLD = 0.08
USER_SIMILARITY_MAGIC = [0,1.5,0.85,0.8,0.6,0.7,1,0.4,0.6,0,1,1]
#Collaborative
MIN_INTERACTION_FOR_COLLABORATIVE = 1
TAGS_WEIGHT = 0.2
TITLE_WEIGHT = 0.2
SIMILARITY_MIN_ITEM1 = 1
SIMILARITY_MIN_ITEM2 = 6
SIMILARITY_CONFIDENCE_AMPLIFIER = 3
SIMILARITY_SHRINK = 10
COLLABORATIVE_CONFIDENCE_SCORE_BOOST = 2
COLLABORATIVE_SCORE_SHRINK = 2

#--------- DATASET -------
#Check if datasets where passed as args
if len(sys.argv) == 4:
    INTERACTIONS_FILE = sys.argv[1]
    TARGET_FILE = sys.argv[2]
    OUTPUT_FILE = sys.argv[3]

with open(INTERACTIONS_FILE, 'rb') as f:
    reader = csv.reader(f, delimiter='\t')
    interactions_raw = list(reader)[1:]
with open(ITEMS_FILE, 'rb') as f:
    reader = csv.reader(f, delimiter='\t')
    items_raw = list(reader)[1:]
with open(USERS_FILE, 'rb') as f:
    reader = csv.reader(f, delimiter='\t')
    users_raw = list(reader)[1:]
with open(TARGET_FILE, 'rb') as f:
    reader = csv.reader(f, delimiter='\t')
    targets_raw = list(reader)[1:]

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

def convertTarget(target):
    newTarget = intify(target[0])
    return newTarget

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

targets = []
for i in targets_raw:
    targets.append(convertTarget(i))

items = defaultdict(list)
for i in items_raw:
    it = convertItem(i)
    items[it[0]] = it

users = defaultdict(list)
for u in users_raw:
    us = convertUser(u)
    users[us[0]] = us

# Reduce items and interactions only to active ones
item_active = set([x for x in items if items[x][12] == 1])
interactions_active = []
for i in interactions:
    if i[1] in item_active:
        interactions_active.append(i)

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

user_item = user_item_index(interactions)
user_item_typed = user_item_typed_index(interactions)
item_user = item_user_index(interactions)
item_active = set.intersection(item_active, set(item_user.keys()))

# Reduce recommendable_items with last interaction threshold
max_timestamp = 1447023571
items_last_week = defaultdict(long)
for i in interactions_active:
    if i[3] > max_timestamp - (24*60*60 * LAST_INTERACTION_DAYS):
        items_last_week[i[1]] = i[3]
recommendable_items = set.intersection(set(items_last_week.keys()),item_active)
print "recommendable_items: " + str(len(recommendable_items))

# Confidence
def confidence_index(temp_user_item_typed):
    weights = CONFIDENCE_WEIGHTS
    temp_confidence = defaultdict(lambda : defaultdict(float))
    for user in temp_user_item_typed:
        totals = np.sum([x for x in temp_user_item_typed[user].values()], axis=0)
        den = sum([ totals[i] * weights[i+1] for i in [0,1,2]]) + weights[0]
        for item in temp_user_item_typed[user]:
            num = sum([ temp_user_item_typed[user][item][i] * weights[i+1] for i in [0,1,2]])
            temp_confidence[item][user] = num / float(den)
    return temp_confidence

confidence = confidence_index(user_item_typed)

tags = defaultdict(set)
titles = defaultdict(set)
tot = len(items.keys())
item_tags_tf = defaultdict(float)
item_titles_tf = defaultdict(float)
for ID, item in items.iteritems():
	tagslist = item[10]
	titleslist = item[1]
	item_tags_tf[ID] = (1/float(len(tagslist))) * 0.3 + 0.7
	item_titles_tf[ID] = (1/float(len(titleslist))) * 0.7 + 0.3
	for tag in tagslist:
		tags[tag].add(ID)
	for title in titleslist:
		titles[title].add(ID)
tags_idf = {key: math.log(tot/len(value)) for key, value in tags.iteritems()}
titles_idf = {key: math.log(tot/len(value)) for key, value in titles.iteritems()}

item_tags_idf_norm = defaultdict(float)
item_titles_idf_norm = defaultdict(float)
for item in items.keys():
    item_tags_idf_norm[item] = math.sqrt(sum(tags_idf[x]**2 for x in items[10]))
    item_titles_idf_norm[item] = math.sqrt(sum(titles_idf[x]**2 for x in items[1]))


max_tags = max(tags_idf.values())
max_titles = max(titles_idf.values())
tags_idf = {key: value/max_tags for key, value in tags_idf.iteritems()}
titles_idf = {key: value/max_titles for key, value in titles_idf.iteritems()}

#--------- COMPUTE -------

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
                counter += 1
        return counter / (math.sqrt(len(a1) * len(a2)) + 2)

#Similarity between users
def userSimilarity(target, u2):
    u1 = users[target]
    u2 = users[u2]
    #magic = [id, jobroles, career level, discipline, industry, country, region, experience_n_entries, experience_years, experience_years_current, edu_degree,edu_fields]
    #magic = [0,1.8,0.9,1.1,0.6,0.6,0.4,0.4,0.6,0,1.1,1.1]
    magic = USER_SIMILARITY_MAGIC
    score = float(0)
    for i in range(1, min(len(u1),len(u2))):
        score += magic[i] * compareArray(u1[i], u2[i])
    return score/8

def tagsTitlesSimilarity(i1,i2):
    titles1 = set(items[i1][1])
    titles2 = set(items[i2][1])
    tags1 = set(items[i1][10])
    tags2 = set(items[i2][10])
    titles_intersection = set.intersection(titles1, titles2)
    tags_intersection = set.intersection(tags1, tags2)
    sim = 0
    titlesnum = 0
    tagsnum = 0
    for title in titles_intersection:
        titlesnum += titles_idf[title]**2
    sim += titlesnum / (item_titles_idf_norm[i1] * item_titles_idf_norm[i2] + 3)
    for tag in tags_intersection:
        tagsnum += tags_idf[tag]**2
    sim += tagsnum / (item_tags_idf_norm[i1] * item_tags_idf_norm[i2] + 3)
    return sim

def computeSimilarity(i1,i2):
    users1 = set(item_user[i1].keys())
    users2 = set(item_user[i2].keys())
    len1 = len(users1)
    len2 = len(users2)
    if len1 <= SIMILARITY_MIN_ITEM1 and len2 <= SIMILARITY_MIN_ITEM2:
        return 0
    else:
        intersection = set.intersection(users1,users2)
        nequal= len(intersection)
        for i in intersection:
            nequal += (confidence[i1][i] + confidence[i1][i]) * SIMILARITY_CONFIDENCE_AMPLIFIER
        den = math.sqrt(len1 * len2) + SIMILARITY_SHRINK
        score = nequal / den
        return score

# Toppop generation
def build_toppop():
    counter = defaultdict(int)
    for interaction in interactions_active:
        if interaction[1] in recommendable_items:
            counter[interaction[1]] += interaction[2]
    temp_toppop = sorted(counter, key=counter.get,reverse = True)[:20]
    return temp_toppop

toppop = build_toppop()

# USER - Content
def content(target, likes):
    likes = set(likes.keys())
    neighbours = defaultdict(float)
    for u in user_item:
        if u != target:
            sim = userSimilarity(target, u)
            if sim > CONTENT_SIMILARITY_THRESHOLD:
                neighbours[u] = sim
    topneighbours = sorted(neighbours, key=neighbours.get,reverse = True)[:150]
    neigh_interests = defaultdict(float)
    for u in topneighbours:
        for i in user_item[u]:
            if i in recommendable_items and i not in likes:
                neigh_interests[i] += neighbours[u]
    '''
    if len(likes) > 0:
        for i in likes:
            for j in neigh_interests:
                titles = set.intersection(set(items[i][1]), set(items[j][1]))
                tags = set.intersection(set(items[i][10]), set(items[j][10]))
                bonus = 1.0 + 0.2 * (len(tags) + len(titles))
                neigh_interests[j] = neigh_interests[j] * bonus
                print bonus
    '''
    top_interests = sorted(neigh_interests, key=neigh_interests.get,reverse = True)
    j = 0
    while len(top_interests) < 5:
        if toppop[j] not in likes:
            top_interests.append(toppop[j])
        j += 1
    return top_interests[:5]

# Collaborative
def collaborative(user_id):
    # Prepare likes and neighbours
    likes = user_item[user_id]
    likes_keys = set(likes.keys())
    similar_users = set()
    for l in likes_keys:
        similar_users.update(set(item_user[l].keys()))
    similar_items = set()
    for u in similar_users:
        similar_items.update(set(user_item[u].keys()))
    similar_items = set.intersection(similar_items, recommendable_items)
    similar_items = similar_items - set(likes_keys)
    # Compute scores
    scores = defaultdict(float)
    for i in likes_keys:
        for j in similar_items:
            sim = computeSimilarity(i,j)
            sim += tagsTitlesSimilarity(i,j) / 5
            scores[j] += sim * (likes[i] + confidence[i][user_id]*COLLABORATIVE_CONFIDENCE_SCORE_BOOST + COLLABORATIVE_SCORE_SHRINK)
    # if something goes wrong put toppop
    top = sorted(scores, key=scores.get,reverse = True)[:5]
    for i in top:
        if scores[i] == 0 or len(top) < 5:
            return content(user_id, likes)
    return top

# Main function to recommend, call other strategies
def recommend(user_id):
    likes = user_item[user_id]
    print "number of interactions: " + str(len(likes.keys()))
    # If user has more than MIN_INTERACTION_FOR_COLLABORATIVE
    interaction_number = sum(likes.values())
    if interaction_number > MIN_INTERACTION_FOR_COLLABORATIVE:
        # collaborative
        top = collaborative(user_id)
    else:
        # content-based
        top = content(user_id, likes)
    return top[:5]

# -------- MAIN ---------
fout = open(OUTPUT_FILE, 'w')
fout.write("user_id,recommended_items\n")
count = 0
for target in targets:
    count += 1
    print "Recommending user: " + str(count) + " | " + str(target)
    top = recommend(target)
    if len(top) < 5:
        top.extend(toppop)
    fout.write(str(target) + "," + " ".join(map(str, top[:5])) + "\n")
    print top[:5]
fout.close()
