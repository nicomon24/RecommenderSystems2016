"Export dataset to local PostgreSQL DB"

import cPickle as pickle
import csv
import time
import numpy as np
from collections import defaultdict
from functools import partial
import pg8000

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

print "Done with converting"

'''
STARTING REAL THINGS
'''

#Connect to DB
conn = pg8000.connect(user="nicom", password="nico2405", database= "recommender")
cursor = conn.cursor()

#Insert users
for u in users:
    #Insert user all
    cursor.execute("INSERT INTO users (id, career_level, discipline, industry, country, region, experience_entries, experience_years, current_years, edu_degree) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
    (str(u['id']), str(u['career_level']), str(u['discipline']), str(u['industry']), str(u['country']), str(u['region']), str(u['experience_n_entries']), str(u['experience_years']), str(u['experience_years_current']), str(u['edu_degree'])))
    #Insert user jobroles
    for j in u['jobroles']:
        if j > 0:
            cursor.execute("INSERT INTO user_jobroles (userid, value) VALUES (%s, %s)", (str(u['id']), str(j)))
    #Insert user fields
    for f in u['edu_fields']:
        if f >= 0:
            cursor.execute("INSERT INTO user_edufield (userid, value) VALUES (%s, %s)", (str(u['id']), str(f)))

print "Done with users"

#Insert items
for i in items:
    #Insert item all
    cursor.execute("INSERT INTO item (id, career_level, discipline, industry, country, region, lat, lng, employment, ts, active) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
    (str(i['id']), str(i['career_level']), str(i['discipline']), str(i['industry']), str(i['country']), str(i['region']), str(i['latitude']), str(i['longitude']), str(i['employment']), str(i['created']), str(i['active'])))
    #Insert item titles
    for ti in i['title']:
        if ti >= 0:
            cursor.execute("INSERT INTO item_title (itemid, value) VALUES (%s, %s)", (str(i['id']), str(ti)))
    #Insert item tags
    for ta in i['tags']:
        if ta >= 0:
            cursor.execute("INSERT INTO item_tags (itemid, value) VALUES (%s, %s)", (str(i['id']), str(ta)))

print "Done with items"

#Insert interactions
for ix in interactions:
    cursor.execute("INSERT INTO interaction (userid, itemid, type, ts) VALUES (%s, %s, %s, %s)",
    (str(ix['userid']), str(ix['itemid']), str(ix['type']), str(ix['ts'])))

conn.commit()
print "Done"

'''
SCHEMAS:

------ USERS -------

CREATE TABLE users(
    id INT PRIMARY KEY,
    career_level INT,
    discipline INT,
    industry INT,
    country INT,
    region INT,
    experience_entries INT,
    experience_years INT,
    current_years INT,
    edu_degree INT
);

CREATE TABLE user_jobroles(
    userid INT REFERENCES users(id),
    value INT,
    PRIMARY KEY (userid, value)
);

CREATE INDEX user_jobroles_idx ON user_jobroles(userid);

CREATE TABLE user_edufield(
    userid INT REFERENCES users(id),
    value INT,
    PRIMARY KEY (userid, value)
);

CREATE INDEX user_edufield_idx ON user_edufield(userid);

------- ITEMS --------

CREATE TABLE item(
    id INT PRIMARY KEY,
    career_level INT,
    discipline INT,
    industry INT,
    country INT,
    region INT,
    lat REAL,
    lng REAL,
    employment INT,
    ts INT,
    active INT
);

CREATE TABLE item_title(
    itemid INT REFERENCES item(id),
    value INT,
    PRIMARY KEY (itemid, value)
);

CREATE INDEX item_title_idx ON item_title(itemid);

CREATE TABLE item_tags(
    itemid INT REFERENCES item(id),
    value INT,
    PRIMARY KEY (itemid, value)
);

CREATE INDEX item_tags_idx ON item_tags(itemid);

------- INTERACTIONS --------

CREATE TABLE interaction(
    id SERIAL,
    userid INT REFERENCES users(id),
    itemid INT REFERENCES item(id),
    type INT,
    ts INT
);

CREATE INDEX interaction_user_idx ON interaction(userid);
CREATE INDEX interaction_item_idx ON interaction(itemid);

CREATE TABLE useritemconfidence(
    userid INT REFERENCES users(id),
    itemid INT REFERENCES item(id),
    score REAL
);

CREATE MATERIALIZED VIEW useritemconfidence AS SELECT userid, itemid, (scoresum / totsum::float) AS score FROM (
    SELECT userid, itemid, sum(item_score) AS scoresum, ( SELECT sum(total_score) FROM (
        SELECT type, CASE WHEN type = 1 THEN count(*) * 1.0
                    WHEN type = 2 THEN count(*) * 1.4
                    WHEN type = 3 THEN count(*) * 1.8
             END AS total_score
             FROM interaction as x WHERE x.userid = types.userid
             GROUP BY x.type ) AS totaltype
    ) AS totsum FROM (
        SELECT userid, itemid, type, count(*),
                CASE WHEN type = 1 THEN count(*) * 1.0
                     WHEN type = 2 THEN count(*) * 1.4
                     WHEN type = 3 THEN count(*) * 1.8
                END AS item_score
        FROM interaction as i GROUP BY userid, itemid, type
    ) AS types GROUP BY types.userid, types.itemid
) AS result;

CREATE MATERIALIZED VIEW usercounters AS SELECT t1.id, (t1.c+t2.c+t3.c) as total, tc.c as distincted, t1.c as type1, t2.c as type2, t3.c as type3 FROM (
            SELECT i1.id, COUNT(i2.userid) AS c FROM users as i1 LEFT JOIN interaction as i2 ON i1.id = i2.userid AND i2.type=1 GROUP BY i1.id
) AS t1, (
            SELECT i1.id, COUNT(i2.userid) AS c FROM users as i1 LEFT JOIN interaction as i2 ON i1.id = i2.userid AND i2.type=2 GROUP BY i1.id
) AS t2, (
            SELECT i1.id, COUNT(i2.userid) AS c FROM users as i1 LEFT JOIN interaction as i2 ON i1.id = i2.userid AND i2.type=3 GROUP BY i1.id
) AS t3,(
            SELECT i1.id, COUNT(i2.userid) AS c FROM users as i1 LEFT JOIN useritemconfidence as i2 ON i1.id = i2.userid GROUP BY i1.id
) AS tc
WHERE t1.id = t2.id AND t2.id = t3.id AND t1.id = tc.id;

CREATE MATERIALIZED VIEW toptags AS SELECT x.userid, x.value, (x.tagged / c.distincted::float) as score, c.distincted FROM (
        SELECT i.userid, t.value, count(*) as tagged
        FROM useritemconfidence as i, item_tags as t
        WHERE i.itemid = t.itemid
        GROUP BY i.userid, t.value
) AS x, usercounters AS c WHERE x.userid = c.id ORDER BY score DESC;

CREATE MATERIALIZED VIEW toptitles AS SELECT x.userid, x.value, (x.tagged / c.distincted::float) as score, c.distincted FROM (
        SELECT i.userid, t.value, count(*) as tagged
        FROM useritemconfidence as i, item_title as t
        WHERE i.itemid = t.itemid
        GROUP BY i.userid, t.value
) AS x, usercounters AS c WHERE x.userid = c.id ORDER BY score DESC;

CREATE MATERIALIZED VIEW itemcounters AS SELECT t1.id, (t1.c+t2.c+t3.c) as total, tc.c as distincted, t1.c as type1, t2.c as type2, t3.c as type3 FROM (
            SELECT i1.id, COUNT(i2.itemid) AS c FROM item as i1 LEFT JOIN interaction as i2 ON i1.id = i2.itemid AND i2.type=1 GROUP BY i1.id
) AS t1, (
            SELECT i1.id, COUNT(i2.itemid) AS c FROM item as i1 LEFT JOIN interaction as i2 ON i1.id = i2.itemid AND i2.type=2 GROUP BY i1.id
) AS t2, (
            SELECT i1.id, COUNT(i2.itemid) AS c FROM item as i1 LEFT JOIN interaction as i2 ON i1.id = i2.itemid AND i2.type=3 GROUP BY i1.id
) AS t3,(
            SELECT i1.id, COUNT(i2.itemid) AS c FROM item as i1 LEFT JOIN useritemconfidence as i2 ON i1.id = i2.itemid GROUP BY i1.id
) AS tc
WHERE t1.id = t2.id AND t2.id = t3.id AND t1.id = tc.id;

CREATE MATERIALIZED VIEW toproles AS SELECT x.itemid, x.value, (x.tagged / c.distincted::float) as score, c.distincted FROM (
        SELECT i.itemid, j.value, count(*) as tagged
        FROM useritemconfidence as i, user_jobroles as j
        WHERE i.userid = j.userid
        GROUP BY i.itemid, j.value
) AS x, itemcounters AS c WHERE x.itemid = c.id ORDER BY score DESC;

CREATE MATERIALIZED VIEW topedufield AS SELECT x.itemid, x.value, (x.tagged / c.distincted::float) as score, c.distincted FROM (
        SELECT i.itemid, j.value, count(*) as tagged
        FROM useritemconfidence as i, user_edufield as j
        WHERE i.userid = j.userid
        GROUP BY i.itemid, j.value
) AS x, itemcounters AS c WHERE x.itemid = c.id ORDER BY score DESC;

CREATE MATERIALIZED VIEW usertype1 AS SELECT i.userid, i.itemid, count(*) FROM interaction AS i WHERE type = 1 GROUP BY i.userid, i.itemid;

CREATE MATERIALIZED VIEW usertype2 AS SELECT i.userid, i.itemid, count(*) FROM interaction AS i WHERE type = 2 GROUP BY i.userid, i.itemid;

CREATE MATERIALIZED VIEW usertype3 AS SELECT i.userid, i.itemid, count(*) FROM interaction AS i WHERE type = 3 GROUP BY i.userid, i.itemid;

CREATE MATERIALIZED VIEW itemtype1 AS SELECT i.itemid, i.userid, count(*) FROM interaction AS i WHERE type = 1 GROUP BY i.itemid, i.userid;

CREATE MATERIALIZED VIEW itemtype2 AS SELECT i.itemid, i.userid, count(*) FROM interaction AS i WHERE type = 2 GROUP BY i.itemid, i.userid;

CREATE MATERIALIZED VIEW itemtype3 AS SELECT i.itemid, i.userid, count(*) FROM interaction AS i WHERE type = 3 GROUP BY i.itemid, i.userid;

------ QUERIES --------

-- TOPPOP --
SELECT id, (total * distincted) AS score FROM itemcounters ORDER BY score DESC LIMIT 30;

SELECT interaction.userid, min(interaction.ts), count(interaction.itemid), avg(item.industry), avg(item.discipline), avg(item.career_level) FROM interaction, item WHERE interaction.itemid = item.id GROUP BY interaction.userid;

SELECT interaction.userid, interaction.type, ((interaction.ts - item.ts) / 86400) as days FROM interaction, item WHERE interaction.itemid = item.id AND item.id = 1244787 ORDER BY days;

SELECT item.id, item.active, min((interaction.ts - item.ts) / 86400) as mindays, max((interaction.ts - item.ts) / 86400) as maxdays, count(interaction.id) FROM interaction, item WHERE interaction.itemid = item.id AND item.active = 1 AND item.ts > 0 GROUP BY item.id HAVING count(interaction.id) > 5 LIMIT 50;

SELECT item.id, item.career_level, item.discipline, item.industry, item.country, item.region, item.employment FROM interaction, item WHERE interaction.itemid = item.id AND interaction.userid = 20596;

SELECT interaction.userid, item.country, count(*) as total  FROM interaction, item WHERE interaction.itemid = item.id GROUP BY interaction.userid, item.career_level;

select itemid, json_agg(json_build_object(value, score)) from toproles where score > 0.5 group by itemid;

SELECT value, array_agg(userid) FROM toptags WHERE score > 0.6 AND distincted > 10 AND value=3241763 GROUP BY value;

SELECT i1.id, COUNT(i2.userid) AS c FROM (users as u, item LEFT JOIN interaction as i2 ON i1.id = i2.userid AND i2.type=1 GROUP BY i1.id

CREATE MATERIALIZED VIEW usertype1 AS SELECT i.userid, i.itemid, count(*) FROM interaction AS i WHERE type = 1 GROUP BY i.userid, i.itemid;

SELECT userid, array_agg(value), array_agg(score) FROM toptags WHERE score >= 0.5 AND distincted >= 5 AND userid = 47387 GROUP BY userid;

MAX inter TS 1447023571
MIN inter TS 1440021654
MAX item TS 1446937200
MIN item TS 1432245600

inter duration = 7001917 => 81 days
item duration = 14691600 => 170 days

'''
