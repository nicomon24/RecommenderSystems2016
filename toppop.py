import numpy as np
from collections import defaultdict

#Loading interactions (user_id, item_id, type, timestamp)
interactions = np.genfromtxt('./Dataset/interactions.csv', delimiter="\t")
interactions = np.delete(interactions, (0), axis=0)

#Loading target users (user_id, jobroles, career_level, discipline, industry, country, region, experience ...  )
target = np.genfromtxt('./Dataset/target_users.csv', delimiter="\t")
target = np.delete(target, (0), axis=0)

#Scores dictionary
scores = defaultdict(int)
#Computing scores
for row in interactions:
    scores[row[1]] += row[2]
#let's sort it
print len(scores)
o = sorted(scores, key=scores.get,reverse = True)
#output csv
f = open('submission.csv', 'w')
f.write("user_id,recommended_items\n")
for i in range(0,10000):
    f.write(str(int(target[i])) + ", " + str(int(o[0])) + " " + str(int(o[1])) + " "+ str(int(o[2])) + " "+ str(int(o[3])) + " " + str(int(o[4])) + "\n")
f.close()

#double check
for x in range(0,5):
    print scores[o[x]], o[x]
