import csv
from collections import defaultdict

class Dataset:

    def intify(self, str):
        try:
            return int(str)
        except ValueError, ex:
            return int(-1)

    def floatify(self, str):
        try:
            return float(str)
        except ValueError, ex:
            return float(-1)

    def convertInter(self, inter):
        newInter = []
        newInter.append(self.intify(inter[0]))
        newInter.append(self.intify(inter[1]))
        newInter.append(self.intify(inter[2]))
        newInter.append(self.intify(inter[3]))
        return newInter

    def convertTarget(self, target):
        newTarget = self.intify(target[0])
        return newTarget

    def convertUser(self, user):
        newUser = []
        newUser.append(self.intify(user[0]))
        newUser.append(map(self.intify, user[1].split(",")))
        newUser.append(self.intify(user[2]))
        newUser.append(self.intify(user[3]))
        newUser.append(self.intify(user[4]))
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
        newUser.append(self.intify(user[6]))
        newUser.append(self.intify(user[7]))
        newUser.append(self.intify(user[8]))
        newUser.append(self.intify(user[9]))
        newUser.append(self.intify(user[10]))
        newUser.append(map(self.intify, user[11].split(",")))
        #Check meaning
        if len(newUser[1]) == 1 and newUser[1][0] == -1:
            newUser[1] = []
        if len(newUser[11]) == 1 and newUser[11][0] == -1:
            newUser[11] = []
        return newUser

    def convertItem(self, item):
        newItem = []
        newItem.append(self.intify(item[0]))
        newItem.append(map(self.intify,item[1].split(",")))
        newItem.append(self.intify(item[2]))
        newItem.append(self.intify(item[3]))
        newItem.append(self.intify(item[4]))
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
        newItem.append(self.intify(item[6]))
        newItem.append(self.floatify(item[7]))
        newItem.append(self.floatify(item[8]))
        newItem.append(self.intify(item[9]))
        newItem.append(map(self.intify,item[10].split(",")))
        newItem.append(self.intify(item[11]))
        newItem.append(self.intify(item[12]))
        #Check meaning
        if len(newItem[1]) == 1 and newItem[1][0] == -1:
            newItem[1] = []
        if len(newItem[10]) == 1 and newItem[10][0] == -1:
            newItem[10] = []
        return newItem

    def user_item_index(self, temp_interactions):
        u_i = defaultdict(lambda : defaultdict(int))
        for interact in temp_interactions:
            u_i[interact[0]][interact[1]] += 1
        return u_i

    def item_user_index(self, temp_interactions):
        i_u = defaultdict(lambda : defaultdict(int))
        for interact in temp_interactions:
            i_u[interact[1]][interact[0]] += 1
        return i_u

    def user_item_typed_index(self, temp_interactions):
        u_i = defaultdict(lambda : defaultdict(lambda : [0,0,0]))
        for interact in temp_interactions:
            u_i[interact[0]][interact[1]][interact[2]-1] += 1
        return u_i

    def __init__(self, INTERACTIONS_FILE, ITEMS_FILE, USERS_FILE, TARGET_FILE):
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

        self.interactions = []
        for i in interactions_raw:
            self.interactions.append(self.convertInter(i))

        self.targets = []
        for i in targets_raw:
            self.targets.append(self.convertTarget(i))

        self.items = defaultdict(list)
        for i in items_raw:
            it = self.convertItem(i)
            self.items[it[0]] = it

        self.users = defaultdict(list)
        for u in users_raw:
            us = self.convertUser(u)
            self.users[us[0]] = us

        self.user_item = self.user_item_index(self.interactions)
        self.item_user = self.item_user_index(self.interactions)
        self.user_item_typed = self.user_item_typed_index(self.interactions)
        self.item_active = set([x for x in self.items if self.items[x][12] == 1])
        self.interactions_active = []
        for i in self.interactions:
            if i[1] in self.item_active:
                self.interactions_active.append(i)
