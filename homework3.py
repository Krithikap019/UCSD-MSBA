import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy as np
import string
import random
import string
from sklearn import linear_model
import pandas as pd

from tqdm.notebook import tqdm

import warnings
warnings.filterwarnings("ignore")

def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N
    
def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)
        
def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r
        
answers = {}

allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)
    
len(allRatings)

ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
BooksPerUser = defaultdict(set)
UsersPerBook = defaultdict(set)

for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))
    BooksPerUser[u].add(b)
    UsersPerBook[b].add(u)
    
##################################################
# Read prediction                                #
##################################################

### Question 1
def NegativeEntrySample(dataSet, validSet):    
    validSet['read'] = 1
    NegValidSet = validSet
    userBookDict = {}
    for index, row in dataSet.iterrows():
        if row['userID'] not in userBookDict:
            userBookDict[row['userID']] = {row['bookID']}
        else:
            userBookDict[row['userID']].add(row['bookID'])

    for index, row in validSet.iterrows():
        samNegBookID = random.sample(list(set(dataSet['bookID']) - userBookDict[row['userID']]), 1)[0]
        NegValidSet.loc[len(NegValidSet)] = [row['userID'], samNegBookID, 0, 0]

    return NegValidSet, userBookDict 

allRatings_df = pd.DataFrame(allRatings, columns=['userID', 'bookID', 'rating'])  # Ensure allRatings is in DataFrame format
ratingsValid_df = pd.DataFrame(ratingsValid, columns=['userID', 'bookID', 'rating'])  # Ensure ratingsValid is a DataFrame
ratingsTrain_df = pd.DataFrame(ratingsTrain, columns=['userID', 'bookID', 'rating'])  # Ensure ratingsValid is a DataFrame

NegValidSet, userBookDict = NegativeEntrySample(allRatings_df, ratingsValid_df)

# Copied from baseline code
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalRead/2: break
    
correct1 = 0
for index, row in NegValidSet.iterrows():
    if row['bookID'] in return1:
        correct1 += (row['read'] == 1)
    else:
        correct1 += (row['read'] == 0)

acc1 = correct1/len(NegValidSet)
answers['Q1'] = acc1
assertFloat(answers['Q1'])

### Question 2

threshList = list(np.arange(1, 3.1, 0.1))
accList = []

for k in threshList:
    return2 = set()
    count2 = 0
    for ic, i in mostPopular:
        count2 += ic
        return2.add(i)
        if count2 > totalRead/k: break
        
    correct2 = 0
    for index, row in NegValidSet.iterrows():
        if row['bookID'] in return2:
            correct2 += (row['read'] == 1)
        else:
            correct2 += (row['read'] == 0)
    acc2 = correct2/len(NegValidSet)
    accList.append(acc2)

acc2 = max(accList)
threshold2 = round(threshList[accList.index(acc2)],2)

answers['Q2'] = [threshold2, acc2]
assertFloat(answers['Q2'][0])
assertFloat(answers['Q2'][1])

### Question 3

def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return numer / denom

threshList_Jac = [i*0.001 for i in range(1,11)]
accList_Jac = []
for thresh in threshList_Jac:
    correct3 = 0    
    for _, row in NegValidSet.iterrows():
        Book_row = row['bookID']
        User_row = row['userID']
        Books_u = BooksPerUser[User_row]
        jac = [0]
        for book in Books_u:
            if book == Book_row: continue
            else:
                jac.append(Jaccard(set(UsersPerBook[Book_row]), set(UsersPerBook[book])))
        
        if max(jac) > thresh:
            correct3 += (row['read'] == 1)
        else:
            correct3 += (row['read'] == 0)
    
    accList_Jac.append(correct3 / len(NegValidSet))
acc3 = max(accList_Jac)
threshold_jac = round(threshList_Jac[accList_Jac.index(acc3)],4)
answers['Q3'] = acc3
assertFloat(answers['Q3'])

### Question 4

return4 = set()
count4 = 0
for ic, i in mostPopular:
    count4 += ic
    return4.add(i)
    if count4 > totalRead/threshold2: break

correct4 = 0
for index, row in NegValidSet.iterrows():
    Book_row = row['bookID']
    User_row = row['userID']
    Books_u = BooksPerUser[User_row]
    jac = [0]
    for book in Books_u:
        if book == Book_row: continue
        else:
            jac.append(Jaccard(set(UsersPerBook[Book_row]), set(UsersPerBook[book])))
            
    if max(jac) > threshold_jac and row['bookID'] in return4:
        correct4 += (row['read'] == 1)
    else:
        correct4 += (row['read'] == 0)

acc4 = correct4/len(NegValidSet)
answers['Q4'] = acc4
assertFloat(answers['Q4'])

### Question 5

predictions = open("predictions_Read.csv", 'w')
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    Books_u = BooksPerUser[u]
    jac = []
    jac = [0]
    for book in Books_u:
        if book == Book_row: continue
        else:
            jac.append(Jaccard(set(UsersPerBook[b]), set(UsersPerBook[book])))
            
    if max(jac) > threshold_jac and b in return4:
        predictions.write(u + ',' + b + ",1\n")
    else:
        predictions.write(u + ',' + b + ",0\n")       
predictions.close()

answers['Q5'] = "I confirm that I have uploaded an assignment submission to gradescope"

##################################################
# Rating prediction                              #
##################################################

def prediction(user, item):
    return alpha + userBiases[user] + itemBiases[item]

def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)

def unpack(theta):
    global alpha
    global userBiases
    global itemBiases
    alpha = theta[0]
    userBiases = dict(zip(users, theta[1:nUsers+1]))
    itemBiases = dict(zip(items, theta[1+nUsers:]))
    
def cost(theta, labels, lamb):
    unpack(theta)
    predictions = [prediction(d['userID'], d['bookID']) for index, d in ratingsTrain_df.iterrows()]
    cost = MSE(predictions, labels)
    for u in userBiases:
        cost += lamb*userBiases[u]**2
    for i in itemBiases:
        cost += lamb*itemBiases[i]**2
    return cost

def derivative(theta, labels, lamb):
    unpack(theta)
    N = len(ratingsTrain_df)
    dalpha = 0
    dUserBiases = defaultdict(float)
    dItemBiases = defaultdict(float)
    for index, d in ratingsTrain_df.iterrows():
        u,i = d['userID'], d['bookID']
        pred = prediction(u, i)
        diff = pred - d['rating']
        dalpha += 2/N*diff
        dUserBiases[u] += 2/N*diff
        dItemBiases[i] += 2/N*diff
    for u in userBiases:
        dUserBiases[u] += 2*lamb*userBiases[u]
    for i in itemBiases:
        dItemBiases[i] += 2*lamb*itemBiases[i]
    dtheta = [dalpha] + [dUserBiases[u] for u in users] + [dItemBiases[i] for i in items]
    return np.array(dtheta)

lamb = 1
ratingMean = ratingsTrain_df['rating'].mean()
alpha = ratingMean
labels = ratingsTrain_df['rating']

userBiases = defaultdict(float)
itemBiases = defaultdict(float)

users = list(set(ratingsTrain_df['userID']))
items = list(set(ratingsTrain_df['bookID']))
nUsers = len(users)
nItems = len(items)

scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + [0.0]*(nUsers+nItems), derivative, args = (labels, lamb))

### Question 6

predictions = []
for index, d in ratingsValid_df.iterrows():
    u, i = d['userID'], d['bookID']
    if u in userBiases and i in itemBiases:
        predictions.append(prediction(u, i))
    else:
        predictions.append(0)
        

validMSE = MSE(predictions, ratingsValid_df['rating'])
answers['Q6'] = validMSE
assertFloat(answers['Q6'])

### Question 7

maxUser = max(userBiases, key=userBiases.get)
minUser = min(userBiases, key=userBiases.get)
maxBeta = float(max(userBiases.values()))
minBeta = float(min(userBiases.values()))

answers['Q7'] = [maxUser, minUser, maxBeta, minBeta]
assert [type(x) for x in answers['Q7']] == [str, str, float, float]

### Question 8

lambda_values = [0.01, 0.1, 0.5, 1, 5, 10, 20]
best_lambda = None
best_mse = float('inf')

for lamb in lambda_values:
    initial_theta = [alpha] + [0.0] * (nUsers + nItems)
        
    scipy.optimize.fmin_l_bfgs_b(cost, initial_theta, derivative, args=(labels, lamb))
    
    predictions = []
    for index, d in ratingsValid_df.iterrows():
        u, i = d['userID'], d['bookID']
        if u in userBiases and i in itemBiases:
            predictions.append(prediction(u, i))
        else:
            predictions.append(alpha)           
    current_mse = MSE(predictions, ratingsValid_df['rating'])
    
    if current_mse < best_mse:
        best_mse = current_mse
        best_lambda = lamb

answers['Q8'] = [best_lambda, best_mse]
assertFloat(answers['Q8'][0])
assertFloat(answers['Q8'][1])

predictions8 = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"): # header
        predictions8.write(l)
        continue
    u,b = l.strip().split(',') # Read the user and item from the "pairs" file and write out your prediction
    if u in userBiases and i in itemBiases:
        x = str(float(prediction(u, i)))
        predictions8.write(u + ',' + b + "," + x + "\n")
    else:
        predictions8.write(u + ',' + b + "," + str(0) + "\n")

    # (etc.)
    
predictions8.close()

f = open("answers_hw3.txt", 'w')
f.write(str(answers) + '\n')
f.close()