import json
from collections import defaultdict
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import numpy as np
import random
import gzip
import math
import warnings
warnings.filterwarnings("ignore")

def assertFloat(x): # Checks that an answer is a float
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N
    
f = gzip.open("young_adult_10000.json.gz")
dataset = []
for l in f:
    dataset.append(json.loads(l))
print(len(dataset))
answers = {}

### Question 1
def feature1(datum):
    # your implementation\
    f = datum['review_text'].count('!')
    return [1,f]

def predict_rating_func1(datum, theta):
    r_val = theta[0]
    a = datum['review_text'].count('!')  # number of exclamation marks
    if(len(theta)>1):
        for i in range(1, len(theta)):
            r_val += theta[i] * (a**i)
    return r_val

X_1 = [feature1(datum) for datum in dataset]
Y_1 = [datum['rating'] for datum in dataset]

theta1,residuals,rank,s = np.linalg.lstsq(X_1,Y_1)

predicted_ratings1 = [predict_rating_func1(datum,theta1) for datum in dataset]
actual_ratings1 = [datum['rating'] for datum in dataset]

mse_1 = mean_squared_error(actual_ratings1, predicted_ratings1)

answers['Q1'] = [theta1[0], theta1[1], mse_1]
assertFloatList(answers['Q1'], 3) # Check the format of your answer (three floats)

### Question 2
def feature2(datum):
    # your implementation\
    a = datum['review_text'].count('!')
    b = len(datum['review_text'])
    return [1,b,a]

def predict_rating_func2(datum, theta):
    a = datum['review_text'].count('!')  # number of exclamation marks
    b = len(datum['review_text'])        # length of the review
    return theta[0] + theta[1] * b + theta[2] * a

X_2 = [feature2(datum) for datum in dataset]
Y_2 = [datum['rating'] for datum in dataset]

theta2,residuals,rank,s = np.linalg.lstsq(X_2,Y_2)

predicted_ratings2 = [predict_rating_func2(datum, theta2) for datum in dataset]
actual_ratings2 = [datum['rating'] for datum in dataset]
mse_2 = mean_squared_error(actual_ratings2, predicted_ratings2)

answers['Q2'] = [theta2[0], theta2[1], theta2[2], mse_2]
assertFloatList(answers['Q2'], 4)

## Question 3

def feature3(datum, deg):
    # feature for a specific polynomial degree
    r_val = [1]
    f = datum['review_text'].count('!')
    for d in range(1, deg + 1):
        r_val.append(f**d)
    return r_val

def polynomial_func(deg):
    mse_vals = []
    Y_3 = [datum['rating'] for datum in dataset]
    actual_ratings3 = [datum['rating'] for datum in dataset]
    for d in range(1, deg + 1):
        X_3 = [feature3(datum, d) for datum in dataset]
        theta3,residuals,rank,s = np.linalg.lstsq(X_3,Y_3)
        predicted_ratings3 = [predict_rating_func1(datum, theta3) for datum in dataset]
        mse_3 = mean_squared_error(actual_ratings3, predicted_ratings3)
        mse_vals.append(mse_3)
    return mse_vals

answers['Q3'] = polynomial_func(5)

## Question 4

split_index = int(len(dataset) * 0.5)
train_set = dataset[:split_index]
test_set = dataset[split_index:]

def feature4(train_set, deg):
    # feature for a specific polynomial degree
    r_val = [1]
    f = train_set['review_text'].count('!')
    for d in range(1, deg + 1):
        r_val.append(f**d)
    return r_val

def polynomial_func2(deg):
    mse_vals = []
    Y_4 = [datum['rating'] for datum in train_set]
    actual_ratings4 = [datum['rating'] for datum in test_set]
    for d in range(1, deg + 1):
        X_4 = [feature4(datum, d) for datum in train_set]
        theta4,residuals,rank,s = np.linalg.lstsq(X_4,Y_4)
        predicted_ratings4 = [predict_rating_func1(datum, theta4) for datum in test_set]
        mse_4 = mean_squared_error(actual_ratings4, predicted_ratings4)
        mse_vals.append(mse_4)
    return mse_vals

answers['Q4'] = polynomial_func2(5)

## Question 5

from sklearn.metrics import mean_absolute_error
import numpy as np

Y_test = [datum['rating'] for datum in test_set] 
theta0 = np.mean(Y_test)
theta = np.array([theta0])

predicted_trivial1 = [predict_rating_func1(datum, theta) for datum in test_set]
mae = mean_absolute_error(Y_test, predicted_trivial1)

answers['Q5'] = mae

## Question 6

f = open("beer_50000.json")
dataset2 = []
for l in f:
    if 'user/gender' in l:
        dataset2.append(eval(l))
print(len(dataset2))

X = np.array([b['review/text'].count('!') for b in dataset2]).reshape(-1, 1) 
Y = ['Female' in b['user/gender'] for b in dataset2 ]
mod = linear_model.LogisticRegression(C=1.0)
mod.fit(X,Y)
Y_Pred = mod.predict(X)

def conf_Matrix(Y_Pred, Y):
    TP = len([a for a,b in zip(Y_Pred, Y) if a == True and b == True])
    TN = len([a for a,b in zip(Y_Pred, Y) if a == False and b == False])
    FP = len([a for a,b in zip(Y_Pred, Y) if a == True and b == False])
    FN = len([a for a,b in zip(Y_Pred, Y) if a == False and b == True])
    TPR = (TP/(TP+FN))
    TNR = (TN/(TN+ FP))
    BER = 1 - (TPR+TNR)/2
    return [TP, TN, FP, FN, BER]
answers['Q6'] = conf_Matrix(Y_Pred, Y)
assertFloatList(answers['Q6'], 5)

## Question 7

mod_balanced = linear_model.LogisticRegression(C=1.0, class_weight='balanced')
mod_balanced.fit(X, Y)
Y_Pred_Bal = mod_balanced.predict(X)

answers['Q7'] = conf_Matrix(Y_Pred_Bal, Y)
assertFloatList(answers['Q7'], 5)

## Question 8

precisionList = []
K_values = [1, 10, 100, 1000, 10000]

# Ranking
scores = mod.decision_function(X)
scoreslabels = list(zip(scores,Y))
scoreslabels.sort(reverse =True)
sortedlabels = [y for (x,y) in scoreslabels]

for k in K_values:
    prec = sum(sortedlabels[:k])/k
    precisionList.append(prec)
    
answers['Q8'] = precisionList
assertFloatList(answers['Q8'], 5) #List of five floats

print(answers)

f = open("answers_hw1.txt", 'w') # Write your answers to a file
f.write(str(answers) + '\n')
f.close()