from nltk.featstruct import _trace_unify_identity
from scipy.sparse import data
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# with open('train2.csv', encoding="utf8") as file:
#     train = file.readlines()  #list in element

# with open('test2.csv', encoding="utf8") as file:
    #test = file.readlines()  #list in element


data = pd.read_csv('train2.csv',header=None)
dataTest = pd.read_csv('test2.csv',header=None)
norm = preprocessing.normalize(data)
normTest = preprocessing.normalize(dataTest)
norm = np.delete(norm, -1, axis = 1) 
#print(norm.shape)
lastC = data.iloc[:,-1]  #saves the last column which is the 0,1

rf1 = RandomForestClassifier(n_estimators=1000)
x =rf1.fit(norm,lastC)
#print(x)
predictt = rf1.predict(normTest)
np.savetxt("miner2.txt",predictt, fmt = '%i')


#NOW FOR THE TRAINING SPILT

normD = preprocessing.normalize(data)
#print(normD)
normD = np.delete(normD, -1, axis = 1)
X = normD
Y = lastC
#first splitting the data training 80% test 20%  needed for f1
X_tr, X_test, y_tr, y_test = train_test_split(X, Y, test_size=0.20)
#dT = DecisionTreeClassifier()  #creating decision treeObj

# #doing the actual training on the training set(80)
#dt = dT.fit(X_train,y_train)
# predicts spits what model "thinks" that data is y test(is actual score) x_test is 20
# trainPred = dt.predict(X_test)
# print(trainPred)

# #F1 SCORE LOOK AT PP HIGHER F1 THE BETTER NEVER CAN BE ACC
# flScore = f1_score(y_test, trainPred)
# print(flScore)
#The more trees the better precision
rf = RandomForestClassifier(n_estimators=1000)
rf.fit(X_tr,y_tr) 
predictofy = rf.predict(X_test)
#NEEDED F1 BECAUSE THE DATA IS IMBALANCED BY 20% 
f1Score = f1_score(y_test, predictofy)
print(f1Score)







